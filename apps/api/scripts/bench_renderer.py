#!/usr/bin/env python3
"""
Measure the real renderer, on whatever device this machine has.

**Why a script rather than a note in the commit message.** Three numbers currently in this
repo's docstrings were measured on Apple Silicon MPS, and two of them are almost certainly
wrong on CUDA:

  * float16 was 9.15x faster than float32 here. Plausible anywhere, but the *size* of the win
    came partly from MPS running float32 badly.
  * Batch 3-4 beat batch 8, and batch 32 was 5x worse than batch 3. That is a memory-bandwidth
    result on 16 GB of unified memory. On a card with its own VRAM the curve should look like
    the textbook one -- bigger batches better, up to a point -- and carrying the MPS default
    onto a GPU would silently leave most of the card idle.

So this re-measures rather than assuming, prints the device alongside every figure, and
sweeps the two axes that mattered. Nothing here is compared against a stored number: it
reports what this machine did, and a human reads both.

**What it does not do.** No database, no web console, no SFU, no session. Those all work and
are tested elsewhere; including them here would mean a GPU measurement blocked on provisioning
Postgres. This is the renderer and the renderer only.

    python scripts/bench_renderer.py                            # a bundled demo face
    python scripts/bench_renderer.py --reference media/x.mp4    # a real uploaded reference
    python scripts/bench_renderer.py --json bench.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TARGET_MS = 40.0
"""The per-frame budget at 25fps. Every result is reported against this."""

BATCHES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
"""
Batch sizes to sweep.

Wider than the useful range on purpose. The interesting part of the MPS curve was the
*degradation* after 4, which a sweep of (4, 8, 16) would have shown as noise. If a GPU turns
out to keep improving to 32, that is worth knowing too, and the run costs seconds.
"""

WARMUP = 2
RUNS = 4


def device_report() -> dict[str, Any]:
    """Everything about the machine that a reader needs to judge the numbers."""
    import torch

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info |= {
            "gpu": props.name,
            "vram_gb": round(props.total_memory / 1024**3, 1),
            "capability": f"{props.major}.{props.minor}",
            "cuda": torch.version.cuda,
        }
    return info


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def time_stages(backend: Any, identity: dict[str, Any], features: list[Any]) -> dict[str, float]:
    """
    Per-stage ms/frame, so a slow result names its own cause.

    The stages are timed separately rather than as one total because they scale differently:
    the U-Net and VAE are fixed 256x256 work, while blending and JPEG scale with the *output*
    frame size. A total alone cannot tell those apart, which is how "downscale the reference"
    got proposed as a throughput fix when it would not have touched the dominant cost.
    """
    import cv2
    import numpy as np
    import torch
    from musetalk.utils.blending import get_image_blending

    device = backend.device
    unet, vae, pe = backend._models["unet"], backend._models["vae"], backend._models["pe"]
    dtype = backend._models["dtype"]
    size = len(features)
    indices = [i % len(identity["latents"]) for i in range(size)]

    stages: dict[str, list[float]] = {"pe": [], "unet": [], "vae_decode": [], "blend": [], "jpeg": []}
    for run in range(WARMUP + RUNS):
        with torch.no_grad():
            audio = torch.stack([torch.as_tensor(f) for f in features]).to(device, dtype)
            latents = torch.cat([identity["latents"][i] for i in indices]).to(
                device, unet.model.dtype
            )
            _sync(device)
            start = time.perf_counter()
            embeddings = pe(audio)
            _sync(device)
            pe_s = time.perf_counter() - start

            start = time.perf_counter()
            predicted = unet.model(
                latents, torch.tensor([0], device=device), encoder_hidden_states=embeddings
            ).sample
            _sync(device)
            unet_s = time.perf_counter() - start

            start = time.perf_counter()
            decoded = vae.decode_latents(predicted)
            _sync(device)
            vae_s = time.perf_counter() - start

        start = time.perf_counter()
        blended = []
        for i, face in zip(indices, decoded, strict=False):
            x1, y1, x2, y2 = identity["coords"][i]
            resized = cv2.resize(np.asarray(face).astype(np.uint8), (x2 - x1, y2 - y1))
            blended.append(
                get_image_blending(
                    identity["frames"][i], resized, [x1, y1, x2, y2],
                    identity["masks"][i], identity["mask_boxes"][i],
                )
            )
        blend_s = time.perf_counter() - start

        start = time.perf_counter()
        for image in blended:
            backend._encode(image)
        jpeg_s = time.perf_counter() - start

        if run < WARMUP:
            continue
        for key, value in (
            ("pe", pe_s), ("unet", unet_s), ("vae_decode", vae_s),
            ("blend", blend_s), ("jpeg", jpeg_s),
        ):
            stages[key].append(value / size * 1000)

    return {key: round(statistics.median(values), 1) for key, values in stages.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        default=str(ROOT / "vendor" / "MuseTalk" / "assets" / "demo" / "man" / "man.png"),
        help="reference clip or image; defaults to a bundled demo face",
    )
    parser.add_argument("--json", help="also write the full result here")
    parser.add_argument(
        "--batches",
        default=",".join(str(b) for b in BATCHES),
        help="comma-separated batch sizes to sweep",
    )
    parser.add_argument(
        "--precisions",
        default="float16,float32",
        help="comma-separated; float32 is worth one run to confirm fp16 is free",
    )
    args = parser.parse_args()

    import torch

    from avatar.renderers.musetalk_torch import TorchMuseTalkBackend

    info = device_report()
    print(json.dumps(info, indent=2))
    if not info["cuda_available"]:
        print(
            "\n!! No CUDA device. Every number below is this device's, not a GPU's, and must "
            "be reported with the device attached to it.\n"
        )

    result: dict[str, Any] = {"device_info": info, "reference": args.reference, "precisions": {}}
    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    for precision in [p.strip() for p in args.precisions.split(",") if p.strip()]:
        # Rebuilt per precision rather than cast in place: `VAE` keeps its own `_use_float16`
        # flag that its encode path reads, so a module cast from outside would leave that lying
        # and the latents would be prepared in the wrong dtype.
        import os

        os.environ["AVATAR_MUSETALK_FP16"] = "1" if precision == "float16" else "0"

        backend = TorchMuseTalkBackend()
        start = time.perf_counter()
        backend.load()
        load_s = round(time.perf_counter() - start, 1)

        start = time.perf_counter()
        identity = backend.prepare(args.reference)
        prepare_s = round(time.perf_counter() - start, 1)
        usable = identity["usable_frames"]

        features = backend._audio_features(_silence(seconds=3))
        if not features:
            print("!! no audio features produced; cannot measure", file=sys.stderr)
            return 1

        entry: dict[str, Any] = {
            "dtype": str(backend._models["dtype"]),
            "load_s": load_s,
            "prepare_s": prepare_s,
            "prepare_ms_per_frame": round(prepare_s / max(usable, 1) * 1000),
            "usable_frames": usable,
            "source_frames": identity["source_frames"],
            "batches": {},
        }
        print(
            f"\n=== {precision} on {backend.device} ===\n"
            f"load {load_s}s   prepare {prepare_s}s for {usable}/{identity['source_frames']} "
            f"frames ({entry['prepare_ms_per_frame']} ms/frame)\n"
        )
        header = f"{'batch':>6}  {'pe':>6} {'unet':>7} {'vae':>7} {'blend':>7} {'jpeg':>6}"
        print(f"{header}  {'total':>8} {'fps':>7}  vs 40ms")

        for size in batches:
            if size > len(features):
                continue
            backend.batch_size = size
            try:
                stages = time_stages(backend, identity, features[:size])
            except torch.OutOfMemoryError:
                # Recorded, not fatal. A batch that does not fit is a real result about this
                # card -- and dying here would throw away every smaller batch already measured,
                # which is the part anyone actually needs.
                torch.cuda.empty_cache()
                entry["batches"][size] = {"oom": True}
                print(f"{size:>6}  out of memory on {info.get('gpu', backend.device)}")
                continue
            total = round(sum(stages.values()), 1)
            fps = round(1000 / total, 2) if total else 0.0
            entry["batches"][size] = stages | {
                "total_ms_per_frame": total,
                "fps": fps,
                "x_over_budget": round(total / TARGET_MS, 1),
            }
            print(
                f"{size:>6}  {stages['pe']:>6.1f} {stages['unet']:>7.1f} "
                f"{stages['vae_decode']:>7.1f} {stages['blend']:>7.1f} {stages['jpeg']:>6.1f}"
                f"  {total:>8.1f} {fps:>7.2f}  {total / TARGET_MS:>5.1f}x"
            )

        measured = {b: v for b, v in entry["batches"].items() if "total_ms_per_frame" in v}
        if not measured:
            print("\n!! every batch size ran out of memory; nothing measured")
            result["precisions"][precision] = entry
            continue
        best = min(measured, key=lambda b: measured[b]["total_ms_per_frame"])
        entry["best_batch"] = best
        entry["best"] = measured[best]
        print(
            f"\nbest: batch {best} at {entry['best']['total_ms_per_frame']} ms/frame "
            f"= {entry['best']['fps']} fps = {entry['best']['x_over_budget']}x the 40 ms budget"
        )
        if entry["best"]["fps"] >= 25:
            print("      -> clears 25fps. Realtime on this device.")
        else:
            print(f"      -> does NOT clear 25fps. Short by {25 / entry['best']['fps']:.1f}x.")

        result["precisions"][precision] = entry
        backend.unload()
        if info["cuda_available"]:
            torch.cuda.empty_cache()

    fp16 = result["precisions"].get("float16", {}).get("best", {}).get("total_ms_per_frame")
    fp32 = result["precisions"].get("float32", {}).get("best", {}).get("total_ms_per_frame")
    if fp16 and fp32:
        result["fp16_speedup"] = round(fp32 / fp16, 2)
        print(f"\nfloat16 is {result['fp16_speedup']}x float32 on this device.")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")
    return 0


def _silence(*, seconds: float) -> bytes:
    """
    Silent PCM, 16 kHz mono s16le.

    Silence is legitimate here and worth defending: this measures the *cost* of a forward pass,
    which is a function of tensor shapes, not of what the audio says. Using real speech would
    make the benchmark depend on a bundled audio file without changing any timing. It would be
    the wrong choice for measuring output *quality*, which this script does not do.
    """
    return b"\x00\x00" * int(16_000 * seconds)


if __name__ == "__main__":
    sys.exit(main())
