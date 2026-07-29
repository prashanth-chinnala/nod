#!/usr/bin/env python3
"""
The M0 model spike as one command.

    !git clone -q https://github.com/prashanth-chinnala/nod.git \
      && python nod/scripts/m0_spike.py

Paste that into any GPU environment -- Colab in a browser, Colab through the VS Code
extension, Kaggle, or a rented box. It prints a JSON block; paste that back.

Why a script and not the notebook: a notebook cannot be linted, type-checked, diffed
usefully, or reviewed, and copying 22 cells between environments loses a cell every time.
This is version-controlled, `ruff`-clean, and one line to invoke. The notebooks remain for
anyone who wants to step through it.

What it does, in order, and it refuses to skip a step:

  1. Report the GPU and the Python version, and stop if there is no GPU.
  2. Decide whether the pinned stack can install on this interpreter.
  3. Clone MuseTalk and install.
  4. Prove every import resolves -- *before* downloading gigabytes.
  5. Download weights, then **audit every checkpoint** and stop if any is bad.
  6. Run inference with the arguments v1.5 actually requires.
  7. Measure the output. Refuse to print an fps number if no video exists.

Steps 4, 5, and 7 are the gates run 1 lacked, and each one corresponds to a way that run
produced plausible-looking numbers that measured nothing. See PROCESS.md 2.2.1.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = "https://github.com/TMElyralab/MuseTalk.git"

# Expected checkpoints and a conservative floor in MB, from MuseTalk's README tree.
# A zero floor means "presence is enough" -- these are small JSON configs.
EXPECTED: dict[str, int] = {
    "models/musetalkV15/unet.pth": 100,
    "models/musetalkV15/musetalk.json": 0,
    "models/musetalk/pytorch_model.bin": 100,
    "models/musetalk/musetalk.json": 0,
    "models/sd-vae/diffusion_pytorch_model.bin": 100,
    "models/sd-vae/config.json": 0,
    "models/whisper/pytorch_model.bin": 50,
    "models/whisper/config.json": 0,
    "models/dwpose/dw-ll_ucoco_384.pth": 100,
    "models/face-parse-bisent/79999_iter.pth": 30,
    "models/face-parse-bisent/resnet18-5c106cde.pth": 30,
}

IMPORTS = ("torch", "mmcv", "mmpose", "mmdet", "diffusers", "transformers", "omegaconf")

VRAM_FLOOR_MIB = 500
"""
Below this, inference did not touch the GPU.

Run 1 reported a peak of 3 MiB alongside a 15.43-second "warm inference" time. The time
was real; it measured how long the process took to fail. This floor is what turns that into
an explicit verdict instead of a number someone might quote.
"""


@dataclass
class Spike:
    log: dict[str, object] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        stamp = time.strftime("%H:%M")
        self.notes.append(f"{stamp}  {message}")
        print(f"  note: {message}", flush=True)


def run(cmd: str, *, quiet: bool = False, tail: int = 2500) -> subprocess.CompletedProcess[str]:
    if not quiet:
        print(f"$ {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout and not quiet:
        print(result.stdout[-tail:], flush=True)
    if result.returncode != 0 and not quiet:
        print(f"STDERR: {(result.stderr or '')[-tail:]}", file=sys.stderr, flush=True)
    return result


def timed(cmd: str, label: str) -> dict[str, object]:
    """Run a command, capturing wall clock and peak *device* VRAM.

    Sampled from `nvidia-smi` in a thread rather than `torch.cuda.max_memory_allocated`,
    because inference runs as a subprocess and an in-process reading would be zero. It is
    slightly pessimistic -- it includes anything else resident on the GPU.
    """
    samples: list[int] = []
    stop = threading.Event()

    def poll() -> None:
        query = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        while not stop.is_set():
            got = subprocess.run(query, shell=True, capture_output=True, text=True)
            if got.returncode == 0 and got.stdout.strip():
                samples.append(int(got.stdout.strip().splitlines()[0]))
            time.sleep(0.25)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    print(f"--- {label} ---", flush=True)
    started = time.perf_counter()
    result = run(cmd)
    elapsed = time.perf_counter() - started
    stop.set()
    watcher.join(timeout=2)

    record: dict[str, object] = {
        "seconds": round(elapsed, 2),
        "peak_vram_mib": max(samples) if samples else None,
        "exit_code": result.returncode,
    }
    print(
        f"{label}: {record['seconds']}s, peak VRAM {record['peak_vram_mib']} MiB, "
        f"exit {record['exit_code']}",
        flush=True,
    )
    return record


def audit_weights() -> list[str]:
    """Report every expected checkpoint; return the bad ones.

    Catches the two silent-corruption modes that produce a file of the wrong kind rather
    than no file: a git-lfs pointer (130 bytes of text) and a Google Drive quota page
    (HTML). Both would otherwise be loaded as a checkpoint and fail obscurely.
    """
    bad: list[str] = []
    width = max(len(name) for name in EXPECTED)
    for name, floor_mb in EXPECTED.items():
        path = Path(name)
        if not path.is_file():
            status, ok = "ABSENT", False
        else:
            head = path.open("rb").read(64)
            megabytes = path.stat().st_size / 1e6
            if head.startswith(b"version https://git-lfs"):
                status, ok = "GIT-LFS POINTER", False
            elif head.lstrip()[:14].lower() in (
                b"<!doctype html",
                b"<html>",
            ) or head.lstrip().startswith(b"<html"):
                status, ok = "HTML (Drive quota page)", False
            elif megabytes < floor_mb:
                status, ok = f"{megabytes:.1f}MB TOO SMALL (expect >{floor_mb}MB)", False
            else:
                status, ok = f"{megabytes:.1f}MB ok", True
        print(f"  {'  ' if ok else '!!'} {name:<{width}}  {status}", flush=True)
        if not ok:
            bad.append(name)
    return bad


def probe(path: Path) -> dict[str, object]:
    query = (
        "ffprobe -v error -select_streams v:0 -show_entries "
        "stream=width,height,nb_frames,duration -of json "
        f'"{path}"'
    )
    got = run(query, quiet=True)
    if got.returncode != 0:
        return {"error": got.stderr.strip()[-300:]}
    stream = json.loads(got.stdout)["streams"][0]
    return {
        "resolution": f"{stream.get('width')}x{stream.get('height')}",
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        "duration_s": round(float(stream["duration"]), 2) if stream.get("duration") else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", default="/content/m0", help="scratch directory")
    parser.add_argument("--skip-install", action="store_true", help="reuse an existing install")
    parser.add_argument(
        "--allow-version-drift",
        action="store_true",
        help="install newest mmcv/mmpose instead of the pinned ones (Route B)",
    )
    args = parser.parse_args()

    spike = Spike()
    log, note = spike.log, spike.note

    # -- 1. hardware ------------------------------------------------------
    gpu = run(
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
        quiet=True,
    )
    if gpu.returncode != 0:
        print(
            "No GPU visible. On Colab: Runtime -> Change runtime type -> T4 GPU.",
            file=sys.stderr,
        )
        return 2
    name, vram, driver = (f.strip() for f in gpu.stdout.strip().split(","))
    log["gpu"] = {"name": name, "vram_total": vram, "driver": driver}
    log["python"] = sys.version.split()[0]
    print(f"GPU: {name}, {vram}, driver {driver}\nPython: {log['python']}\n", flush=True)

    if shutil.which("ffmpeg") is None:
        note("ffmpeg missing; installing (MuseTalk requires it at inference time)")
        run("apt-get -qq install -y ffmpeg")
    ffmpeg = shutil.which("ffmpeg")
    log["ffmpeg"] = ffmpeg
    ffmpeg_dir = str(Path(ffmpeg).parent) if ffmpeg else "/usr/bin"

    # -- 2. can the pinned stack install here? ----------------------------
    pinned_ok = sys.version_info[:2] == (3, 10)
    log["python_matches_pin"] = pinned_ok
    if not pinned_ok and not args.allow_version_drift:
        note(
            f"MuseTalk pins Python 3.10; this is {log['python']}. mmcv==2.0.1 has no wheel "
            "for it -- that is why run 1's pip install finished in 13 seconds"
        )
        print(
            "\nTwo options:\n"
            "  Route A (recommended): use notebooks/m0_musetalk_v2.ipynb, which installs\n"
            "    conda and builds a real Python 3.10 environment.\n"
            "  Route B (faster, may fail): re-run this with --allow-version-drift to try\n"
            "    the newest mmcv/mmpose that have wheels for this interpreter.\n",
            file=sys.stderr,
        )
        return 3

    # -- 3. clone and install --------------------------------------------
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    checkout = work / "MuseTalk"
    if not checkout.exists():
        started = time.perf_counter()
        if run(f"git clone --depth 1 {REPO} {checkout}").returncode != 0:
            note("clone failed")
            return 4
        log["clone_s"] = round(time.perf_counter() - started, 1)
    import os

    os.chdir(checkout)
    log["commit"] = run("git rev-parse --short HEAD", quiet=True).stdout.strip()
    print(f"MuseTalk at {log['commit']}\n", flush=True)

    if not args.skip_install:
        started = time.perf_counter()
        run("pip install -q -r requirements.txt 2>&1 | tail -5")
        run('pip install -q -U openmim "huggingface_hub[cli]" gdown 2>&1 | tail -3')
        run("mim install mmengine 2>&1 | tail -3")
        if args.allow_version_drift:
            run("mim install mmcv mmdet mmpose 2>&1 | tail -5")
        else:
            run('mim install "mmcv==2.0.1" 2>&1 | tail -5')
            run('mim install "mmdet==3.1.0" "mmpose==1.1.0" 2>&1 | tail -3')
        log["install_s"] = round(time.perf_counter() - started, 1)

    # -- 4. gate on imports BEFORE downloading gigabytes -----------------
    print("\n--- imports ---", flush=True)
    missing = []
    for module in IMPORTS:
        got = run(f'python -c "import {module}; print({module}.__version__)"', quiet=True)
        version = got.stdout.strip() if got.returncode == 0 else "MISSING"
        print(f"  {module:14} {version}", flush=True)
        if got.returncode != 0:
            missing.append(module)
    log["missing_imports"] = missing
    if missing:
        note(f"imports unresolved: {missing}")
        print(
            "\nStopping before the weight download -- there is no point fetching several "
            "GB for a stack that cannot import. Take Route A.",
            file=sys.stderr,
        )
        log["setup_notes"] = spike.notes
        print("=" * 70 + f"\n{json.dumps(log, indent=2, default=str)}\n" + "=" * 70)
        return 5

    # -- 5. weights, then audit ------------------------------------------
    started = time.perf_counter()
    script = next(
        (
            s
            for s in ("download_weights.sh", "scripts/download_weights.sh")
            if Path(s).is_file()
        ),
        None,
    )
    if script is None:
        note("no download_weights.sh in this clone; check the README")
    else:
        # Force the real Hugging Face endpoint: the bundled script points at a mirror that
        # is often unreachable from Colab, and its failures do not change the exit code.
        run(f"HF_ENDPOINT=https://huggingface.co bash {script} 2>&1 | tail -20")
    log["weights_s"] = round(time.perf_counter() - started, 1)
    log["weights_on_disk"] = (
        run("du -sh models", quiet=True).stdout.split()[0] if Path("models").exists() else "0"
    )
    print(f"\ndownload {log['weights_s']}s, models/ is {log['weights_on_disk']}\n", flush=True)

    bad = audit_weights()
    log["bad_weights"] = bad
    if bad:
        note(f"{len(bad)} checkpoint(s) missing or corrupt")
        print(
            f"\nStopping. {len(bad)} checkpoint(s) bad -- see the !! rows.\n"
            "Running inference now would produce a timing that measures a crash, and that\n"
            "timing looks plausible enough to be quoted by mistake.\n"
            "  Re-run once: transient Hugging Face rate limits are common.\n"
            "  GIT-LFS POINTER -> apt-get install -y git-lfs && git lfs install\n"
            "  HTML            -> the gdown Drive link hit its quota; fetch 79999_iter.pth\n"
            "                     by hand into models/face-parse-bisent/\n",
            file=sys.stderr,
        )
        log["setup_notes"] = spike.notes
        print("=" * 70 + f"\n{json.dumps(log, indent=2, default=str)}\n" + "=" * 70)
        return 6
    print("\nall checkpoints present and plausibly sized.\n", flush=True)

    # -- 6. inference, with the arguments v1.5 actually needs -------------
    base = (
        "python -m scripts.inference "
        "--inference_config configs/inference/test.yaml "
        "--unet_model_path models/musetalkV15/unet.pth "
        "--unet_config models/musetalkV15/musetalk.json "
        "--version v15 "
        f"--ffmpeg_path {ffmpeg_dir} "
    )
    log["inference_cold"] = timed(base + "--result_dir ./results/cold", "inference (cold)")
    if log["inference_cold"]["exit_code"] == 0:  # type: ignore[index]
        log["inference_warm"] = timed(base + "--result_dir ./results/warm", "inference (warm)")
    else:
        note("cold inference failed; read the stderr above. No fps number will be printed")

    # -- 7. measure, and refuse to invent ---------------------------------
    outputs = sorted(Path("results").rglob("*.mp4")) if Path("results").exists() else []
    if not outputs:
        note("no output video -- nothing proven; do not record any fps number")
    else:
        newest = max(outputs, key=lambda p: p.stat().st_mtime)
        log["output"] = {"path": str(newest), **probe(newest)}
        warm = log.get("inference_warm") or log.get("inference_cold")
        frames = log["output"].get("frames")  # type: ignore[union-attr]
        duration = log["output"].get("duration_s")  # type: ignore[union-attr]
        seconds = warm.get("seconds") if isinstance(warm, dict) else None
        if frames and seconds:
            log["effective_fps"] = round(frames / float(seconds), 2)
        if duration and seconds:
            ratio = round(float(seconds) / duration, 2)
            log["realtime_ratio"] = ratio
            log["faster_than_realtime"] = ratio < 1.0

    cold = log.get("inference_cold") or {}
    ran = (
        isinstance(cold, dict)
        and cold.get("exit_code") == 0
        and bool(log.get("output"))
        and (cold.get("peak_vram_mib") or 0) > VRAM_FLOOR_MIB
    )
    log["inference_actually_ran"] = ran
    log["setup_notes"] = spike.notes
    log["finished_at"] = time.strftime("%Y-%m-%d %H:%M")

    print("\n" + "=" * 70)
    print(json.dumps(log, indent=2, default=str))
    print("=" * 70)
    if ran:
        print(
            "\nInference ran: exit 0, an output file exists, and VRAM passed "
            f"{VRAM_FLOOR_MIB} MiB. These numbers are real -- paste the block back."
        )
    else:
        print(
            "\nInference did NOT run. Every timing above measures a failure, not a "
            "model. Paste the block back anyway; setup_notes is the finding."
        )
    return 0 if ran else 1


if __name__ == "__main__":
    raise SystemExit(main())
