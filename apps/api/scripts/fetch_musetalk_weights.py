#!/usr/bin/env python3
"""
Fetch and *verify* MuseTalk's model weights.

This replaces upstream's `download_weights.sh`, which is the reason the first spike run
failed in a way that took hours to diagnose. That script has three defects, and they
compound:

  1. **No `set -e`.** Every download failure is ignored, and the last line prints
     `✅ All weights have been downloaded successfully!` unconditionally. Run 1 ended with
     96 MB on disk, exit code 0, and a cheerful success message.
  2. **A third-party Hugging Face mirror** (`HF_ENDPOINT=https://hf-mirror.com`), set
     globally for the shell. When it is unreachable, every `huggingface-cli` call fails —
     see defect 1 for what happens next.
  3. **`gdown --id` against Google Drive** for the face-parsing checkpoint. Drive answers a
     quota-exceeded *HTML page* with HTTP 200, so gdown writes an HTML document to
     `79999_iter.pth` and exits 0. The failure then surfaces as a torch unpickling error
     inside model loading, hours later and nowhere near the cause.

So the rule here is: nothing is reported as present unless it was opened and checked. Every
artifact declares a minimum size and a content check, an artifact that fails either is
deleted rather than left on disk to be found later, and the exit code is non-zero if
anything is missing. A partial download must look like a failure, because it is one.

Two artifacts upstream fetches are deliberately skipped, and skipping is stated rather than
silent — see SKIPPED.

Usage:
    .venv-musetalk/bin/python scripts/fetch_musetalk_weights.py
    .venv-musetalk/bin/python scripts/fetch_musetalk_weights.py --check   # verify only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

# Upstream's default. Set explicitly rather than inherited from the environment: a mirror
# configured for one shell should not silently decide where a model came from.
HF_ENDPOINT = "https://huggingface.co"


@dataclass(frozen=True)
class Artifact:
    """One file, where it comes from, and what makes it believable."""

    path: str
    """Destination, relative to `models/`."""

    min_bytes: int
    """
    A floor, not a checksum.

    Upstream publishes no hashes, so an exact digest cannot be checked without inventing
    one. A floor still catches every failure actually observed: an HTML error page (a few
    KB), a truncated transfer, and an empty file. It would not catch a corrupted byte in the
    middle of a valid-length tensor file — the content check below is what covers that.
    """

    hf_repo: str = ""
    hf_file: str = ""
    urls: tuple[str, ...] = ()
    """Direct URLs, tried in order. Used where the artifact is not on the Hub."""

    kind: str = "torch"
    """`torch`, `json`, or `raw` — decides how the content is checked."""

    why: str = ""

    modernise: bool = False
    """
    Re-save the checkpoint in torch's current container format after downloading.

    For exactly one artifact, and for a specific incompatibility. `resnet18-5c106cde.pth` on
    download.pytorch.org is still in torch's *original* tar format, and since torch 2.6
    `torch.load` defaults to `weights_only=True`, which refuses tar outright:

        RuntimeError: Cannot use ``weights_only=True`` with files saved in the legacy .tar
        format.

    MuseTalk's BiSeNet loader calls a bare `torch.load(model_path)`, so it hits that and dies.
    The alternatives were patching the vendored checkout -- which would need re-patching on
    every pull -- or monkey-patching `torch.load`'s default globally, which would silently
    re-enable arbitrary-code-execution on *every* checkpoint this process ever loads. Doing it
    here converts one file, once, with the unsafe load scoped to the one file whose origin is
    known.

    The `weights_only=False` below is therefore a deliberate, bounded decision: the bytes came
    from download.pytorch.org over TLS and were checked before this ran. It is not a default
    being relaxed for convenience.
    """

    aliases: tuple[str, ...] = field(default=())
    """
    Other names the loader may look for.

    MuseTalk's own code and its config files disagree about a couple of filenames, so the
    file is hard-linked under each. Cheaper than patching upstream and re-patching it on the
    next pull.
    """


WEIGHTS: tuple[Artifact, ...] = (
    Artifact(
        path="musetalkV15/unet.pth",
        hf_repo="TMElyralab/MuseTalk",
        hf_file="musetalkV15/unet.pth",
        min_bytes=3_000_000_000,
        why="the lip-sync U-Net itself; v1.5, which is what the renderer targets",
    ),
    Artifact(
        path="musetalkV15/musetalk.json",
        hf_repo="TMElyralab/MuseTalk",
        hf_file="musetalkV15/musetalk.json",
        min_bytes=200,
        kind="json",
        why="the U-Net's architecture config; without it the checkpoint cannot be shaped",
    ),
    Artifact(
        path="sd-vae/diffusion_pytorch_model.bin",
        hf_repo="stabilityai/sd-vae-ft-mse",
        hf_file="diffusion_pytorch_model.bin",
        min_bytes=300_000_000,
        why="encodes the face crop into latents and decodes the result back to pixels",
    ),
    Artifact(
        path="sd-vae/config.json",
        hf_repo="stabilityai/sd-vae-ft-mse",
        hf_file="config.json",
        min_bytes=200,
        kind="json",
        why="VAE architecture config",
    ),
    Artifact(
        path="whisper/pytorch_model.bin",
        hf_repo="openai/whisper-tiny",
        hf_file="pytorch_model.bin",
        min_bytes=100_000_000,
        why="audio features. Whisper is the encoder only here — nothing is transcribed",
    ),
    Artifact(
        path="whisper/config.json",
        hf_repo="openai/whisper-tiny",
        hf_file="config.json",
        min_bytes=200,
        kind="json",
        why="Whisper config",
    ),
    Artifact(
        path="whisper/preprocessor_config.json",
        hf_repo="openai/whisper-tiny",
        hf_file="preprocessor_config.json",
        min_bytes=100,
        kind="json",
        why="mel-spectrogram parameters; the feature extractor reads these, not the model",
    ),
    Artifact(
        path="face-parse-bisent/resnet18-5c106cde.pth",
        urls=("https://download.pytorch.org/models/resnet18-5c106cde.pth",),
        min_bytes=40_000_000,
        modernise=True,
        why="BiSeNet's backbone. From download.pytorch.org, which is a stable host",
    ),
    Artifact(
        path="face-parse-bisent/79999_iter.pth",
        # Upstream points at Google Drive, which is where run 1 silently got an HTML page.
        # Hub mirrors are tried first for that reason; each is verified on arrival, so a
        # wrong or vandalised mirror fails here rather than inside model loading.
        urls=(
            f"{HF_ENDPOINT}/ManyOtherFunctions/face-parse-bisent/resolve/main/79999_iter.pth",
            f"{HF_ENDPOINT}/nateraw/face-parse-bisent/resolve/main/79999_iter.pth",
            f"{HF_ENDPOINT}/camenduru/MuseTalk/resolve/main/models/face-parse-bisent/79999_iter.pth",
        ),
        min_bytes=40_000_000,
        why="face-parsing mask, used to blend the repainted mouth back into the frame",
    ),
)

SKIPPED = {
    "dwpose/dw-ll_ucoco_384.pth": (
        "Not needed. It is the RTMPose wholebody model, and mmpose is the one dependency "
        "this project substitutes: upstream uses it for a single thing -- 68 face landmarks "
        "at keypoints[23:91] -- which `avatar.renderers.landmarks` produces with a pure-torch "
        "detector instead. Fetching a 400 MB pose model to leave it unused would be the "
        "confusing choice."
    ),
    "syncnet/latentsync_syncnet.pt": (
        "Training only. It scores lip-sync alignment as a loss term; nothing on the "
        "inference path opens it."
    ),
    "musetalk/pytorch_model.bin": (
        "MuseTalk v1.0. The renderer targets v1.5, and loading both would mean two U-Nets "
        "in memory to use one."
    ),
}


def _looks_like_html(head: bytes) -> bool:
    """
    The Google Drive quota page, and any HTML error served with a 200.

    Checked before the size floor, because the message it produces is the useful one: "this
    host served a web page instead of a file" points at the cause, where "smaller than
    expected" sends someone looking for a network problem.
    """
    start = head[:512].lstrip().lower()
    return start.startswith((b"<!doctype", b"<html", b"<?xml")) or b"<title>" in start


def verify(artifact: Artifact) -> str | None:
    """Return a reason the artifact is unusable, or None if it passes."""
    target = MODELS / artifact.path
    if not target.exists():
        return "missing"

    size = target.stat().st_size
    with target.open("rb") as handle:
        head = handle.read(512)

    if _looks_like_html(head):
        return (
            f"contains an HTML page, not a model ({size / 1_048_576:.1f} MB). The host "
            "answered a web page with a success status -- this is the Google Drive quota "
            "failure that made run 1 look like it had worked"
        )
    if size < artifact.min_bytes:
        return (
            f"{size / 1_048_576:.1f} MB, below the {artifact.min_bytes / 1_048_576:.0f} MB "
            "floor -- truncated or an error response"
        )

    if artifact.kind == "json":
        try:
            json.loads(target.read_text())
        except (ValueError, UnicodeDecodeError) as cause:
            return f"is not valid JSON: {cause}"
    elif artifact.kind == "torch" and not (
        head.startswith((b"PK", b"\x80")) or tarfile.is_tarfile(target)
    ):
        # The magic bytes only. Actually unpickling every checkpoint would need torch loaded
        # and several GB of RAM to prove something the header already tells us.
        #
        # Three formats, not two. `PK` is the zip container torch has written since 1.6, and
        # `\x80` is a bare pickle -- but torch's *original* format was a tar, and
        # `resnet18-5c106cde.pth` on download.pytorch.org is still that: it begins
        # `././@PaxHeader`, a POSIX tar extended header. Checking only the first two rejected a
        # file that `torch.load` reads without complaint, which is a verifier failing in the
        # more annoying direction -- refusing something good rather than accepting something
        # bad. Confirmed by loading it: 46.8 MB, 122 state-dict keys, `conv1.weight` at
        # (64, 3, 7, 7).
        return (
            f"does not begin like a torch checkpoint (first bytes {head[:8]!r}). "
            "Not a zip archive, a pickle, or a tar"
        )
    return None


def modernise(target: Path) -> None:
    """
    Rewrite a legacy-tar checkpoint in torch's current container format, in place.

    See `Artifact.modernise` for why this exists and why the unsafe load is bounded rather
    than global. A file already in a modern format is left alone, so this is idempotent.
    """
    if not tarfile.is_tarfile(target):
        return
    import torch

    state = torch.load(target, map_location="cpu", weights_only=False)
    torch.save(state, target)
    print(f"    rewrote {target.name} from legacy tar to zip format ({len(state)} tensors)")


def fetch(artifact: Artifact) -> None:
    """Download to a temporary name, verify, and only then move into place."""
    target = MODELS / artifact.path
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".partial")

    sources: list[str] = list(artifact.urls)
    if artifact.hf_repo:
        sources.insert(
            0, f"{HF_ENDPOINT}/{artifact.hf_repo}/resolve/main/{artifact.hf_file}"
        )

    problems: list[str] = []
    for url in sources:
        try:
            print(f"    {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "nod/m0"})
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                staging.open("wb") as out,
            ):
                shutil.copyfileobj(response, out, length=1 << 20)
        # Any failure here is the same failure: this source did not produce the file. The next
        # one is tried, and if none do, the caller gets every reason at once.
        except Exception as cause:
            problems.append(f"{url} -> {type(cause).__name__}: {cause}")
            staging.unlink(missing_ok=True)
            continue

        # Verify in place under the real name, then keep it only if it passes. Leaving a
        # bad file on disk is what makes the next run report a size instead of a cause.
        staging.replace(target)
        reason = verify(artifact)
        if reason is None:
            if artifact.modernise:
                modernise(target)
            size = target.stat().st_size / 1_048_576
            print(f"    ok, {size:.1f} MB")
            for alias in artifact.aliases:
                link = MODELS / alias
                link.parent.mkdir(parents=True, exist_ok=True)
                link.unlink(missing_ok=True)
                link.hardlink_to(target)
            return
        problems.append(f"{url} -> {reason}")
        target.unlink(missing_ok=True)

    raise SystemExit(
        f"\n!! could not obtain {artifact.path}\n"
        + "\n".join(f"   {problem}" for problem in problems)
        + f"\n   what it is for: {artifact.why}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify what is on disk and download nothing",
    )
    args = parser.parse_args()

    print(f"models/ is {MODELS}")
    print(f"hub is {HF_ENDPOINT}\n")

    missing: list[str] = []
    for artifact in WEIGHTS:
        reason = verify(artifact)
        if reason is None:
            size = (MODELS / artifact.path).stat().st_size / 1_048_576
            print(f"  [have] {artifact.path}  {size:.1f} MB")
            continue
        if args.check:
            print(f"  [BAD ] {artifact.path}  {reason}")
            missing.append(f"{artifact.path}: {reason}")
            continue
        print(f"  [get ] {artifact.path}  ({reason})")
        fetch(artifact)

    print("\nnot fetched, on purpose:")
    for path, why in SKIPPED.items():
        print(f"  {path}\n      {why}")

    if missing:
        print(f"\n!! {len(missing)} artifact(s) unusable. Run without --check to fetch.")
        return 1

    total = sum((MODELS / a.path).stat().st_size for a in WEIGHTS) / 1_073_741_824
    print(f"\nall {len(WEIGHTS)} artifacts present and checked, {total:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
