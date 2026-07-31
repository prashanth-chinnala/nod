"""
Give a photograph real motion, so a still reference can blink.

**The gap this closes.** MuseTalk repaints the mouth of the frames it is given. With one frame
there is nothing else to work with, so an uploaded photograph becomes a persona that holds one
pose for ever -- head fixed, eyes fixed, only the mouth moving. That is not a renderer bug to
fix; it is what "repaint the frames you were given" means with a single frame.

The fix is at enrollment: generate a reference *clip* with real motion once, then let MuseTalk
lip-sync that clip live. Nothing about serving changes.

**Why LivePortrait and not a generator.** SadTalker, Hallo and the diffusion animators
*generate* motion; LivePortrait *transfers* it from a driving clip, via implicit keypoints, and
explicitly does not swap faces. That distinction is the whole reason identity survives --
measured at **3.6% face-proportion deviation** from the source photograph, confirmed by looking
at the frames. A model that invented motion would also be free to invent a different person,
which is worse than a frozen head: nobody wants a persona that is *nearly* the employee whose
photo they uploaded.

**Why a subprocess and not an import.** The same lesson the voice cloner taught. LivePortrait
pins `numpy==1.26.4` and `opencv-python==4.10`; the renderer needs numpy 2.x because `diffusers`
uses `np.long`, and `opencv` compiled against numpy 1 aborts on import against numpy 2.
Installing both into one environment broke torch outright. So this shells out to LivePortrait's
own interpreter, and the runtime imports nothing from it.

**The driving clip is the limit, not the model.** Measured: the clip used in the spike contained
zero blinks in the window tested and the output contained two, so LivePortrait adds eye motion
of its own -- but everything else it produces is borrowed. A driving clip of someone sitting
rigidly yields a persona that sits rigidly. `scripts/make_driving_clip.py` builds the bundled
one, and swapping it is the highest-leverage change available here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

LIVEPORTRAIT_ROOT_ENV = "AVATAR_LIVEPORTRAIT_ROOT"
LIVEPORTRAIT_PYTHON_ENV = "AVATAR_LIVEPORTRAIT_PYTHON"
DRIVING_CLIP_ENV = "AVATAR_DRIVING_CLIP"

TIMEOUT_S = 1800
"""
Thirty minutes. Generous because the measured cost is ~414 ms/frame on a T4 with CPU ONNX
providers -- a 20s driving clip is 500 frames, so 3-4 minutes -- and a bound exists so a hung
subprocess fails the job instead of holding a worker for ever.
"""


class AnimationUnavailable(RuntimeError):
    """
    LivePortrait is not installed, or is not usable. Distinct from a failure to animate.

    Its own type because the two need different answers: unavailable means "this deployment
    cannot
    do this, and the upload should still succeed as a still", where a failure means "this
    specific
    photograph did not work". Collapsing them would make a missing install look like a bad
    photo.
    """


@dataclass(frozen=True)
class Animated:
    clip: Path
    frames: int
    seconds: float
    ms: int
    driving: Path


def _root() -> Path:
    configured = os.environ.get(LIVEPORTRAIT_ROOT_ENV)
    if configured:
        return Path(configured)
    # Beside the repository, not inside it: it is a third-party checkout with its own licence
    # and
    # 2 GB of weights, and the same reasoning keeps the MuseTalk checkout out of git.
    return Path(__file__).resolve().parents[4] / "LivePortrait"


def _python() -> Path:
    configured = os.environ.get(LIVEPORTRAIT_PYTHON_ENV)
    if configured:
        return Path(configured)
    return _root() / ".venv" / "bin" / "python"


def driving_clip() -> Path:
    """
    The motion source. Bundled, overridable, and the thing to change to improve output.

    A fixed clip rather than a per-face choice, because the operator uploading a photograph has
    no
    way to judge which driving video suits it -- and the answer is the same for every portrait:
    a
    person sitting still, looking ahead, blinking normally.
    """
    configured = os.environ.get(DRIVING_CLIP_ENV)
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "assets" / "driving" / "idle.mp4"


def available() -> str:
    """
    Empty if animation is usable, otherwise the reason it is not.

    Returns a reason rather than a bool so the caller can put it in front of an operator. "Not
    available" with no explanation is the kind of message that produces a support conversation.
    """
    root, python, clip = _root(), _python(), driving_clip()
    if not (root / "inference.py").is_file():
        return (
            f"no LivePortrait checkout at {root}. Clone it and set "
            f"{LIVEPORTRAIT_ROOT_ENV}, or leave photographs as stills."
        )
    if not python.is_file():
        return (
            f"no interpreter at {python}. LivePortrait needs its own environment -- it pins "
            f"numpy 1.26 against the renderer's 2.x. Set {LIVEPORTRAIT_PYTHON_ENV}."
        )
    if not (root / "pretrained_weights" / "liveportrait").is_dir():
        return f"LivePortrait weights are missing under {root / 'pretrained_weights'}."
    if not clip.is_file():
        return (
            f"no driving clip at {clip}. Build one with scripts/make_driving_clip.py, or set "
            f"{DRIVING_CLIP_ENV}."
        )
    if not shutil.which("ffprobe"):
        return "ffprobe is not installed, so the result could not be verified."
    return ""


def animate(image: Path, *, into: Path | None = None) -> Animated:
    """
    Animate one photograph against the driving clip. Returns the generated reference.

    Slow and synchronous -- minutes. Every caller runs it on a worker thread; see `avatar.jobs`.

    The output is verified before being returned rather than trusted from the exit code. A
    generation that produces one frame, or no file, is a failure that would otherwise surface
    much
    later as a persona that still does not blink, and the exit code alone does not distinguish
    it.
    """
    reason = available()
    if reason:
        raise AnimationUnavailable(reason)

    root, python, clip = _root(), _python(), driving_clip()
    destination = into or image.parent
    output = destination / f"{image.stem}-animated"
    output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    result = subprocess.run(
        [
            str(python), "inference.py",
            "-s", str(image.resolve()),
            "-d", str(clip.resolve()),
            "-o", str(output.resolve()),
            # Crop the driving video to its face, so motion is transferred from the head rather
            # than from whatever else moves in the frame.
            "--flag_crop_driving_video",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        # No stdin. `inference.py` does not read it, but ffmpeg further down does, and an ffmpeg
        # that inherits a pipe consumes whatever is on it -- which has already eaten a shell
        # script
        # in this project.
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-4:]
        raise RuntimeError("LivePortrait failed: " + " / ".join(tail or ["no output"]))

    # It writes `<source>--<driving>.mp4` and a side-by-side `_concat.mp4`. The concat one is
    # twice
    # the width and shows the driving face next to the result, which would be a very confusing
    # persona, so it is excluded by name rather than by picking whichever file is newest.
    produced = sorted(
        (p for p in output.glob("*.mp4") if not p.stem.endswith("_concat")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not produced:
        raise RuntimeError(f"LivePortrait exited cleanly but wrote no video into {output}")

    clip_path = produced[0]
    frames, seconds = _probe(clip_path)
    if frames < 2:
        raise RuntimeError(
            f"the animated clip has {frames} frame(s), so nothing was animated. The photograph "
            "may have no detectable face."
        )
    return Animated(
        clip=clip_path,
        frames=frames,
        seconds=seconds,
        ms=round((time.perf_counter() - started) * 1000),
        driving=clip,
    )


def _probe(path: Path) -> tuple[int, float]:
    """Frame count and duration, read from the file rather than assumed from the driving."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames",
            "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        stdin=subprocess.DEVNULL,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    frames = next((int(v) for v in values if v.isdigit()), 0)
    seconds = next((float(v) for v in values if not v.isdigit()), 0.0)
    return frames, seconds
