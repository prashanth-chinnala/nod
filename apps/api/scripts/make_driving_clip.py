#!/usr/bin/env python3
"""
Build the driving clip that gives an uploaded photograph its motion.

**Why this is a script and not a committed asset.** It produces an `.mp4`, and every `.mp4` in this
repository is gitignored -- deliberately, because that rule is what keeps interview recordings and a
real person's reference media out of git. Carving an exemption for one file is how that rule starts
eroding. So the clip is derived, at setup time, from the MuseTalk checkout that is already required.

**Why the choice of segment is the important part.** LivePortrait *transfers* motion; it does not
invent it. Measured during the spike: the driving clip used contained **zero blinks** in the window
tested, and the animated output contained two -- so the model adds a little eye motion of its own,
and borrows everything else. A driving clip of someone sitting rigidly produces a persona that sits
rigidly. Which makes this file, not the model, the highest-leverage thing to improve.

So the segment is picked by measurement rather than by eye: every candidate window is scored on how
much the eyes actually move, using inner-eye aspect ratio per frame, and the best-scoring window
wins. A blink lasts 3-4 frames at 25fps, so this samples every frame -- an earlier attempt at this
measurement sampled every fifth and reported no blinks at all in a clip that had them.

    .venv-musetalk/bin/python scripts/make_driving_clip.py
    .venv-musetalk/bin/python scripts/make_driving_clip.py --source my_clip.mp4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WINDOW_SECONDS = 20.0
"""
How long the clip runs.

The reference loops forward then backward, so a 20s clip repeats every 40s -- past the point a
candidate notices. It is also `RECOMMENDED_VIDEO_SECONDS`, so a face built from it uploads with no
warning.
"""

FPS = 25
TARGET_HEIGHT = 768


def _eye_openness(landmarks) -> float:  # type: ignore[no-untyped-def]
    """
    Mean eye aspect ratio. Low means closed, so variance across frames means blinking.

    iBUG-68: 36-41 is one eye, 42-47 the other. Vertical spread over horizontal, which is
    scale-invariant -- so it survives the subject moving nearer or further from the camera.
    """
    import numpy as np

    def one(points) -> float:  # type: ignore[no-untyped-def]
        vertical = (
            float(np.linalg.norm(points[1] - points[5]))
            + float(np.linalg.norm(points[2] - points[4]))
        ) / 2.0
        horizontal = max(float(np.linalg.norm(points[0] - points[3])), 1e-6)
        return vertical / horizontal

    marks = landmarks.astype(float)
    return (one(marks[36:42]) + one(marks[42:48])) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(ROOT / "vendor" / "MuseTalk" / "data" / "video" / "sun.mp4"),
        help="a clip of one person sitting still and looking at the camera",
    )
    parser.add_argument("--out", default=str(ROOT / "assets" / "driving" / "idle.mp4"))
    parser.add_argument("--seconds", type=float, default=WINDOW_SECONDS)
    args = parser.parse_args()

    import cv2
    import numpy as np

    from avatar.renderers.landmarks import LandmarkDetector

    source = Path(args.source)
    if not source.is_file():
        print(f"!! no source clip at {source}", file=sys.stderr)
        return 1

    detector = LandmarkDetector()
    detector.load()

    capture = cv2.VideoCapture(str(source))
    openness: list[float] = []
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        found = detector.detect(frame)
        # A frame with no face scores as fully open, so a window containing one never wins: a
        # driving clip must have a trackable face in every frame or the transfer stutters.
        openness.append(_eye_openness(found.landmarks) if found.found else 1.0)
    capture.release()

    if frames == 0:
        print(f"!! could not read any frames from {source}", file=sys.stderr)
        return 1

    span = int(args.seconds * FPS)
    if frames <= span:
        best_start, score = 0, float(np.std(openness))
        print(f"source is {frames} frames, shorter than the window -- using all of it")
    else:
        # Standard deviation of eye openness across the window. Blinks are the only thing that
        # moves this much, so the window with the most variance is the one with the most blinking.
        scores = [
            (float(np.std(openness[i : i + span])), i) for i in range(0, frames - span, FPS // 2)
        ]
        score, best_start = max(scores)
        worst = min(scores)[0]
        print(
            f"scanned {len(scores)} windows of {args.seconds:.0f}s across {frames} frames: "
            f"best eye-motion score {score:.4f} at {best_start / FPS:.1f}s, worst {worst:.4f}"
        )

    if score < 0.005:
        # Stated rather than silently accepted. The clip will still work; the personas built from
        # it will not blink, which is the one thing this whole path exists to add.
        print(
            "!! the best window has almost no eye motion, so animated faces will not blink. "
            "Pass --source with a clip of someone blinking normally.",
            file=sys.stderr,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-ss", f"{best_start / FPS:.2f}",
            "-i", str(source),
            "-t", f"{args.seconds:.2f}",
            "-an",
            "-r", str(FPS),
            "-vf", f"scale=-2:{TARGET_HEIGHT}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(out),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    print(f"wrote {out} ({out.stat().st_size / 1_048_576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
