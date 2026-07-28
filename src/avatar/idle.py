"""
Where idle-loop frames come from.

Two sources, and the difference between them is the honest state of the prototype:

`load_idle_loop` reads a real clip that `scripts/prepare_idle_loop.py` decoded and
annotated -- frames on disk plus a `mouth_closed.json` naming the frames the
renderer can cut from without the jaw popping. That script is M4.

`placeholder_idle_loop` synthesises the same placeholder the stub renderer draws, with
its mouth closed and a slow brightness pulse standing in for breathing. It is not a
face.

It shares `draw_placeholder` with the stub renderer deliberately: the idle loop and
the rendered frames have to be visually continuous, or the handover between them pops
regardless of how carefully `at_clean_exit` times it. Two placeholders that looked
different would hide exactly the artifact the seam logic exists to prevent. When a
real renderer lands, its idle frames come from the real reference clip via
`load_idle_loop`, and this function stops being used at all.

Every frame is declared a clean exit, which is true in the only sense available: the
placeholder's mouth is always closed, so any frame is safe to cut from. The seam
constraint is therefore untested against real footage, and the demo running smoothly
today is not evidence that it will.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from avatar.mixer import TARGET_FPS, IdleLoop
from avatar.renderers.stub import draw_placeholder

MOUTH_CLOSED_MANIFEST = "mouth_closed.json"

BREATH_PERIOD_SECONDS = 4.0
"""
One full brightness cycle. Roughly a resting respiratory rate.

Chosen so the loop length is not an obvious multiple of anything a viewer can count;
a two-second cycle reads as a flicker rather than as breathing.
"""


BREATH_SWING = 0.13
"""
How far the head brightness swings either side of neutral.

Large enough to read as alive on a glance, small enough not to look like a fault.
"""


def placeholder_idle_loop(
    *, width: int = 320, height: int = 180, swing: float = BREATH_SWING
) -> IdleLoop:
    """
    The stub placeholder with its mouth closed, breathing. Not a face.

    Seamless by construction rather than by careful clip selection: the frame count is
    exactly one sine period, so the last frame flows into the first with no cross-fade
    needed. A real clip almost never has that property, which is why `IdleLoop`'s
    docstring talks about cross-fading and this function does not need to.
    """
    count = int(BREATH_PERIOD_SECONDS * TARGET_FPS)
    frames = [
        draw_placeholder(
            width,
            height,
            level=0,  # mouth closed: the avatar is not speaking
            brightness=1.0 + swing * math.sin(2 * math.pi * i / count),
        )
        for i in range(count)
    ]

    # Every frame, because the placeholder's mouth is always closed and so any frame
    # is safe to cut from. Marking a subset would look more rigorous and mean nothing.
    return IdleLoop(frames, range(count))


def load_idle_loop(directory: Path) -> IdleLoop:
    """
    Load a prepared clip: `*.bmp` frames in sort order plus `mouth_closed.json`.

    Fails loudly on a missing manifest rather than defaulting to "every frame is
    fine". A silently-wrong clean-exit set produces an intermittent visible pop at
    the start of some turns and not others, which is close to undebuggable from a
    recording -- so the error belongs at load time.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"no prepared idle loop at {directory}")

    frame_paths = sorted(directory.glob("*.bmp"))
    if not frame_paths:
        raise FileNotFoundError(f"no .bmp frames in {directory}")

    manifest = directory / MOUTH_CLOSED_MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(
            f"{directory} has frames but no {MOUTH_CLOSED_MANIFEST}; "
            "run scripts/prepare_idle_loop.py to regenerate it"
        )

    indices = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(indices, list) or not all(isinstance(i, int) for i in indices):
        raise ValueError(f"{manifest} must contain a JSON list of frame indices")

    out_of_range = [i for i in indices if not 0 <= i < len(frame_paths)]
    if out_of_range:
        raise ValueError(
            f"{manifest} names frames {out_of_range} but only "
            f"{len(frame_paths)} frames exist -- the manifest is stale"
        )

    return IdleLoop([p.read_bytes() for p in frame_paths], indices)
