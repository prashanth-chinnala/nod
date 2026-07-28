"""
Where idle-loop frames come from.

Two sources, and the difference between them is the honest state of the prototype:

`load_idle_loop` reads a real clip that `scripts/prepare_idle_loop.py` decoded and
annotated -- frames on disk plus a `mouth_closed.json` naming the frames the
renderer can cut from without the jaw popping. That script is M4.

`placeholder_idle_loop` synthesises a slow brightness pulse. It is not a face. It
exists so the session layer, transport, and client can be demonstrated end to end
before a model or a reference clip exists, and it declares every frame a clean exit
-- which is true in the only sense available, since a solid rectangle has no mouth
to be caught open. The seam constraint that `at_clean_exit` enforces is therefore
untested against real footage until M4, and the demo running smoothly today is not
evidence that it will.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from avatar.bmp import solid_bmp
from avatar.mixer import TARGET_FPS, IdleLoop

MOUTH_CLOSED_MANIFEST = "mouth_closed.json"

BREATH_PERIOD_SECONDS = 4.0
"""
One full brightness cycle. Roughly a resting respiratory rate.

Chosen so the loop length is not an obvious multiple of anything a viewer can count;
a two-second cycle reads as a flicker rather than as breathing.
"""


def placeholder_idle_loop(
    *,
    width: int = 320,
    height: int = 180,
    base_rgb: tuple[int, int, int] = (44, 58, 68),
    swing: int = 10,
) -> IdleLoop:
    """
    A synthetic pulse standing in for a neutral clip. Not a face.

    Seamless by construction rather than by careful clip selection: the frame count
    is exactly one sine period, so the last frame flows into the first with no
    cross-fade needed. A real clip almost never has that property, which is why
    `IdleLoop`'s docstring talks about cross-fading and this function does not need
    to.
    """
    count = int(BREATH_PERIOD_SECONDS * TARGET_FPS)
    frames: list[bytes] = []
    for i in range(count):
        phase = math.sin(2 * math.pi * i / count)
        colour = tuple(max(0, min(255, channel + round(swing * phase))) for channel in base_rgb)
        frames.append(solid_bmp(width, height, colour))  # type: ignore[arg-type]

    # Every frame, because a solid rectangle has no mouth. Marking a subset would
    # look more rigorous and mean nothing.
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
