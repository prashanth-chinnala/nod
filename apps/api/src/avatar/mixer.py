"""
Frame cadence and the idle-loop fallback.

The video track opens once at session start and closes at session end. It never
stops carrying frames in between. State changes swap which *source* the mixer
draws from; they never renegotiate or pause the track, because a stalled track is
far more visible to a viewer than a dropped frame -- and it corrupts the
receiver's jitter estimate, so the recovery is worse than the stall.

The mixer also owns presentation timestamps. It stamps `pts_ms` on every frame it
emits, overriding whatever the renderer supplied. Two reasons: the renderer has no
idea how many idle frames were shown while it was warming up, and A/V sync needs
one monotonic clock rather than one per source.

Imports only `contracts`, `state`, and `telemetry`. No torch, no renderer.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable

from avatar.contracts import (
    FRAME_INTERVAL_MS,
    IDLE_EPOCH,
    TARGET_FPS,
    Clock,
    Frame,
    Sleep,
)
from avatar.state import FrameSource
from avatar.telemetry import Telemetry

FRAME_INTERVAL = 1.0 / TARGET_FPS
"""
Seconds per frame. `TARGET_FPS` and `FRAME_INTERVAL_MS` are re-exported from `contracts` rather
than defined here, because the renderer and the server need the same values and a second
definition is a second thing to forget to change.
"""


class IdleLoop:
    """
    Pre-decoded frames of the avatar breathing and blinking.

    Seam handling: choose a clip whose first and last frames are near-identical
    (mouth closed, neutral pose, eyes open) and cross-fade across a few frames.
    Ping-pong playback guarantees continuity but reverses gestures, which reads as
    uncanny for anything but the smallest movements.

    Exit constraint: only hand control to the renderer on a frame where the mouth
    is closed, otherwise the cut pops -- the idle clip's open mouth is replaced by
    a rendered closed one in a single frame. `mouth_closed_indices` carries that
    information, produced offline by `scripts/prepare_idle_loop.py`.
    """

    def __init__(self, frames: Iterable[bytes], mouth_closed_indices: Iterable[int]) -> None:
        self._frames = list(frames)
        if not self._frames:
            raise ValueError("idle loop needs at least one frame")
        self._mouth_closed = frozenset(mouth_closed_indices)
        if not self._mouth_closed:
            # Not fatal, but it means the orchestrator can never find a clean exit
            # and will hand over on an arbitrary frame. Better to know at startup.
            raise ValueError(
                "idle loop needs at least one mouth-closed index, "
                "or the handover to the renderer can never be seam-free"
            )
        self._i = 0

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def index(self) -> int:
        return self._i

    def next_frame(self, pts_ms: int) -> Frame:
        frame = Frame(data=self._frames[self._i], epoch=IDLE_EPOCH, pts_ms=pts_ms)
        self._i = (self._i + 1) % len(self._frames)
        return frame

    def at_clean_exit(self) -> bool:
        """True when the next idle frame shown is one the renderer can cut from."""
        return self._i in self._mouth_closed


class FrameMixer:
    """
    Emits at a constant TARGET_FPS for the whole session lifetime.

    Starvation policy, in order of preference:

      1. a rendered frame from the queue
      2. repeat the last rendered frame, and count it -- the mouth freezes for
         40ms, which is far less noticeable than a stall
      3. fall back to a live idle frame, if the renderer has not produced
         anything at all yet

    Case 2 is the interesting metric. `frames_repeated` climbing means the GPU is
    behind real time, which is the signal that matters and the one a pure fps
    average hides.
    """

    def __init__(
        self,
        idle: IdleLoop,
        telemetry: Telemetry,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._idle = idle
        self._telemetry = telemetry
        self._clock = clock
        self._sleep = sleep

        self._rendered: asyncio.Queue[Frame] = asyncio.Queue()
        self._source = FrameSource.IDLE_LOOP
        self._last_rendered: Frame | None = None
        self._pts_ms = 0

        self.frames_emitted = 0
        self.frames_repeated = 0
        self.frames_discarded = 0

    # -- source selection ---------------------------------------------------

    @property
    def source(self) -> FrameSource:
        return self._source

    def set_source(self, source: FrameSource) -> None:
        """
        The only way the frame source changes.

        Called from exactly one place -- `SessionOrchestrator._transition` -- with
        the value looked up from `state.FRAME_SOURCE`. Keeping this single-entry
        is what makes "which state shows which source" a table a test can walk
        rather than behaviour scattered across the pipeline.
        """
        if source == self._source:
            return
        self._source = source
        if source is FrameSource.IDLE_LOOP:
            # Whatever is queued belongs to a turn we are no longer showing.
            self.frames_discarded += self._drain()
            self._last_rendered = None

    def _drain(self) -> int:
        dropped = 0
        while not self._rendered.empty():
            self._rendered.get_nowait()
            dropped += 1
        if dropped:
            self._telemetry.increment("frames_discarded", amount=dropped)
        return dropped

    # -- producer side ------------------------------------------------------

    def offer(self, frame: Frame, current_epoch: int) -> bool:
        """
        Hand a rendered frame to the mixer. Returns False if it was dropped.

        The epoch check is the consumer-side half of cancellation: the renderer is
        allowed to keep producing frames for a turn that has been abandoned, and
        they die here rather than requiring the renderer to be interruptible.
        """
        if frame.epoch != current_epoch:
            self._telemetry.stale_artifact_dropped(
                "frame", stale_epoch=frame.epoch, current=current_epoch
            )
            return False
        self._rendered.put_nowait(frame)
        return True

    def buffered(self) -> int:
        return self._rendered.qsize()

    def at_clean_exit(self) -> bool:
        """
        Whether the idle loop is currently on a frame the renderer can cut from.

        Delegated rather than exposing the `IdleLoop` itself: the orchestrator
        needs the answer to time the handover, but giving it the loop would let
        state logic start advancing frames.
        """
        return self._idle.at_clean_exit()

    # -- consumer side ------------------------------------------------------

    def next_frame(self) -> Frame:
        """
        Produce exactly one frame. Never blocks, never returns None.

        Split out from `stream` so cadence and starvation behaviour can be tested
        without driving the event loop.
        """
        frame: Frame | None = None

        if self._source is FrameSource.RENDERER:
            try:
                frame = self._rendered.get_nowait()
            except asyncio.QueueEmpty:
                if self._last_rendered is not None:
                    self.frames_repeated += 1
                    self._telemetry.frame_repeated(total=self.frames_repeated)
                    frame = self._last_rendered
            if frame is not None:
                self._last_rendered = frame

        if frame is None:
            # Either the idle loop is selected, or the renderer is selected but has
            # not yet produced a first frame. A live idle frame beats freezing on
            # the last one we happened to be showing.
            frame = self._idle.next_frame(self._pts_ms)

        stamped = Frame(data=frame.data, epoch=frame.epoch, pts_ms=self._pts_ms)
        self._pts_ms += FRAME_INTERVAL_MS
        self.frames_emitted += 1
        return stamped

    async def stream(self) -> AsyncIterator[Frame]:
        """Yield frames at a constant cadence until cancelled."""
        next_due = self._clock()
        while True:
            yield self.next_frame()
            next_due += FRAME_INTERVAL
            await self._sleep(max(0.0, next_due - self._clock()))
