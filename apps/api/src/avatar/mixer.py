"""
Frame cadence — when a frame goes out, and under which clock.

**What this is now, and what moved.** This used to answer two questions: *which frame should the
viewer see* and *when should it be sent*. The first moved to `presentation.FramePresenter`,
because it is a product decision — the idle⇄speaking handover, the mouth-closed seam, epoch
cancellation — and because a push-based delivery model needs exactly the same decision. What is
left here is the pull half: emit one frame per tick, stamp a monotonic clock, and keep the track
alive.

The video track opens once at session start and closes at session end. It never stops carrying
frames in between. State changes swap which *source* the presenter draws from; they never
renegotiate or pause the track, because a stalled track is far more visible to a viewer than a
dropped frame -- and it corrupts the receiver's jitter estimate, so the recovery is worse than
the stall.

This layer owns presentation timestamps. It stamps `pts_ms` on every frame it emits, overriding
whatever the renderer supplied, for two reasons: the renderer has no idea how many idle frames
were shown while it was warming up, and A/V sync needs one monotonic clock rather than one per
source. **That ownership is exactly what a synchroniser replaces.** When frames are published to
a LiveKit room, `AVSynchronizer` holds the clock and pairs each frame with its audio; this class
is then the WebSocket path's equivalent rather than the only path.

Imports only `contracts`, `state`, `presentation` and `telemetry`. No torch, no renderer.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import replace

from avatar.contracts import (
    FRAME_INTERVAL_MS,
    TARGET_FPS,
    Clock,
    Frame,
    Sleep,
)
from avatar.presentation import FramePresenter, IdleLoop
from avatar.state import FrameSource
from avatar.telemetry import Telemetry

__all__ = [
    "FRAME_INTERVAL",
    "FRAME_INTERVAL_MS",
    "TARGET_FPS",
    "FrameMixer",
    "IdleLoop",
]
"""
`IdleLoop` is re-exported, and that is the one compatibility shim here.

It lives in `presentation` now, where the decision logic is. It stays importable from this
module because `renderers/musetalk.py` builds one and `idle.py` builds the placeholder, and
neither has any business knowing whether the thing that consumes it is a cadence loop or a
synchroniser. Naming the re-export in `__all__` rather than leaving it as an incidental import
makes it a stated decision.
"""

FRAME_INTERVAL = 1.0 / TARGET_FPS
"""
Seconds per frame. `TARGET_FPS` and `FRAME_INTERVAL_MS` are re-exported from `contracts` rather
than defined here, because the renderer and the server need the same values and a second
definition is a second thing to forget to change.
"""


class FrameMixer:
    """
    Emits at a constant `TARGET_FPS` for the whole session lifetime, over a pull loop.

    A thin pacing layer over `FramePresenter`. Every method that decides *what* to show
    delegates; what remains here is the clock. The public surface is deliberately unchanged from
    when this class did both jobs, so `orchestrator.py` and `server.py` did not move when the
    decision half was extracted -- the point of the extraction was to add a second delivery
    model, not to churn the callers of the first.
    """

    def __init__(
        self,
        idle: IdleLoop,
        telemetry: Telemetry,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._presenter = FramePresenter(idle, telemetry)
        self._clock = clock
        self._sleep = sleep
        self._pts_ms = 0

        self.frames_emitted = 0

    @property
    def presenter(self) -> FramePresenter:
        """
        The decision half, for a delivery model that is not this one.

        Exposed so a `VideoGenerator` can drive the same presenter this mixer drives, rather
        than constructing a second one and splitting the session's state across two objects.
        """
        return self._presenter

    # -- delegated decisions ------------------------------------------------

    def set_idle(self, idle: IdleLoop) -> None:
        self._presenter.set_idle(idle)

    @property
    def source(self) -> FrameSource:
        return self._presenter.source

    def set_source(self, source: FrameSource) -> None:
        self._presenter.set_source(source)

    def offer(self, frame: Frame, current_epoch: int) -> bool:
        return self._presenter.offer(frame, current_epoch)

    def buffered(self) -> int:
        return self._presenter.buffered()

    def at_clean_exit(self) -> bool:
        return self._presenter.at_clean_exit()

    @property
    def frames_repeated(self) -> int:
        return self._presenter.frames_repeated

    @property
    def frames_discarded(self) -> int:
        return self._presenter.frames_discarded

    # -- cadence ------------------------------------------------------------

    def next_frame(self) -> Frame:
        """
        Produce exactly one frame, stamped. Never blocks, never returns None.

        Split out from `stream` so cadence and starvation behaviour can be tested without
        driving the event loop.
        """
        frame = self._presenter.take()
        # `replace`, not a fresh `Frame`: rebuilding from three fields silently reset
        # `codec`, `width` and `height` to their defaults, so a raw frame would reach the
        # transport claiming to be JPEG with no dimensions. Only the timestamp changes here.
        stamped = replace(frame, pts_ms=self._pts_ms)
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
