"""
Which frame to show next — independent of how it gets delivered.

**Why this is its own module.** `FrameMixer` used to answer two questions in one class:
*what should the viewer see right now* and *when should it go out*. The first is a product
decision — is the persona standing by or speaking, is this a safe frame to cut on, does this
frame belong to a turn that was abandoned. The second is transport mechanics: emit at a cadence,
stamp a monotonic clock.

Separating them matters because the two questions have different owners. The delivery side is
generic and somebody else does it better: LiveKit's `AVSynchronizer` paces frames and pairs them
with audio. That is the fix for the audio/video drift measured at −66 ms to +172 ms
(MEASUREMENTS §8b): the gap is what two publishers on two independent clocks produce, and one
synchroniser removes it by construction rather than by tuning. The decision side is ours and
must not be handed to a framework: the idle⇄speaking handover, the mouth-closed seam and epoch
cancellation are the things that make this an interview rather than a video call.

**The concrete reason this is a separate class and not a refactor of the pull loop.** A cadence
loop *pulls* one frame per tick. LiveKit's `VideoGenerator.__aiter__` *pushes* frames into a
synchroniser. Both need exactly the same decision made, and the decision is the
highest-regression surface in this repo — the clean-exit seam has never been checked against
real footage even today. Implementing it twice, once per delivery model, is the single worst
duplication available here. So it lives once, and both delivery models call `take()`.

Imports only `contracts`, `state` and `telemetry`. No torch, no renderer, no transport.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from avatar.contracts import IDLE_EPOCH, Frame, FrameCodec
from avatar.state import FrameSource
from avatar.telemetry import Telemetry


class IdleLoop:
    """
    Pre-decoded frames of the avatar breathing and blinking.

    Seam handling: choose a clip whose first and last frames are near-identical (mouth closed,
    neutral pose, eyes open) and cross-fade across a few frames. Ping-pong playback guarantees
    continuity but reverses gestures, which reads as uncanny for anything but the smallest
    movements.

    Exit constraint: only hand control to the renderer on a frame where the mouth is closed,
    otherwise the cut pops -- the idle clip's open mouth is replaced by a rendered closed one in
    a single frame. `mouth_closed_indices` carries that information;
    `MuseTalkRenderer.idle_loop` derives it from the reference's own quietest frames.
    """

    def __init__(
        self,
        frames: Iterable[bytes],
        mouth_closed_indices: Iterable[int],
        *,
        codec: str = FrameCodec.JPEG,
        width: int = 0,
        height: int = 0,
    ) -> None:
        """
        `codec` describes these frames, and getting it wrong is a silent corruption.

        The idle loop is built by whoever produced the frames -- the placeholder generator, or a
        renderer from its own reference -- and only that caller knows the format. Defaulting to
        JPEG rather than requiring it, because every existing caller produced encoded frames. A
        caller making raw pixels has to say so, and `Frame.is_raw` then insists on dimensions.
        """
        self._codec = codec
        self._width = width
        self._height = height
        self._frames = list(frames)
        if not self._frames:
            raise ValueError("idle loop needs at least one frame")
        self._mouth_closed = frozenset(mouth_closed_indices)
        if not self._mouth_closed:
            # Not fatal, but it means the orchestrator can never find a clean exit and will hand
            # over on an arbitrary frame. Better to know at startup.
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

    def next_frame(self) -> Frame:
        """
        The next idle frame, with `pts_ms` left at zero.

        **No timestamp, deliberately.** This used to take a `pts_ms` that the mixer immediately
        overwrote when it re-stamped the frame, so the argument was threaded through for nothing
        and implied the idle loop knew about a clock it has no access to. Whoever delivers the
        frame owns the clock -- the cadence loop today, `AVSynchronizer` tomorrow.
        """
        frame = Frame(
            data=self._frames[self._i],
            epoch=IDLE_EPOCH,
            pts_ms=0,
            codec=self._codec,
            width=self._width,
            height=self._height,
        )
        self._i = (self._i + 1) % len(self._frames)
        return frame

    def at_clean_exit(self) -> bool:
        """True when the next idle frame shown is one the renderer can cut from."""
        return self._i in self._mouth_closed


class FramePresenter:
    """
    Decides which frame the viewer should see next. Owns no clock and no timestamps.

    Starvation policy, in order of preference:

      1. a rendered frame from the queue
      2. repeat the last rendered frame, and count it -- the mouth freezes for one frame
         interval, which is far less noticeable than a stall
      3. fall back to a live idle frame, if the renderer has not produced anything at all yet

    Case 2 is the interesting metric. `frames_repeated` climbing means the renderer is behind
    real time, which is the signal that matters and the one a pure fps average hides.

    **`take()` never blocks and never returns None**, and both halves of that are load-bearing.
    A cadence loop must have something to send every tick or the track stalls -- which is more
    visible than a repeated frame and corrupts the receiver's jitter estimate, so the recovery
    is worse than the stall. A push-based generator has the same requirement for the same
    reason.
    """

    def __init__(self, idle: IdleLoop, telemetry: Telemetry) -> None:
        self._idle = idle
        self._telemetry = telemetry

        # An asyncio queue rather than a deque, because `offer` is called from the
        # orchestrator's loop and `take` from the delivery loop; the unbounded, non-blocking
        # `put_nowait` / `get_nowait` pair is exactly the shape both sides need and neither side
        # ever awaits it.
        self._rendered: asyncio.Queue[Frame] = asyncio.Queue()
        self._source = FrameSource.IDLE_LOOP
        self._last_rendered: Frame | None = None

        self.frames_repeated = 0
        self.frames_discarded = 0

    def set_idle(self, idle: IdleLoop) -> None:
        """
        Replace the idle loop, once, at session start.

        Exists so a renderer can supply an idle loop made from the persona's own reference
        frames instead of the grey placeholder -- see `MuseTalkRenderer.idle_loop`. Called from
        `SessionOrchestrator.start` before any frame is produced, which is why this does not
        have to reason about swapping mid-playback.
        """
        self._idle = idle

    # -- source selection ---------------------------------------------------

    @property
    def source(self) -> FrameSource:
        return self._source

    def set_source(self, source: FrameSource) -> None:
        """
        The only way the frame source changes.

        Called from exactly one place -- `SessionOrchestrator._transition` -- with the value
        looked up from `state.FRAME_SOURCE`. Keeping this single-entry is what makes "which
        state shows which source" a table a test can walk rather than behaviour scattered across
        the pipeline.
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
        Hand a rendered frame over. Returns False if it was dropped.

        The epoch check is the consumer-side half of cancellation: the renderer is allowed to
        keep producing frames for a turn that has been abandoned, and they die here rather than
        requiring the renderer to be interruptible. That property has to survive the move to a
        separate renderer process, where interrupting the producer is a round trip and this
        check is still one integer comparison.
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

        Delegated rather than exposing the `IdleLoop` itself: the orchestrator needs the answer
        to time the handover, but giving it the loop would let state logic start advancing
        frames.
        """
        return self._idle.at_clean_exit()

    # -- consumer side ------------------------------------------------------

    def take(self) -> Frame:
        """
        The frame to show next. Never blocks, never returns None, never stamps a timestamp.

        `pts_ms` on the returned frame is whatever the producer set, and callers are expected to
        overwrite it: the renderer has no idea how many idle frames were shown while it was
        warming up, and A/V sync needs one monotonic clock rather than one per source.
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
            # Either the idle loop is selected, or the renderer is selected but has not yet
            # produced a first frame. A live idle frame beats freezing on the last one we
            # happened to be showing.
            frame = self._idle.next_frame()

        return frame
