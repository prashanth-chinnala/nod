"""
Paired delivery: audio and video leave through one sequence instead of two.

**The defect this addresses.** In the arrangement this system has shipped with, the orchestrator
sends audio the moment it has it (`transport.send_audio`) while a separate task drains the mixer
at a frame cadence. Two publishers, two clocks, nothing reconciling them — and the measured
result is a trailing audio/video gap between −66 ms and +172 ms with individual turns up to 538
ms late (MEASUREMENTS §8b), against roughly 100 ms before a viewer notices.

**Why this is a decorator and not a change to the orchestrator.** `PairedDelivery` satisfies the
same `Transport` shape the orchestrator already talks to, so nothing in `orchestrator.py`,
`state.py` or the turn policy moves. `send_audio` queues into an `AvStream` instead of writing
to the wire; `flush_audio` clears it; and the caller drains one interleaved sequence and sends
both media from a single loop. That is the same composition the LLM boundary already uses --
knowledge, guardrails and the plan each wrap a stream without the orchestrator learning they
exist -- and it is what makes this switchable per deployment rather than a fork of the pipeline.

**Measured, and the answer was no -- for this transport.** Both modes were run three turns each
on the same machine and agent. Split: 25.2 fps delivered, trailing gap +27 ms median. Paired,
after two pacing bugs were fixed: 25.2 fps, and a trailing gap of **-5.9 s** -- video finishing
six seconds before the audio it belongs to, which for a viewer is a mouth that stops moving
while the interviewer is still talking. Worse, not better.

**Why, and it is the useful part.** Metering audio to a video cadence is wrong over a transport
where the client buffers. TTS produces far faster than real time; split mode sends each chunk
the moment it exists and the browser schedules playback from its own buffer, so transmission
finishes long before playback does. Paired mode meters audio out at 1x real time, so a turn
takes as long to *transmit* as to *play* -- and the turn's video, which is finite, runs out
first.

**Where it is right.** When the consumer itself consumes in real time. `rtc.AVSynchronizer`
pushes audio through `AudioSource.capture_frame`, which blocks at playback rate by design, and
pairs video against it -- so a paired sequence is exactly what it wants, and
`scripts/avatar_worker.py` measured 174 and 176 video frames arriving at a remote subscriber
that way. That is the target. This module is the seam that reaches it without rewriting the
session layer, and it stays **off** for WebSocket because the measurement says so.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from avatar.avstream import AvStream, SegmentEnd
from avatar.contracts import AudioChunk, Frame
from avatar.presentation import FramePresenter


class PairedDelivery:
    """
    A `Transport` that routes audio into an `AvStream` rather than straight to the wire.

    Wraps a real transport and forwards everything it does not intercept, so a caller holds one
    object and the orchestrator cannot tell the difference.
    """

    def __init__(
        self,
        inner: object,
        presenter: FramePresenter,
        *,
        frame_interval_ms: int,
    ) -> None:
        self._inner = inner
        self.stream = AvStream(presenter, frame_interval_ms=frame_interval_ms)
        self.audio_sent = 0
        self.frames_sent = 0

    # -- the Transport surface the orchestrator uses -------------------------

    async def open_track(self) -> None:
        await self._inner.open_track()  # type: ignore[attr-defined]

    async def send_audio(self, chunk: AudioChunk) -> None:
        """
        Queue the chunk instead of sending it.

        **Returns without awaiting the wire, and that is a real behavioural change.** The
        orchestrator uses this call's completion for nothing but its own `audio_sent_ms`
        bookkeeping, which counts what was handed over rather than what was transmitted -- so
        the accounting is unaffected. What does change is that audio no longer overtakes video
        by however long a socket write takes, which is the point.
        """
        self.stream.offer_audio(chunk)

    async def flush_audio(self) -> None:
        """
        Barge-in: drop queued audio here *and* tell the client to drop what it is playing.

        Both, in that order. Clearing locally first means the client's flush is the last word:
        if the order were reversed, a chunk queued between the two calls would be sent after the
        client had already been told to discard, and the candidate would hear a fragment of the
        sentence they interrupted.
        """
        self.stream.clear()
        await self._inner.flush_audio()  # type: ignore[attr-defined]

    async def close_track(self) -> None:
        await self._inner.close_track()  # type: ignore[attr-defined]

    async def send_frame(self, frame: Frame) -> None:
        """
        Forwarded, not queued.

        The frame path already goes through the presenter, which the `AvStream` reads from -- so
        a frame arriving here has been through the decision layer and is on its way out. Queuing
        it a second time would double it.
        """
        await self._inner.send_frame(frame)  # type: ignore[attr-defined]

    # -- the consumer side --------------------------------------------------

    async def pump(self) -> AsyncIterator[None]:
        """
        Drain the interleaved sequence, sending each item through the wrapped transport.

        An async generator rather than a plain coroutine so the caller keeps its cancellation
        and its own stop condition -- the existing frame pump checks for a closed session on
        every iteration, and that check has to keep working here.
        """
        async for item in self.stream.stream():
            if isinstance(item, SegmentEnd):
                # Nothing on the wire for this. It marks the end of an utterance, which the
                # WebSocket client infers from the audio simply stopping. `AvatarRunner` uses it
                # to fire `notify_playback_finished`; there is no equivalent here, and inventing
                # a message the client does not read would be worse than dropping it.
                yield None
                continue
            if isinstance(item, AudioChunk):
                await self._inner.send_audio(item)  # type: ignore[attr-defined]
                self.audio_sent += 1
            else:
                await self._inner.send_frame(item)  # type: ignore[attr-defined]
                self.frames_sent += 1
            yield None
