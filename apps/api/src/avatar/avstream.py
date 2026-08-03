"""
One interleaved stream of audio and video, for a synchroniser to pair.

**The problem this exists to remove.** Today audio leaves via `transport.send_audio()` and video
via the mixer's cadence loop. Two publishers, two clocks, reconciled by nothing — and the
measured result is a trailing audio/video gap between −66 ms and +172 ms with individual turns
up to 538 ms late (MEASUREMENTS §8b), against roughly 100 ms of tolerance before a viewer
notices. No amount of tuning fixes a two-clock system; the fix is one stream.

This is that stream. It yields audio chunks and video frames from a single sequence so that
whatever consumes it — LiveKit's `AVSynchronizer` in production, a test in CI — sees them in the
order they should play and can pair them itself.

**What it deliberately does not own.** Not the renderer: the orchestrator drives that and offers
frames to the presenter, which keeps turn and epoch logic in one place. Not a clock for
timestamps: the consumer stamps. Not the decision of which frame to show: `FramePresenter` does
that, and it is shared with the pull-based mixer precisely so the idle⇄speaking seam is not
implemented twice.

**The interleaving policy, stated because it is a real choice.** Audio is emitted the moment it
arrives; video is emitted on a fixed interval. Audio wins ties because audio is what the
candidate is listening to — a late frame is a frozen mouth for one interval, a late chunk is a
gap in speech, and only one of those is recoverable. This mirrors what the two-publisher path
does today, with the difference that both now pass through one sequence and are therefore
pairable.

Imports only `contracts`, `presentation` and `state`. No torch, no renderer, no transport, and
in particular no LiveKit: the binding that satisfies `VideoGenerator` lives elsewhere and is a
shim over this. That is what lets the interesting logic here be tested with neither a GPU nor an
SFU.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from avatar.contracts import AudioChunk, Clock, Frame, Sleep
from avatar.presentation import FramePresenter


@dataclass(frozen=True, slots=True)
class SegmentEnd:
    """
    The end of one utterance.

    **Our own marker rather than LiveKit's `AudioSegmentEnd`**, so this module carries no
    dependency on a framework it is only one consumer of. The binding maps between them; a test
    reads this one.

    Carries the epoch it ends. LiveKit's marker carries nothing, which is fine for its purpose —
    it only drives `notify_playback_finished`. Ours has to survive a barge-in arriving between
    the last chunk of a turn and its terminator, and without the epoch a stale terminator would
    report the wrong turn finished.
    """

    epoch: int


Emitted = AudioChunk | Frame | SegmentEnd
"""
What the stream yields.

A union rather than two streams, because the whole point is one ordered sequence. Consumers
switch on the type: audio goes to an audio source, video to a video source, and a terminator
says the utterance is over.
"""


class AvStream:
    """
    Interleaves audio and video from one session into a single ordered sequence.

    Drive it by pushing audio in and iterating frames out. Both halves are non-blocking on the
    producer side: `offer_audio` returns immediately, and the stream paces itself.

    Not thread-safe and not intended to be: everything here runs on one event loop, the same one
    the orchestrator runs on, and the only slow work in the pipeline — the render — already
    happens on a worker thread behind `asyncio.to_thread`.
    """

    def __init__(
        self,
        presenter: FramePresenter,
        *,
        frame_interval_ms: int,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._presenter = presenter
        self._interval = frame_interval_ms / 1000.0
        self._interval_ms = frame_interval_ms
        self._clock = clock
        self._sleep = sleep

        self._audio: asyncio.Queue[AudioChunk | SegmentEnd] = asyncio.Queue()
        self._pts_ms = 0
        self._audio_pts_ms = 0

        self.audio_emitted = 0
        self.frames_emitted = 0
        self.audio_dropped = 0
        """
        Chunks discarded by `clear()`. The audio equivalent of `frames_discarded`.

        Counted because a barge-in that drops speech the candidate had already begun hearing is
        a different event from one that drops speech still queued, and the only way to tell them
        apart later is to have counted.
        """

    # -- producer side ------------------------------------------------------

    def offer_audio(self, chunk: AudioChunk) -> None:
        """
        Queue one chunk of synthesised speech for emission.

        No epoch check here, unlike `FramePresenter.offer`. The asymmetry is deliberate: a frame
        can be produced long after its turn was abandoned because rendering is slow and runs
        behind, whereas audio is queued by the same coroutine that owns the turn and is dropped
        by
        `clear()` at the moment of the barge-in. Adding a check would be a second cancellation
        mechanism for a race that cannot happen on this side.
        """
        self._audio.put_nowait(chunk)

    def end_segment(self, epoch: int) -> None:
        """
        Mark the end of an utterance, after its last chunk.

        Queued rather than emitted immediately so it stays in order behind the audio it
        terminates.
        A terminator that overtook its own final chunk would tell the consumer playback had
        finished while a chunk was still to come, and the visible result is an avatar that stops
        moving a beat before it stops talking.
        """
        self._audio.put_nowait(SegmentEnd(epoch=epoch))

    def clear(self) -> None:
        """
        Drop everything pending. The stream's half of a barge-in.

        Both queues, because both belong to the turn being abandoned: the presenter discards
        rendered frames, and the retained audio goes here. `FramePresenter.set_source` already
        drains frames when the source returns to idle — this is the audio that had no
        equivalent, and without it a barge-in would cut the video and keep speaking.

        Sync, not async, so it can be called from a signal handler or an RPC callback without
        scheduling. LiveKit's `clear_buffer` accepts either.
        """
        dropped = 0
        while not self._audio.empty():
            self._audio.get_nowait()
            dropped += 1
        self.audio_dropped += dropped

    def pending_audio(self) -> int:
        return self._audio.qsize()

    # -- consumer side ------------------------------------------------------

    def take_audio(self) -> AudioChunk | SegmentEnd | None:
        """
        The next audio item, or None if nothing is waiting. Never blocks.

        Separate from `take_frame` so a consumer that owns its own pacing — a synchroniser, or a
        test walking the sequence deterministically — can drain audio without being handed a
        video cadence it did not ask for.
        """
        try:
            item = self._audio.get_nowait()
        except asyncio.QueueEmpty:
            return None
        if isinstance(item, SegmentEnd):
            return item
        stamped = AudioChunk(
            pcm=item.pcm, epoch=item.epoch, duration_ms=item.duration_ms
        )
        self._audio_pts_ms += item.duration_ms
        self.audio_emitted += 1
        return stamped

    def take_frame(self) -> Frame:
        """
        The next video frame, stamped. Never blocks, never returns None.

        Stamped here rather than in the presenter because this is the delivery side and the
        delivery side owns the clock. When a synchroniser holds the clock instead, it overwrites
        this — harmlessly, and the monotonic sequence is still correct in the meantime, which
        matters for the WebSocket path and for tests.
        """
        frame = self._presenter.take()
        stamped = Frame(data=frame.data, epoch=frame.epoch, pts_ms=self._pts_ms)
        self._pts_ms += self._interval_ms
        self.frames_emitted += 1
        return stamped

    async def stream(self) -> AsyncIterator[Emitted]:
        """
        Yield audio and video interleaved, until cancelled.

        One frame per interval, with every audio item that has arrived emitted first. That
        ordering is the policy stated in the module docstring: audio is what the candidate
        hears, and a chunk held back to keep a frame cadence tidy is a gap in speech.

        The cadence is computed against a monotonic deadline rather than by sleeping a fixed
        interval, so a slow iteration does not accumulate drift — the same reason
        `FrameMixer.stream` does it that way. It never yields nothing on a tick, because a
        consumer starved of video stalls its track, which is worse than a repeated frame.
        """
        next_due = self._clock()
        while True:
            while (item := self.take_audio()) is not None:
                yield item
            yield self.take_frame()
            next_due += self._interval
            await self._sleep(max(0.0, next_due - self._clock()))
