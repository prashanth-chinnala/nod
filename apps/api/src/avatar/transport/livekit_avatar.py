"""
The LiveKit binding: satisfy `VideoGenerator` by wrapping `AvStream`.

**Why this file is thin on purpose.** Everything interesting — the interleaving policy, the
idle⇄speaking handover, the barge-in drain, the terminator ordering — is in `avstream.py` and
`presentation.py`, tested with no GPU, no SFU and no `livekit-agents` installed. What is left
here is type conversion and three method names. That split is deliberate: a framework binding is
the worst place to keep logic, because it can only be tested where the framework is, and the
framework is the part we do not control.

**The interface, read from `livekit/agents/voice/avatar/_types.py`:**

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None
    def clear_buffer(self) -> None | Coroutine[None, None, None]
    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]

The third signature is the one worth reading twice, and the reason `AvStream` exists: **the
generator yields audio as well as video.** `AvatarRunner` pushes whatever comes out into an
`rtc.AVSynchronizer`, which is what pairs them. A generator that yielded video only would
publish a silent track and leave the audio to a second publisher — which is exactly the
two-clock arrangement this work exists to remove.

**Status: written, not yet run against a room.** `livekit-agents` is not a dependency of this
app and there is no SFU in the test environment, so nothing below has executed against a real
room. It is here so the conversion is reviewable and so the shape of the remaining work is
visible; the frame
conversion in particular (`_to_video_frame`) needs a real `rtc.VideoFrame` to validate, and
JPEG-encoded input is the wrong format for it — see the note there. Do not read the absence of
tests as confidence.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from avatar.avstream import AvStream, SegmentEnd
from avatar.contracts import AudioChunk, Frame, FrameCodec

BYTES_PER_SAMPLE = 2
"""16-bit PCM, which is what every adapter in `avatar.audio` produces."""


class SegmentEpochs:
    """
    Turn numbers derived from segment boundaries, for a worker in its own process.

    **This dissolves what the notes called the one genuinely unresolved piece of the split.**
    The worry was that LiveKit's audio frames carry samples and a rate but not our turn number,
    so the epoch would have to be put on the wire. It does not: an epoch's only job is to
    distinguish one turn from the next so that in-flight work for an abandoned turn can be
    recognised and dropped.
    Nothing needs the *same integer* on both sides -- only a counter that changes when the turn
    does.

    And the wire already carries exactly that signal. `DataStreamAudioOutput.flush()` closes the
    byte stream at the end of an utterance, so **one stream is one turn**, and the receiver
    surfaces the boundary as `AudioSegmentEnd`. Counting those is a complete, local answer.

    A barge-in bumps it too, which is the whole point: `clear_buffer` arrives, the counter
    moves, and every frame already rendered for the abandoned turn is stale at the presenter
    without anyone having to reach into the renderer.
    """

    def __init__(self, start: int = 1) -> None:
        # Starts at 1 because IDLE_EPOCH is 0: an idle frame must never collide with a turn.
        self._epoch = start

    def __call__(self) -> int:
        return self._epoch

    def advance(self) -> int:
        """Move to the next turn. Called on a segment end and on a barge-in."""
        self._epoch += 1
        return self._epoch


class LiveKitVideoGenerator:
    """
    Adapts `AvStream` to LiveKit's `VideoGenerator`.

    Structural conformance, not inheritance: `VideoGenerator` is a Protocol, and subclassing it
    would mean importing `livekit.agents` at module scope. This module has to stay importable
    without it so the rest of the package — and the test suite — does not acquire a dependency
    on a framework only the worker needs.
    """

    def __init__(
        self,
        stream: AvStream,
        *,
        sample_rate: int,
        epoch: Callable[[], int],
        channels: int = 1,
        on_audio: Callable[[AudioChunk], None] | None = None,
    ) -> None:
        self._stream = stream
        self._sample_rate = sample_rate
        self._channels = channels
        self._on_audio = on_audio
        """
        Called with each chunk as it arrives, before it is queued for emission.

        **This is where received audio becomes a rendered face**, and without it a worker in its
        own process publishes its idle loop and nothing else -- which is what the first split
        run did: audio crossed the wire, epochs advanced correctly, and the video was the
        persona standing by, because nothing had told the renderer to render. Verified rather
        than assumed only because `frames_discarded` stayed at zero when it should not have.

        A hook rather than the generator owning a renderer: the renderer's lifecycle, its
        identity and its session belong to whoever built it. This module converts types.
        """
        self._epoch = epoch
        """
        Which turn the audio arriving now belongs to.

        Injected rather than inferred, because the answer differs by topology and only the
        caller knows which one it is in. **In-process** -- Step 1, one process,
        `QueueAudioOutput` -- this is the orchestrator's own epoch and is exactly right.
        **Across a process boundary** it is not: LiveKit's audio frames carry samples and a
        rate, not our turn number, so the sender has to put the epoch on the wire and this has
        to read it from there.

        A default of `lambda: 1` would have made the in-process case work and the split-process
        case silently wrong, which is the failure mode this codebase keeps paying for. So there
        is no default.
        """

    # -- the three methods LiveKit calls ------------------------------------

    async def push_audio(self, frame: Any) -> None:
        """
        Receive one audio frame, or the marker that an utterance ended.

        Dispatched on the presence of `data` rather than on the type, because importing
        `AudioSegmentEnd` to isinstance against it is the dependency this module is avoiding.
        LiveKit's terminator has no payload; an `rtc.AudioFrame` does.

        **A terminator advances the epoch when the epoch source is a `SegmentEpochs`.** That is
        what makes a worker in its own process able to tell one turn from the next without the
        sender putting a turn number on the wire — see `SegmentEpochs`. With an injected
        in-process source the orchestrator already owns the number and nothing is advanced here.
        """
        data = getattr(frame, "data", None)
        if data is None:
            self._stream.end_segment(epoch=self._epoch())
            advance = getattr(self._epoch, "advance", None)
            if callable(advance):
                advance()
            return

        pcm = bytes(data)
        samples = len(pcm) // (BYTES_PER_SAMPLE * self._channels)
        duration_ms = round(samples / self._sample_rate * 1000)
        chunk = AudioChunk(pcm=pcm, epoch=self._epoch(), duration_ms=duration_ms)
        # The renderer first, then the queue. Order matters only in that a renderer which raised
        # must not leave the audio silently unqueued -- so it is not wrapped: a renderer that
        # cannot render is a broken session, and hiding it here would produce a silent avatar
        # with no explanation.
        if self._on_audio is not None:
            self._on_audio(chunk)
        self._stream.offer_audio(chunk)

    def clear_buffer(self) -> None:
        """
        Barge-in. Drop retained audio and move past the abandoned turn.

        Sync, which the Protocol permits (`None | Coroutine`), and preferable: it can be called
        straight from the RPC handler with nothing scheduled. `AvStream.clear` is sync for the
        same reason.

        **Advancing the epoch is the half that makes the video stop too.** Clearing the audio
        queue alone would leave frames already rendered for the abandoned turn queued in the
        presenter and still current, so the mouth would keep speaking a sentence nobody can
        hear. With the epoch moved, those frames are stale by the existing `offer` check — the
        same mechanism that has always handled cancellation, now reachable from an RPC in
        another process.
        """
        self._stream.clear()
        advance = getattr(self._epoch, "advance", None)
        if callable(advance):
            advance()

    async def __aiter__(self) -> AsyncIterator[Any]:
        """
        Yield video and audio interleaved, converting each to LiveKit's types on the way out.

        The ordering is `AvStream`'s and is tested there. This loop only translates.
        """
        async for item in self._stream.stream():
            if isinstance(item, SegmentEnd):
                yield self._segment_end()
            elif isinstance(item, AudioChunk):
                yield self._to_audio_frame(item)
            else:
                yield self._to_video_frame(item)

    # -- conversion ---------------------------------------------------------

    def _segment_end(self) -> Any:
        from livekit.agents.voice.avatar import AudioSegmentEnd

        return AudioSegmentEnd()

    def _to_audio_frame(self, chunk: AudioChunk) -> Any:
        from livekit import rtc

        samples = len(chunk.pcm) // (BYTES_PER_SAMPLE * self._channels)
        return rtc.AudioFrame(
            data=chunk.pcm,
            sample_rate=self._sample_rate,
            num_channels=self._channels,
            samples_per_channel=samples,
        )

    def _to_video_frame(self, frame: Frame) -> Any:
        """
        Our `Frame` to an `rtc.VideoFrame`. Requires a raw codec.

        **An encoded frame is refused rather than decoded.** Our renderers can produce JPEG for
        the
        WebSocket transport or RGB24 for this path, chosen by `AVATAR_FRAME_CODEC`. Decoding a
        JPEG here would undo an encode this path should never have paid for — 23.7 ms/frame
        measured,
        MEASUREMENTS §2.2 — and would ship a working-looking pipeline doing round-trip work. So
        a misconfigured deployment fails immediately with the setting to change, rather than
        serving a slightly worse interview forever.

        `Frame.is_raw` is what enforces the dimensions: a raw buffer is not self-describing, and
        libwebrtc given the wrong stride renders a sheared image.
        """
        from livekit import rtc

        if not frame.is_raw:
            raise ValueError(
                f"this path publishes raw pixels and got a {frame.codec!r} frame. Set "
                f"AVATAR_FRAME_CODEC={FrameCodec.RGB24} on the renderer -- decoding it here "
                "would undo an encode measured at 23.7 ms/frame."
            )
        return rtc.VideoFrame(
            width=frame.width,
            height=frame.height,
            type=rtc.VideoBufferType.RGB24,
            data=frame.data,
        )
