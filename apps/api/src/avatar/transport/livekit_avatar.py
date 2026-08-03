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
    ) -> None:
        self._stream = stream
        self._sample_rate = sample_rate
        self._channels = channels
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

        The epoch comes from the injected source, not from the frame — LiveKit's audio frames
        carry samples and a rate, not our turn number. See `_epoch`: correct in-process, and
        something the sender must put on the wire once the renderer is a separate process.
        """
        data = getattr(frame, "data", None)
        if data is None:
            self._stream.end_segment(epoch=self._epoch())
            return

        pcm = bytes(data)
        samples = len(pcm) // (BYTES_PER_SAMPLE * self._channels)
        duration_ms = round(samples / self._sample_rate * 1000)
        self._stream.offer_audio(
            AudioChunk(pcm=pcm, epoch=self._epoch(), duration_ms=duration_ms)
        )

    def clear_buffer(self) -> None:
        """
        Barge-in. Drop retained audio; the presenter drops frames when the source returns to
        idle.

        Sync, which the Protocol permits (`None | Coroutine`), and preferable: it can be called
        straight from the RPC handler with nothing scheduled. `AvStream.clear` is sync for the
        same reason.
        """
        self._stream.clear()

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
