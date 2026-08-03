"""
The parts of the LiveKit binding that can be tested without LiveKit.

**What is and is not covered, stated plainly.** `push_audio` and `clear_buffer` are pure
translation into `AvStream` and are tested here with duck-typed stand-ins for `rtc.AudioFrame`
and `AudioSegmentEnd` — LiveKit's terminator has no payload and its audio frame does, which is
exactly what the binding dispatches on, so a stand-in exercises the real branch.

`__aiter__`'s conversion to `rtc.VideoFrame` is **not** covered and cannot be until the renderer
has a raw-buffer output path: our frames are JPEG bytes for the WebSocket transport, and
`rtc.VideoFrame` wants I420 or RGBA. That method raises with the reason rather than doing a
decode/re-encode round trip that would look like it worked, and the test below asserts it raises
— which is the honest thing to pin until the renderer changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from avatar.avstream import AvStream, SegmentEnd
from avatar.contracts import AudioChunk, Frame
from avatar.presentation import FramePresenter, IdleLoop
from avatar.telemetry import NullSink, Telemetry
from avatar.transport.livekit_avatar import LiveKitVideoGenerator

SAMPLE_RATE = 16_000


@dataclass
class FakeAudioFrame:
    """Stands in for `rtc.AudioFrame`. The binding only reads `.data`."""

    data: bytes


class FakeSegmentEnd:
    """Stands in for `AudioSegmentEnd`: no payload, which is the branch the binding takes."""


def build(epoch: int = 3) -> tuple[LiveKitVideoGenerator, AvStream]:
    presenter = FramePresenter(IdleLoop([b"idle"], [0]), Telemetry(sink=NullSink()))
    stream = AvStream(presenter, frame_interval_ms=40)
    generator = LiveKitVideoGenerator(
        stream, sample_rate=SAMPLE_RATE, epoch=lambda: epoch
    )
    return generator, stream


@pytest.mark.asyncio
async def test_an_audio_frame_becomes_a_chunk_with_a_real_duration() -> None:
    """
    Duration is derived from the byte count, not assumed.

    The orchestrator reasons about how much of a turn the candidate has heard using
    `duration_ms`, so a wrong value here corrupts history truncation — silently, and in the
    direction that makes the interviewer refer to sentences nobody heard.
    """
    generator, stream = build()
    # 320 samples of 16-bit mono at 16 kHz = 20 ms.
    await generator.push_audio(FakeAudioFrame(data=b"\x00\x00" * 320))

    item = stream.take_audio()

    assert isinstance(item, AudioChunk)
    assert item.duration_ms == 20
    assert item.epoch == 3


@pytest.mark.asyncio
async def test_a_payloadless_frame_becomes_a_terminator() -> None:
    """LiveKit's `AudioSegmentEnd` has no data; that absence is the whole signal."""
    generator, stream = build(epoch=5)

    await generator.push_audio(FakeSegmentEnd())

    assert stream.take_audio() == SegmentEnd(epoch=5)


@pytest.mark.asyncio
async def test_the_epoch_is_injected_and_has_no_default() -> None:
    """
    A default would have made the in-process case work and the split-process case silently
    wrong.

    Across a boundary the epoch is not on the wire — LiveKit's frames carry samples and a rate —
    so the sender must supply it. Requiring it at construction is what stops that from being
    discovered in production.
    """
    with pytest.raises(TypeError):
        LiveKitVideoGenerator(  # type: ignore[call-arg]
            build()[1], sample_rate=SAMPLE_RATE
        )


def test_clear_buffer_is_synchronous_and_drops_pending_audio() -> None:
    """
    Sync is permitted by the Protocol (`None | Coroutine`) and preferable.

    It can be called straight from the RPC handler with nothing scheduled, which matters because
    the whole value of `lk.clear_buffer` is that a barge-in reaches the renderer promptly.
    """
    generator, stream = build()
    stream.offer_audio(AudioChunk(pcm=b"\x00\x00", epoch=3, duration_ms=1))

    result = generator.clear_buffer()

    assert result is None, "clear_buffer returned a coroutine; the RPC path expects sync"
    assert stream.pending_audio() == 0
    assert stream.audio_dropped == 1


def test_video_conversion_refuses_an_encoded_frame_and_names_the_setting() -> None:
    """
    Refused, not decoded, because the alternative looks like success.

    Decoding a JPEG here would undo an encode this path should never have paid for -- 23.7
    ms/frame measured -- and would ship a working-looking pipeline doing round-trip work. A
    misconfigured deployment should fail at once naming the variable to change, rather than
    serving a slightly worse interview forever.
    """
    generator, _ = build()

    with pytest.raises(ValueError, match="AVATAR_FRAME_CODEC"):
        generator._to_video_frame(
            Frame(data=b"\xff\xd8jpeg", epoch=1, pts_ms=0, codec="jpeg")
        )


def test_a_raw_frame_without_dimensions_is_refused_by_the_contract() -> None:
    """
    `Frame.is_raw` raises rather than letting an undescribed buffer through.

    libwebrtc given the wrong stride renders a sheared image, and tracing that back to a
    missing integer three layers up is an afternoon nobody should spend.
    """
    generator, _ = build()

    with pytest.raises(ValueError, match="width and height"):
        generator._to_video_frame(
            Frame(data=b"\x00" * 12, epoch=1, pts_ms=0, codec="rgb24")
        )
