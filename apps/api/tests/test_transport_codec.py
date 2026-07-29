"""
The wire protocol, and the transport that speaks it.

Tested without FastAPI on purpose: `WebSocketTransport` takes two send callables, so the codec
and the `Transport` implementation are exercisable with a list as the socket. That keeps the web
stack out of the CI dependency set and keeps this file fast, and it is the same reason the
orchestrator takes a `Transport` rather than a WebSocket.

The web client reimplements `decode` in TypeScript (`apps/web/src/lib/session.ts`), so the
header layout is duplicated across two languages by necessity. The round-trip tests here are
what pin the byte layout the two sides have to agree on -- network byte order for the header,
little-endian for the PCM payload, which is an easy pair to get backwards.
"""

from __future__ import annotations

import pytest

from avatar.contracts import IDLE_EPOCH, AudioChunk, Frame
from avatar.transport.websocket import (
    HEADER_SIZE,
    MAX_PAYLOAD,
    Kind,
    WebSocketTransport,
    decode,
    encode,
    encode_audio,
    encode_frame,
)


class FakeSocket:
    def __init__(self) -> None:
        self.binary: list[bytes] = []
        self.text: list[str] = []

    async def send_bytes(self, data: bytes) -> None:
        self.binary.append(data)

    async def send_text(self, data: str) -> None:
        self.text.append(data)


# -- codec -----------------------------------------------------------------


def test_round_trip_preserves_every_header_field() -> None:
    message = encode(Kind.VIDEO, 1234, 7, b"payload")

    envelope = decode(message)

    assert envelope.kind is Kind.VIDEO
    assert envelope.pts_ms == 1234
    assert envelope.epoch == 7
    assert envelope.payload == b"payload"


def test_header_is_thirteen_bytes() -> None:
    """The JS client hard-codes this. If it changes, that changes too."""
    assert HEADER_SIZE == 13
    assert len(encode(Kind.AUDIO, 0, 0, b"")) == HEADER_SIZE


def test_frame_encoding_carries_the_epoch_for_client_side_filtering() -> None:
    envelope = decode(encode_frame(Frame(data=b"bmp", epoch=4, pts_ms=880)))

    assert (envelope.kind, envelope.epoch, envelope.pts_ms) == (Kind.VIDEO, 4, 880)


def test_idle_frames_encode_with_the_idle_epoch() -> None:
    envelope = decode(encode_frame(Frame(data=b"idle", epoch=IDLE_EPOCH, pts_ms=40)))

    assert envelope.epoch == IDLE_EPOCH


def test_audio_encoding_uses_the_supplied_timeline_not_the_chunk() -> None:
    """
    Audio pts comes from the transport's running cursor.

    `AudioChunk` has a duration but no position -- the orchestrator produces chunks without
    knowing where in the session they land, and only the transport is in a position to say.
    """
    chunk = AudioChunk(pcm=b"\x01\x02", epoch=3, duration_ms=80)

    envelope = decode(encode_audio(chunk, pts_ms=560))

    assert (envelope.kind, envelope.pts_ms, envelope.payload) == (Kind.AUDIO, 560, b"\x01\x02")


@pytest.mark.parametrize(
    ("message", "match"),
    [
        (b"", "shorter than the header"),
        (b"\x01\x00", "shorter than the header"),
    ],
)
def test_truncated_messages_are_rejected(message: bytes, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        decode(message)


def test_payload_shorter_than_the_header_claims_is_rejected() -> None:
    # A truncated frame must be an error, not a canvas full of garbage.
    corrupted = encode(Kind.VIDEO, 0, 1, b"12345678")[:-3]

    with pytest.raises(ValueError, match="header declares"):
        decode(corrupted)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown message kind"):
        decode(encode(Kind.VIDEO, 0, 0, b"x").replace(b"\x01", b"\x63", 1))


def test_oversized_payload_is_refused_at_encode_time() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        encode(Kind.VIDEO, 0, 0, b"\x00" * (MAX_PAYLOAD + 1))


def test_negative_header_values_are_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        encode(Kind.VIDEO, -1, 0, b"")


# -- transport -------------------------------------------------------------


async def test_open_and_close_bracket_the_track() -> None:
    socket = FakeSocket()
    transport = WebSocketTransport(socket.send_bytes, socket.send_text)

    await transport.open_track()
    assert transport.open is True
    await transport.close_track()

    assert transport.open is False
    assert socket.text == ['{"type":"track_open"}', '{"type":"track_close"}']


async def test_audio_timeline_advances_by_chunk_duration() -> None:
    socket = FakeSocket()
    transport = WebSocketTransport(socket.send_bytes, socket.send_text)

    for _ in range(3):
        await transport.send_audio(AudioChunk(pcm=b"\x00\x00", epoch=1, duration_ms=80))

    assert [decode(m).pts_ms for m in socket.binary] == [0, 80, 160]


async def test_flush_tells_the_client_rather_than_only_dropping_locally() -> None:
    socket = FakeSocket()
    transport = WebSocketTransport(socket.send_bytes, socket.send_text)
    await transport.send_audio(AudioChunk(pcm=b"\x00\x00", epoch=1, duration_ms=80))

    await transport.flush_audio()

    # By the time a barge-in happens the audio is already in the client's Web Audio
    # graph. A flush that only dropped server-side state would leave the candidate
    # listening to an abandoned sentence.
    assert socket.text == ['{"type":"flush_audio"}']


async def test_transport_counts_bytes_for_the_stats_readout() -> None:
    socket = FakeSocket()
    transport = WebSocketTransport(socket.send_bytes, socket.send_text)

    await transport.send_frame(Frame(data=b"x" * 100, epoch=1, pts_ms=0))

    assert transport.bytes_sent == 100 + HEADER_SIZE


# -- webrtc epoch marking --------------------------------------------------
#
# A WebSocket frame is a framed binary message carrying its own epoch, which is what the tests
# above assert. A WebRTC video track is anonymous pixels, so the epoch has to travel beside
# it on the data channel or no painted frame can be attributed to a turn -- which is exactly why
# end-to-end latency was unmeasurable on that path. These pin the marking discipline.


class MarkingTransport:
    """
    A `LiveKitTransport` with the SDK replaced by recorders.

    Subclassed rather than mocked at the module level so the epoch-tracking logic under test is
    the real one; only the two calls that would touch a network are swapped.
    """

    def __init__(self) -> None:
        from avatar.transport.livekit import LiveKitTransport

        self.inner = LiveKitTransport("session-x", width=8, height=8)
        self.controls: list[dict[str, object]] = []
        self.frames: list[int] = []

        async def send_control(payload: dict[str, object]) -> None:
            self.controls.append(payload)

        self.inner.send_control = send_control  # type: ignore[method-assign]

        # A stand-in with the one method `send_frame` reaches for. A bare `object()` is not
        # enough: the attribute is read while building the `asyncio.to_thread` arguments, so it
        # fails before any patch on `to_thread` could intercept it.
        class Source:
            def capture_frame(self, frame: object) -> None:
                return None

        self.inner._video_source = Source()

    async def send(self, epoch: int) -> None:
        from unittest.mock import patch

        from avatar.contracts import Frame

        # The decode-and-capture tail needs Pillow and a real source; the epoch marking happens
        # before it, so it is stubbed out rather than exercised here.
        with patch("avatar.transport.livekit._decode_to_rgba", return_value=(b"", 8, 8)), patch(
            "asyncio.to_thread", new=_noop
        ):
            await self.inner.send_frame(Frame(data=b"x", epoch=epoch, pts_ms=0))
        self.frames.append(epoch)

    @property
    def marked(self) -> list[int]:
        return [int(c["epoch"]) for c in self.controls if c.get("type") == "frame_epoch"]


async def _noop(*args: object, **kwargs: object) -> None:
    return None


@pytest.mark.asyncio
async def test_the_first_frame_of_a_turn_is_marked_on_the_data_channel() -> None:
    """
    Without this the client sees a frame arrive and cannot say which turn it closes.

    Epoch 0 is included deliberately: it is the opening question, and a tracker initialised to 0
    rather than -1 would swallow its marker and leave the first turn of every session
    unmeasurable -- which is the turn a demo looks at.
    """
    transport = MarkingTransport()
    await transport.send(0)
    assert transport.marked == [0]


@pytest.mark.asyncio
async def test_only_the_first_frame_of_each_turn_is_marked() -> None:
    """
    One message per turn, not one per frame.

    Frames leave at 25fps; marking each would spend the data channel on instrumentation and put
    a control message between every pair of frames, which is the same mistake `RELAYED_EVENTS`
    excludes `frame_repeated` for.
    """
    transport = MarkingTransport()
    for epoch in (3, 3, 3, 4, 4, 5):
        await transport.send(epoch)
    assert transport.marked == [3, 4, 5]
    assert len(transport.frames) == 6  # every frame still went out


@pytest.mark.asyncio
async def test_a_repeated_epoch_after_a_gap_is_not_re_marked() -> None:
    """
    The tracker holds the last epoch seen, not a set.

    Epochs only ever increase -- the orchestrator bumps one per turn -- so remembering the last
    is sufficient, and a set would grow for the life of the session to answer a question about
    the present.
    """
    transport = MarkingTransport()
    for epoch in (1, 2, 2, 2):
        await transport.send(epoch)
    assert transport.marked == [1, 2]
