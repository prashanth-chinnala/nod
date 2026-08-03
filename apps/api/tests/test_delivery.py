"""
Paired delivery: one sequence out, and an orchestrator that cannot tell the difference.

The point of this file is that `PairedDelivery` is a decorator. If the orchestrator can detect
it, the abstraction has leaked and every session-layer test becomes mode-dependent — so the
assertions here are mostly about what stays the same. The one behaviour that genuinely changes
is where audio goes, and the one that must not is barge-in.
"""

from __future__ import annotations

import pytest

from avatar.contracts import AudioChunk, Frame
from avatar.delivery import PairedDelivery
from avatar.presentation import FramePresenter, IdleLoop
from avatar.state import FrameSource
from avatar.telemetry import NullSink, Telemetry

INTERVAL_MS = 40


class FakeTransport:
    """Records what reached the wire, in order, so ordering can be asserted."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []
        self.flushes = 0
        self.opened = 0
        self.closed = 0

    async def open_track(self) -> None:
        self.opened += 1

    async def close_track(self) -> None:
        self.closed += 1

    async def send_audio(self, chunk: AudioChunk) -> None:
        self.sent.append(("audio", chunk))

    async def send_frame(self, frame: Frame) -> None:
        self.sent.append(("frame", frame))

    async def flush_audio(self) -> None:
        self.flushes += 1


def build() -> tuple[PairedDelivery, FakeTransport, FramePresenter]:
    presenter = FramePresenter(
        IdleLoop([b"idle-0", b"idle-1"], [0], codec="png", width=8, height=8),
        Telemetry(sink=NullSink()),
    )
    inner = FakeTransport()
    return (
        PairedDelivery(inner, presenter, frame_interval_ms=INTERVAL_MS),
        inner,
        presenter,
    )


def chunk(tag: str, epoch: int = 1) -> AudioChunk:
    return AudioChunk(pcm=tag.encode(), epoch=epoch, duration_ms=20)


async def drain(delivery: PairedDelivery, steps: int) -> None:
    n = 0
    async for _ in delivery.pump():
        n += 1
        if n >= steps:
            return


# -- the decorator must be invisible ------------------------------------------


@pytest.mark.asyncio
async def test_audio_is_queued_rather_than_written() -> None:
    """
    The one intended behavioural change.

    `send_audio` returning before the wire is reached is what stops audio overtaking video by
    however long a socket write takes. The orchestrator only uses this call's completion for its
    own `audio_sent_ms` bookkeeping, which counts what was handed over, so nothing it relies on
    changes.
    """
    delivery, inner, _ = build()

    await delivery.send_audio(chunk("a"))

    assert inner.sent == [], "audio reached the wire before the sequence"
    assert delivery.stream.pending_audio() == 1


@pytest.mark.asyncio
async def test_lifecycle_calls_pass_straight_through() -> None:
    """Anything not intercepted must reach the transport, or the wrapper is a black hole."""
    delivery, inner, _ = build()

    await delivery.open_track()
    await delivery.close_track()

    assert (inner.opened, inner.closed) == (1, 1)


@pytest.mark.asyncio
async def test_frames_offered_directly_are_forwarded_not_requeued() -> None:
    """
    The frame path already runs through the presenter, which the sequence reads.

    Queuing a frame here as well would emit it twice — once from the caller and once from the
    sequence — which on screen is a stutter rather than an error.
    """
    delivery, inner, _ = build()
    frame = Frame(data=b"x", epoch=1, pts_ms=0, codec="png", width=8, height=8)

    await delivery.send_frame(frame)

    assert inner.sent == [("frame", frame)]


# -- what actually goes out ---------------------------------------------------


@pytest.mark.asyncio
async def test_the_pump_sends_audio_and_video_from_one_loop() -> None:
    """
    Both media, one sequence, audio first — the ordering `AvStream` guarantees and this
    forwards.
    """
    delivery, inner, presenter = build()
    presenter.set_source(FrameSource.RENDERER)
    await delivery.send_audio(chunk("a"))
    await delivery.send_audio(chunk("b"))

    await drain(delivery, 3)

    assert [kind for kind, _ in inner.sent] == ["audio", "audio", "frame"]
    assert delivery.audio_sent == 2
    assert delivery.frames_sent == 1


@pytest.mark.asyncio
async def test_video_still_flows_with_no_audio_at_all() -> None:
    """Standing by is most of an interview, and the track must not stall through it."""
    delivery, inner, _ = build()

    await drain(delivery, 2)

    assert [kind for kind, _ in inner.sent] == ["frame", "frame"]


@pytest.mark.asyncio
async def test_a_segment_end_puts_nothing_on_the_wire() -> None:
    """
    The WebSocket client infers the end of an utterance from the audio stopping.

    `AvatarRunner` uses the marker to fire `notify_playback_finished`; there is no equivalent
    here, and inventing a message the client does not read would be worse than dropping it.
    """
    delivery, inner, _ = build()
    delivery.stream.end_segment(epoch=1)

    await drain(delivery, 2)

    assert all(kind != "audio" for kind, _ in inner.sent)
    assert delivery.audio_sent == 0


# -- barge-in, which must not regress ----------------------------------------


@pytest.mark.asyncio
async def test_a_flush_clears_the_queue_and_tells_the_client() -> None:
    """
    Both, and locally first.

    Reversed, a chunk queued between the two calls would be sent after the client had already
    been told to discard — and the candidate hears a fragment of the sentence they interrupted.
    """
    delivery, inner, _ = build()
    await delivery.send_audio(chunk("doomed"))

    await delivery.flush_audio()

    assert delivery.stream.pending_audio() == 0
    assert delivery.stream.audio_dropped == 1
    assert inner.flushes == 1


@pytest.mark.asyncio
async def test_nothing_queued_before_a_flush_reaches_the_wire_after_it() -> None:
    """The property the ordering above exists to produce, asserted end to end."""
    delivery, inner, _ = build()
    await delivery.send_audio(chunk("doomed"))
    await delivery.flush_audio()
    await delivery.send_audio(chunk("kept", epoch=2))

    await drain(delivery, 2)

    audio = [item for kind, item in inner.sent if kind == "audio"]
    assert [item.pcm for item in audio] == [b"kept"]  # type: ignore[union-attr]
