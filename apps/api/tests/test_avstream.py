"""
The interleaved audio/video sequence, tested without a GPU, an SFU, or LiveKit installed.

**Why this is worth testing hard before any framework is involved.** The whole reason to publish
audio and video through one sequence is that two publishers on two clocks produced a measured
trailing gap of −66 ms to +172 ms, with turns up to 538 ms late. Replacing that with one
sequence only helps if the sequence is actually ordered correctly — and the failure modes are
quiet ones. A terminator that overtakes its final chunk, audio held back to keep a frame cadence
tidy, a barge-in that cuts video and keeps speaking: none of those raise, and all of them
present as "the avatar feels wrong".

So every ordering property a synchroniser will depend on is asserted here, deterministically,
with `take_audio`/`take_frame` rather than the paced loop. The paced loop is tested separately
on a fake clock; mixing the two would make an ordering failure look like a timing flake.
"""

from __future__ import annotations

import pytest

from avatar.avstream import AvStream, SegmentEnd
from avatar.contracts import IDLE_EPOCH, AudioChunk, Frame
from avatar.presentation import FramePresenter, IdleLoop
from avatar.state import FrameSource
from avatar.telemetry import NullSink, Telemetry

INTERVAL_MS = 40


class FakeClock:
    """A clock the test advances, so cadence is asserted rather than waited for."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def build(frames: int = 3) -> tuple[AvStream, FramePresenter, FakeClock]:
    presenter = FramePresenter(
        IdleLoop([f"idle-{i}".encode() for i in range(frames)], [0]),
        Telemetry(sink=NullSink()),
    )
    clock = FakeClock()
    stream = AvStream(
        presenter, frame_interval_ms=INTERVAL_MS, clock=clock, sleep=clock.sleep
    )
    return stream, presenter, clock


def chunk(tag: str, epoch: int = 1, duration_ms: int = 20) -> AudioChunk:
    return AudioChunk(pcm=tag.encode(), epoch=epoch, duration_ms=duration_ms)


def rendered(tag: str, epoch: int = 1) -> Frame:
    return Frame(data=tag.encode(), epoch=epoch, pts_ms=0)


# -- ordering, which is the entire point --------------------------------------


def test_audio_comes_out_in_the_order_it_went_in() -> None:
    stream, _, _ = build()
    for tag in ("a", "b", "c"):
        stream.offer_audio(chunk(tag))

    got = [stream.take_audio() for _ in range(3)]

    assert [item.pcm for item in got] == [b"a", b"b", b"c"]  # type: ignore[union-attr]
    assert stream.take_audio() is None


def test_a_terminator_stays_behind_the_audio_it_terminates() -> None:
    """
    The failure this prevents is subtle and visible.

    A terminator that overtook its final chunk would tell the consumer playback had finished
    while a chunk was still to come — and the result on screen is an avatar that stops moving a
    beat before it stops talking. Queued, not emitted directly, for exactly this reason.
    """
    stream, _, _ = build()
    stream.offer_audio(chunk("first"))
    stream.offer_audio(chunk("last"))
    stream.end_segment(epoch=1)

    got = [stream.take_audio() for _ in range(3)]

    assert isinstance(got[2], SegmentEnd)
    assert [item.pcm for item in got[:2]] == [b"first", b"last"]  # type: ignore[union-attr]


def test_the_terminator_names_the_turn_it_ends() -> None:
    """
    LiveKit's own marker carries nothing, which is fine for driving `notify_playback_finished`.

    Ours has to survive a barge-in landing between a turn's last chunk and its terminator;
    without the epoch, a stale terminator would report the wrong turn finished.
    """
    stream, _, _ = build()
    stream.end_segment(epoch=7)

    item = stream.take_audio()

    assert item == SegmentEnd(epoch=7)


# -- the video half -----------------------------------------------------------


def test_frames_are_stamped_monotonically_at_the_interval() -> None:
    stream, _, _ = build()

    stamps = [stream.take_frame().pts_ms for _ in range(4)]

    assert stamps == [0, INTERVAL_MS, INTERVAL_MS * 2, INTERVAL_MS * 3]


def test_a_frame_is_always_available_even_with_nothing_rendered() -> None:
    """
    A consumer starved of video stalls its track, which is worse than a repeated frame and
    corrupts the receiver's jitter estimate. So the idle loop covers the gap.
    """
    stream, _, _ = build()

    frames = [stream.take_frame() for _ in range(3)]

    assert all(f.epoch == IDLE_EPOCH for f in frames)
    assert [f.data for f in frames] == [b"idle-0", b"idle-1", b"idle-2"]


def test_rendered_frames_are_preferred_once_the_source_switches() -> None:
    stream, presenter, _ = build()
    presenter.set_source(FrameSource.RENDERER)
    presenter.offer(rendered("spoken"), 1)

    frame = stream.take_frame()

    assert frame.data == b"spoken"
    assert frame.epoch == 1


# -- barge-in -----------------------------------------------------------------


def test_clear_drops_pending_audio_and_counts_it() -> None:
    """
    Without this, a barge-in cut the video and kept speaking.

    `FramePresenter.set_source` already drains rendered frames when the source returns to idle.
    Retained audio had no equivalent, which is the gap this closes — and the count exists
    because dropping speech already begun is a different event from dropping speech still
    queued.
    """
    stream, _, _ = build()
    for tag in ("a", "b", "c"):
        stream.offer_audio(chunk(tag))

    stream.clear()

    assert stream.pending_audio() == 0
    assert stream.audio_dropped == 3
    assert stream.take_audio() is None


def test_clear_also_drops_a_queued_terminator() -> None:
    """
    A terminator surviving a barge-in would announce that the abandoned turn finished playing.
    """
    stream, _, _ = build()
    stream.offer_audio(chunk("a"))
    stream.end_segment(epoch=1)

    stream.clear()

    assert stream.take_audio() is None
    assert stream.audio_dropped == 2


def test_clear_leaves_the_video_side_to_the_presenter() -> None:
    """
    Two owners, one barge-in, and the split is deliberate.

    Frames are discarded by the presenter when the source goes back to idle, because that is
    where the source decision lives and the pull-based mixer needs the same behaviour.
    Duplicating the drain here would double-count `frames_discarded`.
    """
    stream, presenter, _ = build()
    presenter.set_source(FrameSource.RENDERER)
    presenter.offer(rendered("a"), 1)

    stream.clear()

    assert presenter.buffered() == 1, "clear() reached into the presenter's queue"
    presenter.set_source(FrameSource.IDLE_LOOP)
    assert presenter.buffered() == 0
    assert presenter.frames_discarded == 1


def test_the_stream_recovers_after_a_clear() -> None:
    """A barge-in ends a turn; it does not end the session."""
    stream, _, _ = build()
    stream.offer_audio(chunk("dropped"))
    stream.clear()

    stream.offer_audio(chunk("kept", epoch=2))

    item = stream.take_audio()
    assert item is not None and item.pcm == b"kept"  # type: ignore[union-attr]
    assert stream.take_frame().epoch == IDLE_EPOCH


# -- the paced loop -----------------------------------------------------------


async def drain(stream: AvStream, items: int) -> list[object]:
    got: list[object] = []
    async for item in stream.stream():
        got.append(item)
        if len(got) >= items:
            break
    return got


@pytest.mark.asyncio
async def test_the_loop_emits_all_pending_audio_before_each_frame() -> None:
    """
    The stated policy: audio the moment it arrives, video on the interval.

    Audio wins ties because it is what the candidate hears — a late frame is a frozen mouth for
    one interval, a late chunk is a gap in speech, and only one of those is recoverable.
    """
    stream, _, _ = build()
    for tag in ("a", "b"):
        stream.offer_audio(chunk(tag))

    got = await drain(stream, 3)

    assert [type(item).__name__ for item in got] == ["AudioChunk", "AudioChunk", "Frame"]


@pytest.mark.asyncio
async def test_the_loop_still_emits_video_with_no_audio_at_all() -> None:
    """Silence is the common case — the persona stands by far longer than it speaks."""
    stream, _, _ = build()

    got = await drain(stream, 3)

    assert all(isinstance(item, Frame) for item in got)


@pytest.mark.asyncio
async def test_the_cadence_is_a_deadline_not_a_fixed_sleep() -> None:
    """
    Computed against a monotonic deadline so a slow iteration does not accumulate drift.

    The same reason `FrameMixer.stream` does it that way: sleeping a fixed interval after
    variable work means the real frame rate is always below the target and drifts further the
    longer a session runs.
    """
    stream, _, clock = build()

    await drain(stream, 3)

    # Two sleeps for three frames: the loop sleeps *after* yielding, so the consumer breaking on
    # the third frame never reaches the third sleep. Asserted as n-1 rather than n so the shape
    # of the loop is pinned -- a version that slept before yielding would stall the first frame
    # by a whole interval, which is a real regression and would pass a laxer assertion.
    assert clock.slept == pytest.approx([INTERVAL_MS / 1000.0] * 2, abs=1e-9)


@pytest.mark.asyncio
async def test_counters_report_what_was_emitted() -> None:
    stream, _, _ = build()
    stream.offer_audio(chunk("a"))

    await drain(stream, 3)

    assert stream.audio_emitted == 1
    assert stream.frames_emitted == 2


# -- the pairing budget, which the first policy got wrong ----------------------


@pytest.mark.asyncio
async def test_audio_is_bounded_to_one_frame_interval_per_frame() -> None:
    """
    The bug this pins, measured before it was fixed: audio and video at 32:1.

    The first policy drained every pending chunk before each frame, reasoning that audio must
    never be held back. But a TTS with a real-time factor below 1 delivers a whole utterance in
    a fraction of its playback duration, so "everything pending" is almost everything — and live
    it produced 16 frames where 221 were needed. The budget has to be time, not count.

    40 ms of video pairs with 40 ms of audio, so with 20 ms chunks the ratio is exactly 2:1.
    """
    stream, _, _ = build()
    for index in range(60):
        stream.offer_audio(chunk(f"a{index}"))

    got = await drain(stream, 60)

    audio = sum(1 for item in got if isinstance(item, AudioChunk))
    video = sum(1 for item in got if isinstance(item, Frame))
    assert audio / video == pytest.approx(2.0), f"{audio} audio to {video} video"


@pytest.mark.asyncio
async def test_a_frame_is_emitted_even_when_the_audio_budget_is_unspent() -> None:
    """
    Video is never held waiting for audio.

    A consumer starved of video stalls its track, which is worse than a mouth briefly ahead of
    the words — and the audio may never arrive at all, since silence is the normal state between
    turns.
    """
    stream, _, _ = build()
    stream.offer_audio(chunk("only-20ms"))

    got = await drain(stream, 2)

    assert isinstance(got[0], AudioChunk)
    assert isinstance(got[1], Frame), "the frame waited for a full budget of audio"


@pytest.mark.asyncio
async def test_a_terminator_does_not_consume_the_audio_budget() -> None:
    """
    It costs no playback time, so charging it one would delay the audio it terminates by a
    frame.
    """
    stream, _, _ = build()
    stream.offer_audio(chunk("a"))
    stream.end_segment(epoch=1)
    stream.offer_audio(chunk("b", epoch=2))

    got = await drain(stream, 4)

    assert [type(item).__name__ for item in got] == [
        "AudioChunk",
        "SegmentEnd",
        "AudioChunk",
        "Frame",
    ]


@pytest.mark.asyncio
async def test_a_chunk_longer_than_the_frame_interval_still_pairs_by_duration() -> None:
    """
    The second bug, and the one that only shows up in production.

    Deepgram delivers roughly 78 ms of audio per chunk against a 40 ms frame. Subtracting after
    yielding meant one oversized chunk always got through and consumed the whole budget, so the
    ratio collapsed to one chunk per frame *by count* — running audio at about twice video and
    leaving 143-161 frames per turn queued and then discarded. Overshoot is carried forward as
    debt.

    Asserted as a rate rather than a ratio, because the ratio is supposed to change with chunk
    size while the rate must not.
    """
    stream, _, _ = build()
    long_ms = INTERVAL_MS * 2  # a chunk twice the frame interval
    for index in range(40):
        stream.offer_audio(chunk(f"a{index}", duration_ms=long_ms))

    got = await drain(stream, 60)

    audio = [item for item in got if isinstance(item, AudioChunk)]
    video = [item for item in got if isinstance(item, Frame)]
    audio_ms = len(audio) * long_ms
    video_ms = len(video) * INTERVAL_MS
    assert audio_ms / video_ms == pytest.approx(1.0, abs=0.05), (
        f"{audio_ms} ms of audio against {video_ms} ms of video"
    )
