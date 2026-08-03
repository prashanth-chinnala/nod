"""
The frame-selection contract, tested independently of how frames get delivered.

**Why this file exists separately from `test_mixer_cadence.py`.** That file drives the pull
loop: one frame per tick, monotonic timestamps, cadence under a fake clock. This one tests the
decision underneath it — which frame, from which source, and what happens when the renderer is
behind — with no clock involved at all.

The split matters because a second delivery model is coming. A LiveKit `VideoGenerator` pushes
frames into an `AVSynchronizer` instead of being pulled once per tick, and it needs exactly the
same decisions made: idle when nobody is speaking, a clean seam before cutting to speech, stale
epochs dropped. If those were only ever asserted through the cadence loop, the second consumer
would have no test to inherit — and the clean-exit seam is the highest-regression surface in
this repo.

So every assertion here is about `FramePresenter` alone, and every one of them is a promise the
second consumer gets to rely on.
"""

from __future__ import annotations

import pytest

from avatar.contracts import IDLE_EPOCH, Frame
from avatar.presentation import FramePresenter, IdleLoop
from avatar.state import FrameSource
from avatar.telemetry import NullSink, Telemetry


def loop(count: int = 3) -> IdleLoop:
    """An idle loop whose frames are identifiable, with frame 0 the only clean exit."""
    return IdleLoop([f"idle-{i}".encode() for i in range(count)], [0])


def presenter(count: int = 3) -> FramePresenter:
    return FramePresenter(loop(count), Telemetry(sink=NullSink()))


def rendered(tag: str, epoch: int = 1, pts_ms: int = 999) -> Frame:
    """
    A rendered frame with a deliberately wrong `pts_ms`.

    999 everywhere on purpose: `take()` must not touch it. Whoever delivers the frame owns the
    clock, and a presenter that quietly stamped one would put two clocks back in the system.
    """
    return Frame(data=tag.encode(), epoch=epoch, pts_ms=pts_ms)


# -- the invariant both delivery models depend on -----------------------------


def test_take_always_returns_a_frame_with_nothing_queued() -> None:
    """
    Never blocks, never None, from a cold start.

    A cadence loop must have something to send every tick or the track stalls, which is more
    visible than a repeated frame and corrupts the receiver's jitter estimate. A push-based
    generator has the same requirement for the same reason.
    """
    p = presenter()

    frames = [p.take() for _ in range(5)]

    assert all(isinstance(f, Frame) for f in frames)
    assert all(f.epoch == IDLE_EPOCH for f in frames)


def test_take_does_not_stamp_a_timestamp() -> None:
    """
    The presenter owns no clock, and this is what keeps it that way.

    `AVSynchronizer` will hold the clock when frames are published to a room; the cadence loop
    holds it today. A presenter that stamped would make three clocks out of two.
    """
    p = presenter()
    p.set_source(FrameSource.RENDERER)
    p.offer(rendered("a", pts_ms=999), 1)

    assert p.take().pts_ms == 999, "take() rewrote pts_ms"


def test_idle_frames_advance_and_wrap() -> None:
    p = presenter(count=3)

    assert [p.take().data for _ in range(4)] == [b"idle-0", b"idle-1", b"idle-2", b"idle-0"]


# -- source selection ---------------------------------------------------------


def test_the_renderer_is_preferred_once_selected() -> None:
    p = presenter()
    p.set_source(FrameSource.RENDERER)
    p.offer(rendered("spoken"), 1)

    frame = p.take()

    assert frame.data == b"spoken"
    assert frame.epoch == 1


def test_idle_still_plays_while_the_renderer_warms_up() -> None:
    """
    Selected but not yet producing is a real state, and a live idle frame beats a frozen one.

    This is the window between the state machine choosing to speak and the first rendered frame
    arriving — measured at ~1.5 s on a T4, so it is not a corner case.
    """
    p = presenter()
    p.set_source(FrameSource.RENDERER)

    frame = p.take()

    assert frame.epoch == IDLE_EPOCH
    assert frame.data == b"idle-0"


def test_returning_to_idle_discards_the_abandoned_turn() -> None:
    """
    Frames queued for a turn we have stopped showing must not appear later.

    Showing them would animate a mouth in silence. The counter is the honest record of how much
    video was behind — it is not evidence of a slow renderer, which is how it was misread for a
    long time.
    """
    p = presenter()
    p.set_source(FrameSource.RENDERER)
    for tag in ("a", "b", "c"):
        p.offer(rendered(tag), 1)
    assert p.buffered() == 3

    p.set_source(FrameSource.IDLE_LOOP)

    assert p.buffered() == 0
    assert p.frames_discarded == 3
    assert p.take().epoch == IDLE_EPOCH


def test_setting_the_same_source_twice_does_not_discard() -> None:
    """`_transition` fires on every state change, including ones that keep the source."""
    p = presenter()
    p.set_source(FrameSource.RENDERER)
    p.offer(rendered("a"), 1)

    p.set_source(FrameSource.RENDERER)

    assert p.buffered() == 1
    assert p.frames_discarded == 0


# -- starvation ---------------------------------------------------------------


def test_a_dry_queue_repeats_the_last_frame_and_counts_it() -> None:
    """
    The mouth freezes for one interval rather than the track stalling.

    `frames_repeated` climbing is the signal that the renderer is behind real time, and it is
    the one a pure fps average hides.
    """
    p = presenter()
    p.set_source(FrameSource.RENDERER)
    p.offer(rendered("only"), 1)

    first, second, third = p.take(), p.take(), p.take()

    assert [f.data for f in (first, second, third)] == [b"only"] * 3
    assert p.frames_repeated == 2


def test_a_repeat_is_not_counted_as_a_discard() -> None:
    """Two different signals: one means behind, the other means abandoned."""
    p = presenter()
    p.set_source(FrameSource.RENDERER)
    p.offer(rendered("a"), 1)
    p.take()
    p.take()

    assert p.frames_repeated == 1
    assert p.frames_discarded == 0


def test_after_returning_to_idle_the_stale_frame_is_not_repeated() -> None:
    """
    `_last_rendered` is cleared on the way back to idle.

    Without that, the next turn's warm-up window would repeat a frame from the *previous* turn —
    the wrong mouth shape for the wrong words, which reads as a broken model.
    """
    p = presenter()
    p.set_source(FrameSource.RENDERER)
    p.offer(rendered("old"), 1)
    p.take()

    p.set_source(FrameSource.IDLE_LOOP)
    p.set_source(FrameSource.RENDERER)

    assert p.take().epoch == IDLE_EPOCH, "a frame from the abandoned turn came back"
    assert p.frames_repeated == 0


# -- cancellation -------------------------------------------------------------


def test_a_stale_epoch_is_refused() -> None:
    """
    The consumer-side half of cancellation.

    The renderer may keep producing for an abandoned turn; those frames die here rather than
    requiring an interruptible renderer. This has to survive the renderer becoming a separate
    process, where interrupting the producer is a round trip and this is still one comparison.
    """
    p = presenter()
    p.set_source(FrameSource.RENDERER)

    assert p.offer(rendered("current", epoch=2), 2) is True
    assert p.offer(rendered("stale", epoch=1), 2) is False
    assert p.buffered() == 1


def test_a_refused_frame_is_not_counted_as_discarded() -> None:
    """
    Different counters for different events.

    A stale offer never entered the queue; `frames_discarded` counts what was queued and then
    abandoned. Conflating them would make the queue-depth signal unreadable.
    """
    p = presenter()
    p.set_source(FrameSource.RENDERER)
    p.offer(rendered("stale", epoch=1), 5)

    assert p.frames_discarded == 0


# -- the seam -----------------------------------------------------------------


def test_the_clean_exit_follows_the_idle_position() -> None:
    """
    Only frame 0 is mouth-closed here, so the seam opens once per lap.

    The orchestrator times the handover on this, and a bounded wait forces the cut if the seam
    never comes. Asserted on the presenter because that is the object the second delivery model
    will ask.
    """
    p = presenter(count=3)

    assert p.at_clean_exit() is True
    p.take()
    assert p.at_clean_exit() is False
    p.take()
    assert p.at_clean_exit() is False
    p.take()
    assert p.at_clean_exit() is True


def test_replacing_the_idle_loop_replaces_the_seam() -> None:
    """
    A renderer supplies an idle loop built from the persona's own frames at session start.

    The seam has to come from that loop, not from the placeholder it replaced — otherwise the
    handover is timed against frames nobody is watching.
    """
    p = presenter(count=3)
    p.set_idle(IdleLoop([b"x", b"y"], [1]))

    assert p.at_clean_exit() is False
    assert p.take().data == b"x"
    assert p.at_clean_exit() is True


# -- codec and dimensions, which a raw consumer cannot work without -----------


def test_the_idle_loop_declares_its_own_codec_and_size() -> None:
    """
    An idle frame claiming the wrong codec would reach the transport as the wrong type in
    exactly the state a candidate spends most of an interview looking at.

    The loop is built by whoever produced the frames — the placeholder generator, or a
    renderer from its own reference — and only that caller knows the format.
    """
    p = FramePresenter(
        IdleLoop([b"raw"], [0], codec="rgb24", width=4, height=1),
        Telemetry(sink=NullSink()),
    )

    frame = p.take()

    assert frame.codec == "rgb24"
    assert (frame.width, frame.height) == (4, 1)
    assert frame.is_raw is True


def test_a_raw_frame_with_no_dimensions_refuses_to_describe_itself() -> None:
    """
    `is_raw` raises rather than answering, because the answer alone is not usable.

    A consumer handed a buffer with the wrong stride renders a sheared image, and the
    missing integer is three layers away by then.
    """
    p = FramePresenter(
        IdleLoop([b"raw"], [0], codec="rgb24"), Telemetry(sink=NullSink())
    )

    frame = p.take()
    with pytest.raises(ValueError, match="width and height"):
        assert frame.is_raw


def test_an_encoded_frame_needs_no_dimensions() -> None:
    """A JPEG or PNG carries its own, which is why the requirement is codec-dependent."""
    p = presenter()

    assert p.take().is_raw is False
