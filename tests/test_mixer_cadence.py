"""
Frame cadence, source switching, and starvation behaviour.

The invariant these tests defend is that the track never stops. Every path
through the mixer produces a frame, including the paths where the renderer has
produced nothing at all -- because a stalled track is more visible to a viewer
than a repeated frame, and it corrupts the receiver's jitter estimate so the
recovery is worse than the stall.
"""

from __future__ import annotations

import pytest

from avatar.contracts import IDLE_EPOCH, Frame
from avatar.mixer import FRAME_INTERVAL, FRAME_INTERVAL_MS, TARGET_FPS, FrameMixer, IdleLoop
from avatar.state import FrameSource
from avatar.telemetry import NullSink, Telemetry
from tests.conftest import FakeClock, make_idle_loop, make_mixer

TURN_EPOCH = 1


def rendered(index: int) -> Frame:
    return Frame(data=f"render-{index}".encode(), epoch=TURN_EPOCH, pts_ms=index * 40)


async def take(mixer: FrameMixer, count: int) -> list[Frame]:
    out: list[Frame] = []
    agen = mixer.stream()
    try:
        async for frame in agen:
            out.append(frame)
            if len(out) == count:
                break
    finally:
        await agen.aclose()
    return out


# -- cadence ---------------------------------------------------------------


async def test_emits_exactly_twenty_five_frames_per_simulated_second(
    clock: FakeClock, telemetry: Telemetry
) -> None:
    mixer = make_mixer(clock, telemetry)

    frames = await take(mixer, TARGET_FPS + 5)

    within_first_second = [f for f in frames if f.pts_ms < 1000]
    assert len(within_first_second) == TARGET_FPS


async def test_presentation_timestamps_are_monotonic_and_evenly_spaced(
    clock: FakeClock, telemetry: Telemetry
) -> None:
    mixer = make_mixer(clock, telemetry)

    frames = await take(mixer, 12)

    assert [f.pts_ms for f in frames] == [i * FRAME_INTERVAL_MS for i in range(12)]


async def test_stream_paces_itself_off_the_injected_clock(
    clock: FakeClock, telemetry: Telemetry
) -> None:
    mixer = make_mixer(clock, telemetry)
    start = clock()

    await take(mixer, 10)

    # Nine sleeps for ten frames: the first frame is emitted immediately.
    assert clock() - start == pytest.approx(9 * FRAME_INTERVAL)


async def test_idle_loop_cycles_rather_than_running_out(
    clock: FakeClock, telemetry: Telemetry
) -> None:
    mixer = make_mixer(clock, telemetry, make_idle_loop(count=3))

    frames = await take(mixer, 7)

    assert [f.data for f in frames] == [
        b"idle-0",
        b"idle-1",
        b"idle-2",
        b"idle-0",
        b"idle-1",
        b"idle-2",
        b"idle-0",
    ]
    assert all(f.epoch == IDLE_EPOCH for f in frames)


# -- source switching ------------------------------------------------------


def test_switching_source_leaves_no_gap_in_the_timeline(
    clock: FakeClock, telemetry: Telemetry
) -> None:
    mixer = make_mixer(clock, telemetry)
    for i in range(4):
        mixer.offer(rendered(i), TURN_EPOCH)

    before = [mixer.next_frame() for _ in range(3)]
    mixer.set_source(FrameSource.RENDERER)
    after = [mixer.next_frame() for _ in range(3)]

    timeline = before + after
    assert [f.pts_ms for f in timeline] == [i * FRAME_INTERVAL_MS for i in range(6)]
    assert [f.data for f in before] == [b"idle-0", b"idle-1", b"idle-2"]
    assert [f.data for f in after] == [b"render-0", b"render-1", b"render-2"]


def test_switching_back_to_idle_drains_the_pending_render_queue(
    clock: FakeClock, telemetry: Telemetry
) -> None:
    mixer = make_mixer(clock, telemetry)
    mixer.set_source(FrameSource.RENDERER)
    for i in range(3):
        mixer.offer(rendered(i), TURN_EPOCH)
    assert mixer.buffered() == 3

    mixer.set_source(FrameSource.IDLE_LOOP)

    # Those frames belong to a turn that is no longer being shown. Keeping them
    # would mean the avatar resumes a cancelled sentence on the next turn.
    assert mixer.buffered() == 0
    assert mixer.frames_discarded == 3
    assert mixer.next_frame().data == b"idle-0"


def test_setting_the_same_source_twice_is_a_no_op(
    clock: FakeClock, telemetry: Telemetry
) -> None:
    mixer = make_mixer(clock, telemetry)
    mixer.set_source(FrameSource.RENDERER)
    mixer.offer(rendered(0), TURN_EPOCH)

    mixer.set_source(FrameSource.RENDERER)

    assert mixer.buffered() == 1, "a redundant switch must not discard buffered work"


# -- starvation ------------------------------------------------------------


def test_starvation_repeats_the_last_rendered_frame_and_counts_it(
    clock: FakeClock, telemetry: Telemetry
) -> None:
    mixer = make_mixer(clock, telemetry)
    mixer.set_source(FrameSource.RENDERER)
    mixer.offer(rendered(0), TURN_EPOCH)
    mixer.offer(rendered(1), TURN_EPOCH)

    frames = [mixer.next_frame() for _ in range(5)]

    assert [f.data for f in frames] == [
        b"render-0",
        b"render-1",
        b"render-1",
        b"render-1",
        b"render-1",
    ]
    # This counter climbing is the signal that the GPU is behind real time -- the
    # one an fps average hides, because the average stays at 25 either way.
    assert mixer.frames_repeated == 3
    assert mixer.frames_emitted == 5


def test_renderer_selected_but_silent_falls_back_to_a_live_idle_frame(
    clock: FakeClock, telemetry: Telemetry
) -> None:
    mixer = make_mixer(clock, telemetry)
    mixer.set_source(FrameSource.RENDERER)

    frames = [mixer.next_frame() for _ in range(3)]

    # Nothing has been rendered yet, so there is nothing to repeat. A live idle
    # frame beats freezing, and it is not a repeat, so it must not be counted.
    assert [f.data for f in frames] == [b"idle-0", b"idle-1", b"idle-2"]
    assert mixer.frames_repeated == 0


def test_stale_frames_are_rejected_and_reported(clock: FakeClock, telemetry: Telemetry) -> None:
    mixer = make_mixer(clock, telemetry)

    accepted = mixer.offer(Frame(data=b"stale", epoch=TURN_EPOCH, pts_ms=0), TURN_EPOCH + 1)

    assert accepted is False
    assert mixer.buffered() == 0
    assert any(e["event"] == "stale_dropped" for e in telemetry.events)


# -- idle loop construction ------------------------------------------------


def test_idle_loop_rejects_an_empty_clip() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        IdleLoop([], [0])


def test_idle_loop_rejects_a_clip_with_no_clean_exit() -> None:
    # Without a mouth-closed frame the handover to the renderer can never be
    # seam-free, and the failure would show up as a visible pop in the demo
    # rather than as an error at startup.
    with pytest.raises(ValueError, match="mouth-closed"):
        IdleLoop([b"a", b"b"], [])


def test_clean_exit_tracks_the_current_idle_position() -> None:
    telemetry = Telemetry(sink=NullSink())
    clock = FakeClock()
    mixer = make_mixer(clock, telemetry, make_idle_loop(count=4, mouth_closed=[0, 2]))

    assert mixer.at_clean_exit() is True  # index 0
    mixer.next_frame()
    assert mixer.at_clean_exit() is False  # index 1
    mixer.next_frame()
    assert mixer.at_clean_exit() is True  # index 2
