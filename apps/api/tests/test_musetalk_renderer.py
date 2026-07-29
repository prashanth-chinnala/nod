"""
The MuseTalk renderer's streaming logic, tested without a GPU.

Everything here runs against a fake backend, and that is the point rather than a compromise.
The parts of a GPU renderer most likely to be wrong are not the matrix multiplications — they
are the glue: how a live PCM stream is cut into windows, what happens to a window that is
still filling when `frames()` is called, whether a barge-in mid-window leaves stale audio
behind, and whether the reference cycle continues across windows or snaps back to the start.
None of that needs a GPU, and none of it would be exercised by a single end-to-end render.

The one thing these tests cannot say anything about is whether the model produces a good
face. That needs the spike, and until it runs, every throughput figure stays
`NOT YET MEASURED`.
"""

from __future__ import annotations

import pytest

from avatar.contracts import IDLE_EPOCH, AudioChunk, RendererConfig, TalkingHeadRenderer
from avatar.renderers import build
from avatar.renderers.musetalk import (
    BYTES_PER_FRAME,
    MuseTalkBackend,
    MuseTalkIdentity,
    MuseTalkRenderer,
)


class FakeBackend:
    """
    Records every call and returns one distinguishable image per requested frame.

    Deliberately honest about `count`: it returns exactly what was asked for, so a test that
    expects a different number of frames is testing this module's arithmetic rather than the
    fake's generosity.
    """

    def __init__(self) -> None:
        self.loads = 0
        self.prepared: list[str] = []
        self.renders: list[dict[str, object]] = []
        self.unloads = 0

    def load(self) -> None:
        self.loads += 1

    def prepare(self, reference_path: str) -> object:
        self.prepared.append(reference_path)
        return {"ref": reference_path}

    def render(
        self, prepared: object, pcm: bytes, *, start_frame: int, count: int
    ) -> list[bytes]:
        self.renders.append({"pcm_len": len(pcm), "start_frame": start_frame, "count": count})
        return [f"frame-{start_frame + i}".encode() for i in range(count)]

    def unload(self) -> None:
        self.unloads += 1


def renderer(**kwargs: object) -> tuple[MuseTalkRenderer, FakeBackend]:
    backend = FakeBackend()
    return MuseTalkRenderer(backend=backend, **kwargs), backend  # type: ignore[arg-type]


def session_of(r: MuseTalkRenderer) -> object:
    return r.start_session(r.prepare_identity("persona.mp4"))


def audio(frames: int, *, epoch: int) -> AudioChunk:
    return AudioChunk(
        pcm=b"\x00" * (frames * BYTES_PER_FRAME),
        epoch=epoch,
        duration_ms=frames * 40,
    )


# -- the contract ----------------------------------------------------------


def test_it_satisfies_the_renderer_protocol() -> None:
    r, _ = renderer()
    assert isinstance(r, TalkingHeadRenderer)


def test_the_backend_seam_is_a_protocol_the_fake_satisfies() -> None:
    """If the fake drifts from the real backend's shape, these tests stop meaning anything."""
    assert isinstance(FakeBackend(), MuseTalkBackend)


def test_it_is_selected_by_config_alone() -> None:
    """
    The one-line swap the whole boundary argument rests on.

    Constructing it must not import torch or load weights — the backend is lazy — so this
    passes in GPU-free CI, which is what makes the claim checkable rather than asserted.
    """
    built = build(RendererConfig(name="musetalk"))
    assert isinstance(built, MuseTalkRenderer)


def test_an_unknown_renderer_still_names_what_is_available() -> None:
    with pytest.raises(ValueError, match="'stub', 'musetalk'"):
        build(RendererConfig(name="not-a-model"))


# -- identity is prepared once, not per session ----------------------------


def test_identity_preparation_happens_once_and_is_reusable() -> None:
    """
    §1.2's claim, made mechanical: identity is data, so one prepared persona serves many
    sessions without re-running the expensive step.
    """
    r, backend = renderer()
    identity = r.prepare_identity("persona.mp4")

    first, second = r.start_session(identity), r.start_session(identity)

    assert backend.prepared == ["persona.mp4"], "prepare must not run per session"
    assert first is not second


def test_starting_a_session_with_an_unprepared_identity_is_rejected() -> None:
    """A dict or a path here would fail much later, inside the backend, and confusingly."""
    r, _ = renderer()
    with pytest.raises(TypeError, match="MuseTalkIdentity"):
        r.start_session("persona.mp4")


def test_weights_load_once_per_process_not_per_session() -> None:
    r, backend = renderer()
    identity = r.prepare_identity("a.mp4")
    r.start_session(identity)
    r.start_session(identity)
    r.prepare_identity("b.mp4")

    assert backend.loads == 1, "loading per session is the cold-start cost §1.4 rejects"


# -- windowing -------------------------------------------------------------


def test_a_partial_window_yields_nothing_and_stays_buffered() -> None:
    """
    `frames()` must never block waiting for audio, and must not render a short window.

    Rendering early would emit frames conditioned on truncated audio — a visible artifact —
    and blocking would stall the mixer's cadence, which is worse than either.
    """
    r, backend = renderer(window_frames=8)
    s = session_of(r)

    r.push_audio(s, audio(5, epoch=1))

    assert list(r.frames(s)) == []
    assert backend.renders == [], "no render call at all, not an empty one"


def test_a_full_window_renders_and_the_remainder_carries_forward() -> None:
    r, backend = renderer(window_frames=8)
    s = session_of(r)

    r.push_audio(s, audio(11, epoch=1))
    first = list(r.frames(s))

    assert len(first) == 8
    assert len(backend.renders) == 1

    # The leftover 3 frames are still buffered; 5 more complete a second window.
    r.push_audio(s, audio(5, epoch=1))
    second = list(r.frames(s))

    assert len(second) == 8
    assert len(backend.renders) == 2


def test_several_buffered_windows_all_drain_in_one_call() -> None:
    """A slow consumer must not permanently lag a fast producer."""
    r, backend = renderer(window_frames=4)
    s = session_of(r)

    r.push_audio(s, audio(12, epoch=1))

    assert len(list(r.frames(s))) == 12
    assert len(backend.renders) == 3


# -- the reference cycle advances ------------------------------------------


def test_the_reference_cycle_continues_across_windows() -> None:
    """
    Each window must start where the last one ended.

    Restarting at zero would snap the head back to the first reference frame at every
    window boundary — periodic, at the window rate, and unmistakable on screen.
    """
    r, backend = renderer(window_frames=4)
    s = session_of(r)

    r.push_audio(s, audio(12, epoch=1))
    list(r.frames(s))

    assert [call["start_frame"] for call in backend.renders] == [0, 4, 8]


def test_presentation_timestamps_advance_monotonically() -> None:
    r, _ = renderer(window_frames=4)
    s = session_of(r)

    r.push_audio(s, audio(8, epoch=1))
    stamps = [f.pts_ms for f in r.frames(s)]

    assert stamps == [i * 40 for i in range(8)]


def test_every_frame_carries_the_epoch_that_produced_it() -> None:
    """The tag the consumer uses to drop artifacts from an abandoned turn."""
    r, _ = renderer(window_frames=4)
    s = session_of(r)

    r.push_audio(s, audio(4, epoch=7))

    assert all(f.epoch == 7 for f in r.frames(s))


# -- audio context across window boundaries --------------------------------


def test_the_second_window_is_given_overlapping_context() -> None:
    """
    Without overlap, the mouth jumps at every window boundary.

    Whisper needs audio either side of a frame to encode it well, so the first window is
    exactly its own length and every later one is longer by the context.
    """
    r, backend = renderer(window_frames=8, context_frames=2)
    s = session_of(r)

    r.push_audio(s, audio(16, epoch=1))
    list(r.frames(s))

    assert backend.renders[0]["pcm_len"] == 8 * BYTES_PER_FRAME
    assert backend.renders[1]["pcm_len"] == 10 * BYTES_PER_FRAME


def test_context_can_be_switched_off() -> None:
    """Useful for isolating whether a visible seam is the context or something else."""
    r, backend = renderer(window_frames=8, context_frames=0)
    s = session_of(r)

    r.push_audio(s, audio(16, epoch=1))
    list(r.frames(s))

    assert {call["pcm_len"] for call in backend.renders} == {8 * BYTES_PER_FRAME}


# -- barge-in --------------------------------------------------------------


def test_reset_drops_buffered_audio_and_context() -> None:
    r, backend = renderer(window_frames=8)
    s = session_of(r)
    r.push_audio(s, audio(7, epoch=1))  # nearly a window, deliberately

    r.reset(s)
    r.push_audio(s, audio(7, epoch=2))

    assert list(r.frames(s)) == [], "the abandoned turn's audio must not complete a window"
    assert backend.renders == []


def test_reset_is_safe_with_nothing_in_flight() -> None:
    r, _ = renderer()
    s = session_of(r)

    r.reset(s)
    r.reset(s)  # twice, because a barge-in can race a natural turn end


def test_reset_does_not_unload_weights() -> None:
    """Reloading per barge-in would make interruption cost seconds instead of microseconds."""
    r, backend = renderer()
    s = session_of(r)

    r.reset(s)

    assert backend.unloads == 0
    assert backend.loads == 1


def test_a_new_epoch_discards_the_previous_turn_s_buffer() -> None:
    """
    Audio from two different turns must never be rendered as one continuous window.

    Otherwise the join between an abandoned utterance and its replacement is rendered as
    speech, which looks like the avatar saying something nobody asked for.
    """
    r, backend = renderer(window_frames=8)
    s = session_of(r)

    r.push_audio(s, audio(6, epoch=1))
    r.push_audio(s, audio(8, epoch=2))
    frames = list(r.frames(s))

    assert len(frames) == 8
    assert all(f.epoch == 2 for f in frames)
    assert backend.renders[0]["pcm_len"] == 8 * BYTES_PER_FRAME, (
        "the 6 frames from epoch 1 must be dropped, not prepended"
    )


def test_frame_numbering_restarts_with_a_new_turn() -> None:
    r, _ = renderer(window_frames=4)
    s = session_of(r)

    r.push_audio(s, audio(4, epoch=1))
    list(r.frames(s))
    r.push_audio(s, audio(4, epoch=2))
    second = list(r.frames(s))

    assert [f.pts_ms for f in second] == [0, 40, 80, 120]


# -- session lifecycle -----------------------------------------------------


def test_a_closed_session_is_rejected_rather_than_silently_ignored() -> None:
    r, _ = renderer()
    s = session_of(r)

    r.close_session(s)

    with pytest.raises(RuntimeError, match="closed"):
        r.push_audio(s, audio(1, epoch=1))


def test_closing_a_session_does_not_unload_the_model() -> None:
    """Weights are process-scoped; the next session must not pay to load them again."""
    r, backend = renderer()
    s = session_of(r)

    r.close_session(s)

    assert backend.unloads == 0


def test_a_fresh_session_starts_at_the_idle_epoch() -> None:
    r, _ = renderer()
    s = session_of(r)

    assert s.epoch == IDLE_EPOCH  # type: ignore[attr-defined]


# -- construction guards --------------------------------------------------


@pytest.mark.parametrize("window", [0, -1])
def test_a_nonsensical_window_is_rejected_at_construction(window: int) -> None:
    """Zero would loop forever in `frames`; negative would render backwards."""
    with pytest.raises(ValueError, match="window_frames"):
        MuseTalkRenderer(backend=FakeBackend(), window_frames=window)


def test_negative_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="context_frames"):
        MuseTalkRenderer(backend=FakeBackend(), context_frames=-1)


def test_the_identity_records_where_it_came_from() -> None:
    """Cheap, and it is what makes a wrong-persona bug identifiable from a log line."""
    r, _ = renderer()
    identity = r.prepare_identity("faces/interviewer.mp4")

    assert isinstance(identity, MuseTalkIdentity)
    assert identity.reference_path == "faces/interviewer.mp4"
