"""
The transition table, and the guards that keep the session out of dead ends.

The table-completeness tests at the top are the cheap ones that pay off later: adding a state
without deciding its legal transitions or its frame source fails here rather than producing a
session that reaches a state the mixer has no opinion about.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from avatar.orchestrator import SessionOrchestrator
from avatar.state import FRAME_SOURCE, LEGAL_TRANSITIONS, InvalidTransition, State
from tests.conftest import FakeClock, RecordingTransport, ScriptedLLM, run_until, settle

# -- the tables ------------------------------------------------------------


def test_every_state_has_transitions_and_a_frame_source() -> None:
    for state in State:
        assert state in LEGAL_TRANSITIONS, f"{state} has no transition entry"
        assert state in FRAME_SOURCE, f"{state} has no frame source"


def test_no_state_transitions_back_to_initializing() -> None:
    for source, targets in LEGAL_TRANSITIONS.items():
        assert State.INITIALIZING not in targets, f"{source} can re-initialise"


def test_closed_is_terminal() -> None:
    assert LEGAL_TRANSITIONS[State.CLOSED] == frozenset()


def test_every_state_can_reach_closed() -> None:
    """A session must always be closable, including from a failure state."""
    for state in State:
        if state is State.CLOSED:
            continue
        assert State.CLOSED in LEGAL_TRANSITIONS[state], f"{state} cannot be closed"


def test_transition_targets_are_all_known_states() -> None:
    for targets in LEGAL_TRANSITIONS.values():
        for target in targets:
            assert isinstance(target, State)


# -- lifecycle -------------------------------------------------------------


async def test_start_opens_transport_and_lands_in_idle(
    build_session: Callable[..., SessionOrchestrator], transport: RecordingTransport
) -> None:
    orch = build_session()
    assert orch.state is State.INITIALIZING

    await orch.start("reference.mp4")

    assert orch.state is State.IDLE
    assert transport.opened is True


async def test_close_releases_renderer_and_transport(
    build_session: Callable[..., SessionOrchestrator], transport: RecordingTransport
) -> None:
    orch = build_session()
    await orch.start("reference.mp4")
    await orch.close()

    assert orch.state is State.CLOSED
    assert transport.closed is True


async def test_close_is_idempotent(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    orch = build_session()
    await orch.start("reference.mp4")
    await orch.close()
    await orch.close()  # must not raise InvalidTransition on CLOSED -> CLOSED

    assert orch.state is State.CLOSED


async def test_illegal_transition_raises(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    orch = build_session()
    await orch.start("reference.mp4")
    with pytest.raises(InvalidTransition):
        # IDLE -> SPEAKING skips the turn entirely; there is no audio to speak.
        orch._transition(State.SPEAKING)


# -- guards ----------------------------------------------------------------


async def test_end_of_turn_while_idle_is_a_no_op(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    orch = build_session()
    await orch.start("reference.mp4")

    await orch.on_end_of_turn("did anyone ask?")

    assert orch.state is State.IDLE
    assert orch.history == []
    assert orch.pipeline_task is None


async def test_speech_retraction_returns_to_idle_without_starting_a_turn(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    orch = build_session()
    await orch.start("reference.mp4")

    await orch.on_speech_start()
    assert orch.state is State.LISTENING

    await orch.on_speech_retract()

    assert orch.state is State.IDLE
    assert orch.epoch == 0, "a retracted false positive must not consume a turn epoch"
    assert orch.pipeline_task is None


async def test_speech_retraction_outside_listening_is_ignored(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    orch = build_session()
    await orch.start("reference.mp4")

    await orch.on_speech_retract()

    assert orch.state is State.IDLE


async def test_full_turn_returns_to_idle_and_records_history(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    orch = build_session(llm=ScriptedLLM(["The billing migration.", " Second year."]))
    await orch.start("reference.mp4")

    await orch.on_speech_start()
    await orch.on_end_of_turn("Tell me about a failure.")
    await settle(orch)

    assert orch.state is State.IDLE
    assert orch.history == [
        {"role": "user", "content": "Tell me about a failure."},
        {"role": "assistant", "content": "The billing migration. Second year."},
    ]


# -- silence watchdog ------------------------------------------------------


async def test_idle_reprompt_fires_after_the_timeout(
    build_session: Callable[..., SessionOrchestrator], clock: FakeClock
) -> None:
    orch = build_session(idle_reprompt_seconds=12.0)
    await orch.start("reference.mp4")

    clock.advance(13.0)
    await orch.on_idle_tick()

    # Straight from IDLE to THINKING: a re-prompt never passes through LISTENING,
    # because nobody spoke. Chaining this through on_end_of_turn would hit that
    # method's LISTENING guard and the re-prompt could never fire at all.
    assert orch.state is State.THINKING
    assert orch.history[0]["role"] == "user"
    await settle(orch)


async def test_the_reprompt_announces_itself_so_the_server_can_record_it(
    build_session: Callable[..., SessionOrchestrator], clock: FakeClock
) -> None:
    """
    The re-prompt must emit `heard`, because that event is what opens a turn record.

    The server builds turns from telemetry rather than from explicit writes, so a generation
    with no `heard` in front of it is spoken to the candidate and stored nowhere -- which is
    exactly what happened to every silence re-prompt until this assertion existed. Empty text
    with `silent=True`: nothing was said, and the marker that goes into conversation history
    must not reach the transcript the scorer quotes from.
    """
    events: list[dict[str, object]] = []
    orch = build_session(idle_reprompt_seconds=12.0)
    orch._telemetry.subscribe(events.append)
    await orch.start("reference.mp4")

    clock.advance(13.0)
    await orch.on_idle_tick()

    announcements = [e for e in events if e.get("event") == "heard"]
    assert len(announcements) == 1, events
    assert announcements[0]["silent"] is True
    assert announcements[0]["text"] == ""
    assert announcements[0]["transcribed"] is False
    await settle(orch)


async def test_idle_reprompt_does_not_fire_before_the_timeout(
    build_session: Callable[..., SessionOrchestrator], clock: FakeClock
) -> None:
    orch = build_session(idle_reprompt_seconds=12.0)
    await orch.start("reference.mp4")

    clock.advance(11.0)
    await orch.on_idle_tick()

    assert orch.state is State.IDLE
    assert orch.history == []


async def test_idle_reprompt_only_fires_from_idle(
    build_session: Callable[..., SessionOrchestrator], clock: FakeClock
) -> None:
    orch = build_session(idle_reprompt_seconds=12.0)
    await orch.start("reference.mp4")
    await orch.on_speech_start()
    assert orch.state is State.LISTENING

    clock.advance(30.0)
    await orch.on_idle_tick()

    # The candidate is mid-sentence. Re-prompting them would be the avatar talking
    # over the person it is interviewing.
    assert orch.state is State.LISTENING
    assert orch.history == []


async def test_speech_resets_the_silence_timer(
    build_session: Callable[..., SessionOrchestrator], clock: FakeClock
) -> None:
    orch = build_session(idle_reprompt_seconds=12.0)
    await orch.start("reference.mp4")

    clock.advance(11.0)
    await orch.on_speech_start()
    await orch.on_speech_retract()
    clock.advance(6.0)  # 17s since session start, but only 6s since activity
    await orch.on_idle_tick()

    assert orch.state is State.IDLE
    assert orch.history == []


# -- failure recovery ------------------------------------------------------


async def test_exception_mid_turn_returns_to_idle_rather_than_wedging(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    orch = build_session(llm=ScriptedLLM(["fine"], raise_at_index=0))
    await orch.start("reference.mp4")

    await orch.on_speech_start()
    await orch.on_end_of_turn("Tell me about a failure.")

    task = orch.pipeline_task
    assert task is not None
    with pytest.raises(RuntimeError, match="llm exploded"):
        await task

    # Wedged in THINKING would mean the session never speaks or listens again, and
    # the only visible symptom is an avatar that has quietly stopped interviewing.
    assert orch.state is State.IDLE


async def test_failed_turn_is_reported_as_telemetry(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    orch = build_session(llm=ScriptedLLM(["fine"], raise_at_index=0))
    await orch.start("reference.mp4")
    await orch.on_speech_start()
    await orch.on_end_of_turn("Tell me about a failure.")

    task = orch.pipeline_task
    assert task is not None
    with pytest.raises(RuntimeError):
        await task

    failures = [e for e in orch._telemetry.events if e["event"] == "session_failure"]
    assert failures and failures[0]["cause"] == "RuntimeError"


async def test_session_recovers_enough_to_take_another_turn(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    """A failed turn must not poison the session."""
    llm = ScriptedLLM(["recovered."], raise_at_index=None)
    failing = ScriptedLLM(["boom"], raise_at_index=0)
    orch = build_session(llm=failing)
    await orch.start("reference.mp4")

    await orch.on_speech_start()
    await orch.on_end_of_turn("first")
    task = orch.pipeline_task
    assert task is not None
    with pytest.raises(RuntimeError):
        await task

    orch._llm = llm  # swap in a working LLM; the state machine should not care
    await orch.on_speech_start()
    await orch.on_end_of_turn("second")
    await settle(orch)

    assert orch.state is State.IDLE
    assert orch.history[-1] == {"role": "assistant", "content": "recovered."}


async def test_end_of_turn_from_idle_is_refused_and_says_so(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    """
    A turn the machine will not accept must report that, not just quietly not happen.

    The silent version caused a real defect one level up: `server.py` emitted `heard` telemetry
    before calling this, so an end-of-turn arriving without a preceding `speech_start` put the
    candidate's words in the transcript pane and in the session record while the model never
    received them. The next question then read as the interviewer ignoring the answer, which is
    the precise failure `heard` exists to expose — caused by the instrumentation itself.

    Asserting the history too, because the return value alone would let an implementation say
    "refused" while still appending, and the appended-but-refused case is the one that would
    reach the model on the *following* turn out of order.
    """
    orch = build_session()
    await orch.start("reference.mp4")
    assert orch.state is State.IDLE

    accepted = await orch.on_end_of_turn("I have six years of backend experience.")

    assert accepted is False
    assert orch.history == []


async def test_end_of_turn_from_listening_is_accepted(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    """The accepted path, so the assertion above is about the guard and not about the return."""
    orch = build_session()
    await orch.start("reference.mp4")
    await orch.on_speech_start()

    accepted = await orch.on_end_of_turn("I have six years of backend experience.")

    assert accepted is True
    assert orch.history[-1]["content"] == "I have six years of backend experience."


async def test_cancelling_is_entered_before_the_epoch_bump(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    """
    The ordering that made every barge-in look like a clean turn.

    `_cancel_turn` transitions to CANCELLING and only then bumps the epoch, so a consumer that
    waits for a *stale artifact* to learn about an interruption is always one step behind: the
    transition has already been broadcast, and anything that flushes on it has already written
    the record. `server.py` used to record `interrupted` from `stale_dropped` for exactly that
    reason and never once succeeded — every mid-speech interruption persisted as `interrupted:
    false`, and the sessions API reported zero barge-ins for every interview ever held.

    Pinned here rather than in the server because the ordering is the orchestrator's, and a
    future change that bumped the epoch first would silently make the server's fix unnecessary —
    or, if reversed again, silently reintroduce the bug.
    """
    orch = build_session()
    await orch.start("reference.mp4")
    await orch.on_speech_start()
    await orch.on_end_of_turn("tell me about a failure")
    await run_until(lambda: orch.state is State.SPEAKING, what="SPEAKING")

    seen: list[tuple[str, int]] = []
    original = orch._transition

    def record(state: State) -> None:
        seen.append((str(state), orch.epoch))
        original(state)

    orch._transition = record  # type: ignore[method-assign]
    await orch.on_speech_start()

    cancelling = [epoch for name, epoch in seen if name == "CANCELLING"]
    assert cancelling, "a barge-in during SPEAKING must enter CANCELLING"
    # The epoch at the moment of the transition is still the *old* one: the bump follows.
    assert cancelling[0] < orch.epoch
