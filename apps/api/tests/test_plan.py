"""
Tests for the competency plan.

Each test names the failure it prevents rather than the method it calls. Coverage bookkeeping is
the kind of logic that is wrong quietly: an interview that re-asks a covered area, or reports
coverage it never had, still looks like a working interview from the outside. These are the
assertions that would notice.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence

import pytest

from avatar.contracts import Message
from avatar.plan import (
    Competency,
    Coverage,
    InterviewPlan,
    PlanState,
    brief,
    slug,
    with_plan,
)


def competency(name: str, *signals: str, **kwargs: object) -> Competency:
    return Competency(id=slug(name), name=name, signals=signals, **kwargs)  # type: ignore[arg-type]


def plan(*competencies: Competency) -> InterviewPlan:
    return InterviewPlan(name="Backend", competencies=competencies)


def history(*turns: str) -> list[Message]:
    """Alternating candidate/interviewer turns, candidate last — the shape a turn arrives in."""
    return [{"role": "user", "content": turn} for turn in turns]


class Recorder:
    """A `SentenceStream` that keeps the history it was handed, so injection can be asserted."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def __call__(self, messages: Sequence[Message]) -> AsyncGenerator[str, None]:
        self.calls.append(list(messages))

        async def stream() -> AsyncGenerator[str, None]:
            yield "And how did you find that out?"

        return stream()

    @property
    def last_guidance(self) -> str:
        """The plan's brief from the most recent call, which is always the final message."""
        return str(self.calls[-1][-1]["content"])


# -- ids ------------------------------------------------------------------


def test_slug_is_stable_across_reordering_not_position() -> None:
    """
    Ids come from the name, so moving a competency up the list cannot reassign coverage.

    The positional alternative is the bug this prevents: an operator reorders a rubric in the
    console and every past session's coverage silently starts pointing at a different area.
    """
    assert slug("Debugging under pressure") == "debugging-under-pressure"
    assert slug("  Incident Response!  ") == "incident-response"


def test_slug_never_returns_empty() -> None:
    """A name of only punctuation must still key a coverage record, not an empty string."""
    assert slug("!!!") == "competency"


# -- status ---------------------------------------------------------------


def test_unasked_until_probed() -> None:
    target = competency("Debugging", "profiler")
    assert Coverage(target.id).status(target) == "unasked"


def test_evidence_beats_exhaustion() -> None:
    """
    A competency probed to its limit that also produced signals is evidenced, not exhausted.

    Order matters here: with the checks reversed, the area asked about most in a long interview
    would be reported as a dead end precisely because it was asked about most.
    """
    target = competency("Debugging", "profiler", max_turns=2)
    state = Coverage(target.id, asked=5, signals_hit=["profiler"])
    assert state.status(target) == "evidenced"


def test_exhausted_is_not_evidenced() -> None:
    """
    Probing without signals closes the competency but must not claim coverage.

    This is the distinction a report depends on. "We asked twice and learned nothing" is a
    finding; collapsing it into "covered" would state the opposite of what happened.
    """
    target = competency("Debugging", "profiler", max_turns=2)
    state = Coverage(target.id, asked=2)
    assert state.status(target) == "exhausted"
    assert state.signals_hit == []  # closed, but with nothing to show for it


def test_min_signals_holds_the_bar() -> None:
    target = competency("Debugging", "profiler", "flame graph", min_signals=2)
    state = Coverage(target.id, asked=1, signals_hit=["profiler"])
    assert state.status(target) == "probing"
    state.signals_hit.append("flame graph")
    assert state.status(target) == "evidenced"


# -- observation ----------------------------------------------------------


def test_signals_credit_every_competency_not_just_the_focus() -> None:
    """
    An answer about one area may evidence another, and that evidence must count.

    Crediting only the competency being probed would throw it away and then spend a turn asking
    about something the candidate had already demonstrated.
    """
    state = PlanState(
        plan(competency("Deployment", "canary"), competency("Incidents", "pager", "postmortem"))
    )
    gained = state.observe("We rolled it out as a canary after the pager went off at 3am.")
    assert gained == ["deployment", "incidents"]


def test_repeated_signals_do_not_accumulate() -> None:
    """
    Saying the same word four times is one piece of evidence.

    Otherwise a single enthusiastic sentence closes a competency whose `min_signals` was raised
    specifically to require breadth.
    """
    state = PlanState(plan(competency("Scale", "sharding", "partition", min_signals=2)))
    state.observe("sharding, sharding, and more sharding")
    assert state.coverage["scale"].signals_hit == ["sharding"]
    # Still short of the bar: one distinct signal against min_signals=2. Nothing has been asked
    # yet, so the status is "unasked" rather than "probing" -- the point is only that repetition
    # did not close it.
    assert state.coverage["scale"].status(state.plan.competencies[0]) != "evidenced"


def test_signals_match_terms_a_word_boundary_regex_would_miss() -> None:
    """
    `C++` and `.NET` are the signals an engineering rubric is actually written in.

    A naive `\\b` pattern never matches either, so this would present as a competency that stays
    unevidenced no matter what the candidate says.
    """
    state = PlanState(plan(competency("Languages", "C++", ".NET", "Node.js")))
    gained = state.observe("Mostly C++ and .NET, some Node.js on the tooling side.")
    assert gained == ["languages"]
    assert state.coverage["languages"].signals_hit == ["C++", ".NET", "Node.js"]


def test_substring_does_not_count_as_a_signal() -> None:
    state = PlanState(plan(competency("Messaging", "Kafka")))
    assert state.observe("It felt Kafkaesque, honestly.") == []


def test_empty_answer_credits_nothing() -> None:
    """An untranscribed turn must not silently evidence anything."""
    state = PlanState(plan(competency("Scale", "sharding")))
    assert state.observe("   ") == []


def test_evidence_is_kept_verbatim_and_truncated() -> None:
    """
    A reviewer needs the sentence the machine matched on, not just its verdict.

    Truncated because the full transcript is stored per turn already; a second copy per
    competency would make the session record mostly duplication.
    """
    state = PlanState(plan(competency("Scale", "sharding")))
    state.observe("x" * 500 + " sharding")
    stored = state.coverage["scale"].evidence[0]
    assert len(stored) == PlanState.EVIDENCE_CHARS
    assert stored.startswith("x")


# -- focus ----------------------------------------------------------------


def test_focus_stays_put_until_closed() -> None:
    """
    An interview that rotates topics every turn never gets past the summary answer.

    Staying on one competency until it is evidenced or exhausted is what makes a follow-up
    question possible at all.
    """
    first, second = competency("Deployment", "canary"), competency("Incidents", "pager")
    state = PlanState(plan(first, second))
    assert state.focus() == first
    state.mark_asked(first)
    assert state.focus() == first


def test_focus_advances_once_evidenced() -> None:
    first, second = competency("Deployment", "canary"), competency("Incidents", "pager")
    state = PlanState(plan(first, second))
    state.observe("we ship behind a canary")
    assert state.focus() == second


def test_focus_advances_on_exhaustion_so_the_interview_can_move_on() -> None:
    first = competency("Deployment", "canary", max_turns=2)
    second = competency("Incidents", "pager")
    state = PlanState(plan(first, second))
    state.mark_asked(first)
    state.mark_asked(first)
    assert state.focus() == second


def test_focus_is_none_when_everything_is_closed() -> None:
    state = PlanState(plan(competency("Deployment", "canary")))
    state.observe("canary")
    assert state.focus() is None
    assert state.snapshot()["complete"] is True


# -- the brief ------------------------------------------------------------


def test_brief_names_the_focus_and_forbids_announcing_it() -> None:
    """
    The model must probe the area without saying "now let us discuss X".

    Without the instruction it announces the competency, which turns a conversation into a form
    being filled in.
    """
    state = PlanState(plan(competency("Debugging", "profiler")))
    text = brief(state)
    assert "Probe next: Debugging" in text
    assert "not reading" in text


def test_brief_lists_evidenced_areas_so_they_are_not_re_asked() -> None:
    """
    The model cannot see coverage — it sees history, which contains the answer but not the
    fact that the answer counted. Naming what is done is what stops a repeat question.
    """
    state = PlanState(
        plan(competency("Deployment", "canary"), competency("Incidents", "pager"))
    )
    state.observe("we ship behind a canary")
    text = brief(state)
    assert "Already evidenced, do not re-ask: Deployment." in text


def test_brief_closes_the_interview_when_the_plan_is_complete() -> None:
    """
    A finished plan still needs an ending. Left silent, the model has a plan-shaped hole in its
    context and drifts into small talk.
    """
    state = PlanState(plan(competency("Deployment", "canary")))
    state.observe("canary")
    text = brief(state)
    assert "closing the interview" in text
    assert "Do not start a new topic" in text


# -- the decorator --------------------------------------------------------


@pytest.mark.asyncio
async def test_with_plan_appends_guidance_after_the_history() -> None:
    """
    Appended, not prepended: a system message at the front competes with the interviewer's own
    instructions, while one at the end reads as the brief for this turn.
    """
    llm = Recorder()
    wrapped = with_plan(llm, plan(competency("Debugging", "profiler")))
    # Deliberately an answer containing none of the signals: one that evidenced the competency
    # would complete the plan and produce the closing brief instead of a focus.
    async for _ in wrapped(history("I mostly read the logs")):
        pass
    assert llm.calls[-1][0]["content"] == "I mostly read the logs"
    assert llm.calls[-1][-1]["role"] == "system"
    assert "Probe next" in llm.last_guidance


@pytest.mark.asyncio
async def test_with_plan_observes_before_choosing_a_focus() -> None:
    """
    The ordering bug this pins down: pick a focus first and the competency the candidate just
    demonstrated is still selected, so the very next question asks about it again.
    """
    llm = Recorder()
    wrapped = with_plan(
        llm, plan(competency("Deployment", "canary"), competency("Incidents", "pager"))
    )
    async for _ in wrapped(history("we ship behind a canary")):
        pass
    assert "Probe next: Incidents" in llm.last_guidance
    assert "Deployment" in llm.last_guidance  # named as already evidenced


@pytest.mark.asyncio
async def test_with_plan_is_a_no_op_without_competencies() -> None:
    """
    An agent with no rubric must add no call at all, so the clean-clone path keeps working.

    Identity rather than behavioural equality: a wrapper that merely forwards would still put a
    frame on every turn's stack and a branch in every trace.
    """
    llm = Recorder()
    assert with_plan(llm, InterviewPlan()) is llm


@pytest.mark.asyncio
async def test_with_plan_counts_a_probe_even_if_the_turn_is_abandoned() -> None:
    """
    `asked` increments when the question is generated, not when it is answered.

    A barge-in during THINKING still consumed one of the competency's attempts; counting only
    completed turns would let a candidate who interrupts every question keep the interview on
    its first competency forever.
    """
    llm = Recorder()
    interview = plan(competency("Debugging", "profiler", max_turns=1))
    seen: list[dict[str, object]] = []
    wrapped = with_plan(llm, interview, on_update=seen.append)
    stream = wrapped(history("hello"))
    await stream.asend(None)  # first sentence only, then walk away
    await stream.aclose()
    assert seen[-1]["competencies"][0]["asked"] == 1  # type: ignore[index]
    assert seen[-1]["complete"] is True  # max_turns=1, so it is closed by exhaustion


@pytest.mark.asyncio
async def test_snapshot_survives_a_json_round_trip() -> None:
    """
    The snapshot is written to the session record and read back by the console, so it must be
    plain data — a dataclass leaking in would serialise one way and fail to come back.
    """
    import json

    state = PlanState(plan(competency("Scale", "sharding")))
    state.observe("we did sharding")
    assert json.loads(json.dumps(state.snapshot())) == state.snapshot()
