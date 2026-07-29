"""
The competency plan: what the interviewer is trying to find out, and how far it has got.

**What this changes about the interview.** Until now the agent had knowledge, a voice, tools and
a policy, but no *agenda*. It answered whatever the candidate raised and had no way to know it
had spent eight turns on deployment and never asked about debugging. A plan gives the
conversation a shape: a list of competencies in priority order, a record of which ones the
candidate has actually produced evidence for, and a focus injected into each turn. That is the
difference between a chatbot that happens to ask questions and something an interview can be run
on.

**Coverage is decided by signal matching, not by asking a model.** This is the central trade-off
here, and it goes the way it does because of where the work sits. Coverage has to be known
*before* the next question is generated, so any judgement is on the critical path. A turn
already measures 2.7-5.8s against a sub-second target and LLM time-to-first-token alone measured
1,645-4,724ms; a judging round trip would roughly double the worst case, on every turn, to
decide something the next question only needs approximately. Signal matching is a string scan
over a handful of terms.

The cost of that choice is real and worth naming: matching "sharding" is a far coarser
instrument than a model reading the answer, and a candidate who describes partitioning a
dataset without ever using the operator's vocabulary will not register. So this steers; it
does not score. Scoring is `avatar.scoring`'s job, it runs after the session where latency is
free, and it is allowed the model this deliberately is not.

**Signals are matched against every competency, not just the one being probed.** A candidate
answering a question about deployment often reveals exactly the incident-response experience
that was queued three competencies later. Crediting only the competency that was asked would
throw that evidence away and then spend a turn asking about something they had already
demonstrated, which is the single most irritating thing an interviewer can do.

**Evidenced and exhausted are different states, and the distinction is not cosmetic.** A
competency closes either because the candidate produced signals for it or because it was probed
`max_turns` times and the interview had to move on. Collapsing both into "covered" would let a
report claim coverage of an area where the honest finding is that we asked twice and learned
nothing — which is itself a result, and one a hiring manager needs to be able to see.

Nothing here imports a renderer, torch, or a web framework: this is prompt-shaping and
bookkeeping, so it stays outside the boundary `tests/test_boundaries.py` enforces.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from avatar.contracts import Message
from avatar.knowledge.augment import (
    SentenceStreamLike,
    latest_candidate_text,
    terms_present,
)

Status = Literal["unasked", "probing", "evidenced", "exhausted"]

DEFAULT_MAX_TURNS = 3
"""
How many times one competency may be the focus before the interview moves on.

Three rather than one because a first answer is often a summary and the follow-up is where the
substance is — the whole reason a plan beats a fixed question list. Three rather than ten
because a competency that has produced no signal in three attempts is not going to, and an
interview that cannot leave its first topic has failed differently from one that rushed.
"""

DEFAULT_MIN_SIGNALS = 1
"""
Distinct signals needed before a competency counts as evidenced.

One, because this gates *what to ask next*, not what to conclude. Requiring two would keep
probing an area the candidate has clearly engaged with, spending the interview's scarcest
resource — turns — to raise the confidence of a decision that is not being made here. The
scorer, which is making that decision, sets its own bar.
"""


def slug(name: str) -> str:
    """
    A stable, readable id for a competency, derived from its name.

    Derived rather than positional because coverage is keyed by it and an operator reordering
    the list in the console would otherwise silently reassign every session's history. Readable
    rather than random because these ids appear in a report a human reads.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return cleaned or "competency"


@dataclass(frozen=True)
class Competency:
    """
    One thing the interview is trying to find out.

    Frozen: the plan is configuration loaded once at session start, and the mutable part is
    `Coverage`. Keeping them separate types means no code path can accidentally record progress
    onto the definition and write it back to the operator's rubric.
    """

    id: str
    name: str
    probe: str = ""
    """
    What to explore, in the operator's words — not a question to read out.

    A stored question would make this a script, and a script cannot follow up, which is the
    entire reason a live interviewer is worth more than a form. The prompt below says "probe",
    and the model writes the sentence.
    """
    signals: tuple[str, ...] = ()
    max_turns: int = DEFAULT_MAX_TURNS
    min_signals: int = DEFAULT_MIN_SIGNALS
    weight: float = 1.0
    """
    How much this competency contributes to the scorecard. Read by `avatar.scoring`, ignored
    here.

    Deliberately separate from the declared order, which is what *this* module uses. Order
    answers "what next" and a weight answers "how much did it matter" -- two different
    questions, and one number doing both jobs would mean an operator could not say "ask about
    communication first, but weight it least", which is a perfectly ordinary thing to want.
    """


@dataclass
class Coverage:
    """Progress against one competency, for one session."""

    competency_id: str
    asked: int = 0
    signals_hit: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    """
    The answers that produced signals, verbatim and truncated.

    Stored because "evidenced" on its own is an assertion. A reviewer disagreeing with the
    machine needs to see the sentence it matched on, and a coverage record without quotes is
    something to take on trust.
    """

    def status(self, competency: Competency) -> Status:
        """
        Evidence first, exhaustion second, and that order is load-bearing.

        A competency probed its full allowance that *also* produced signals is evidenced, not
        exhausted — otherwise the last competency asked in a long interview would be reported as
        a dead end purely because it was asked often.
        """
        if len(self.signals_hit) >= competency.min_signals:
            return "evidenced"
        if self.asked >= competency.max_turns:
            return "exhausted"
        return "probing" if self.asked else "unasked"

    def closed(self, competency: Competency) -> bool:
        return self.status(competency) in ("evidenced", "exhausted")


@dataclass(frozen=True)
class InterviewPlan:
    """
    An ordered list of competencies. Declared order is priority order.

    Order rather than a numeric weight because the only decision made here is "what next", and
    for that a total order is both sufficient and unambiguous. Weights are for the scorer, where
    they express how much a competency contributes to a judgement — a different question that
    deserves a different field rather than one number doing two jobs badly.
    """

    name: str = ""
    competencies: tuple[Competency, ...] = ()

    @property
    def active(self) -> bool:
        return bool(self.competencies)


class PlanState:
    """
    Coverage for one session. Mutable, single-threaded, and owned by the decorator's closure.

    One instance per socket, built where the session is built. Nothing shares it, so no locking:
    the orchestrator awaits one turn at a time by construction, and a plan that could be updated
    from two turns at once would have a worse problem than a race — it would mean two turns were
    in flight, which the state machine exists to prevent.
    """

    EVIDENCE_CHARS = 240
    """
    How much of an answer is kept per piece of evidence.

    Enough for a reviewer to judge the match in context; short enough that a session record does
    not become a second copy of the transcript, which is stored per turn already.
    """

    def __init__(self, plan: InterviewPlan) -> None:
        self.plan = plan
        self.coverage: dict[str, Coverage] = {
            competency.id: Coverage(competency.id) for competency in plan.competencies
        }
        self._focus: str | None = None

    # -- reading ------------------------------------------------------------

    def focus(self) -> Competency | None:
        """
        The next competency to probe: the first still open, in declared order.

        First-open rather than round-robin, because an interview that rotates topics every turn
        never gets past the summary answer. Staying on one competency until it is evidenced or
        exhausted is what makes the follow-up possible.
        """
        for competency in self.plan.competencies:
            if not self.coverage[competency.id].closed(competency):
                return competency
        return None

    def snapshot(self) -> dict[str, Any]:
        """
        Coverage as plain data, for telemetry, the session record and the report.

        Plain dicts rather than the dataclasses so the same structure survives a JSON round trip
        to disk and back into the console without a serialiser in the middle that could disagree
        with itself in each direction.
        """
        focus = self.focus()
        items = []
        for competency in self.plan.competencies:
            state = self.coverage[competency.id]
            items.append(
                {
                    "id": competency.id,
                    "name": competency.name,
                    "status": state.status(competency),
                    "asked": state.asked,
                    "signals_hit": list(state.signals_hit),
                    "signals_total": len(competency.signals),
                    "evidence": list(state.evidence),
                }
            )
        return {
            "plan": self.plan.name,
            "focus": focus.id if focus else None,
            "complete": focus is None,
            "competencies": items,
        }

    # -- writing ------------------------------------------------------------

    def observe(self, answer: str) -> list[str]:
        """
        Credit an answer against every competency's signals. Returns the ids that gained one.

        Every competency, not just the focus — see the module docstring. Duplicate signals do
        not re-count: a candidate saying "sharding" four times has demonstrated it once, and
        counting repetition would let one enthusiastic sentence close a competency that
        `min_signals` was set higher to protect.
        """
        if not answer.strip():
            return []
        gained: list[str] = []
        for competency in self.plan.competencies:
            hits = terms_present(answer, competency.signals)
            state = self.coverage[competency.id]
            fresh = [hit for hit in hits if hit not in state.signals_hit]
            if not fresh:
                continue
            state.signals_hit.extend(fresh)
            state.evidence.append(answer.strip()[: self.EVIDENCE_CHARS])
            gained.append(competency.id)
        return gained

    def mark_asked(self, competency: Competency) -> None:
        self.coverage[competency.id].asked += 1
        self._focus = competency.id


def brief(state: PlanState) -> str:
    """
    The plan as the model sees it: where we are, what is next, and what not to do with it.

    Written as a standing brief rather than an instruction to ask a particular question, because
    the failure mode of putting a rubric in a prompt is that the model recites it — the same
    thing `CONTEXT_HEADER` in `augment.py` had to be rewritten to prevent when retrieved
    documents
    started being treated as directives. The clauses below exist for observed reasons:

    **"You are working through this plan, not reading from it"** — without it, the model
    announces the competency ("Now let us discuss debugging"), which turns a conversation into a
    form.

    **Naming what is already evidenced** — otherwise it re-asks. The model cannot see
    coverage; it only sees history, and history contains the candidate's answer but not the
    fact that the answer counted.

    **"go deeper rather than asking again"** — the case where the last answer already spoke to
    the focus. Without explicit permission to build on it, the model asks the queued question
    anyway and the candidate has to repeat themselves.
    """
    focus = state.focus()
    snapshot = state.snapshot()
    evidenced = [
        item["name"] for item in snapshot["competencies"] if item["status"] == "evidenced"
    ]
    remaining = [
        item["name"]
        for item in snapshot["competencies"]
        if item["status"] in ("unasked", "probing") and item["id"] != snapshot["focus"]
    ]

    if focus is None:
        # An interview with no next question still needs an ending. Left silent, the model has
        # a plan-shaped hole in its context and drifts into small talk.
        return (
            f"Interview plan '{state.plan.name}': every competency now has evidence. "
            "Begin closing the interview — invite the candidate's questions, or return once to "
            "the area where their answer was thinnest. Do not start a new topic."
        )

    lines = [
        f"Interview plan '{state.plan.name}'. You are working through this plan, not reading "
        "from it — never name a competency or announce that you are changing topic.",
        f"Probe next: {focus.name}." + (f" {focus.probe}" if focus.probe else ""),
    ]
    if evidenced:
        lines.append(
            "Already evidenced, do not re-ask: " + ", ".join(evidenced) + "."
        )
    if remaining:
        lines.append("Still to come, not yet: " + ", ".join(remaining) + ".")
    lines.append(
        "Ask one question that opens up the area above. If the candidate's last answer already "
        "spoke to it, go deeper on what they actually said rather than asking again."
    )
    return "\n".join(lines)


def with_plan(
    llm: SentenceStreamLike,
    plan: InterviewPlan,
    *,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> SentenceStreamLike:
    """
    Wrap an LLM so every turn is steered by the plan and updates its coverage.

    A decorator on `SentenceStream` for the same reason retrieval, pronunciation and guardrails
    are: an agenda is not a session-lifecycle concern, and the orchestrator's job — states,
    epochs, cancellation — is unchanged by it. This is the fourth implementation of that
    pattern, which is the strongest evidence available that the boundary was drawn in the right
    place.

    An empty plan returns the stream untouched, so an agent with no rubric adds no call at all
    and the clean-clone path the README promises keeps working.

    The order of operations inside is the part worth reading: observe the candidate's answer,
    *then* pick a focus, then inject. Reversed, the focus would be chosen from coverage that did
    not yet include the answer sitting right there in the history — so a competency the
    candidate had just demonstrated would still be selected, and the very next question would
    ask about it.

    `on_update` receives a coverage snapshot per turn. A callback rather than a `Telemetry`
    dependency because this module has no business knowing how a session reports things, and the
    server is already the place that decides what to relay and what to persist.
    """
    if not plan.active:
        return llm

    state = PlanState(plan)

    def planned(history: Sequence[Message]) -> AsyncGenerator[str, None]:
        state.observe(latest_candidate_text(history))
        focus = state.focus()
        if focus is not None:
            state.mark_asked(focus)
        if on_update is not None:
            on_update(state.snapshot())

        # Appended after the history, like retrieved context and for the same reason: a system
        # message at the front competes with the interviewer's own instructions and can override
        # its tone, while one at the end reads as the brief for *this* turn -- which it is.
        guidance: Message = {"role": "system", "content": brief(state)}
        return llm([*history, guidance])

    return planned
