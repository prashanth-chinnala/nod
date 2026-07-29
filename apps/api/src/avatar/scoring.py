"""
The scorecard: a model reading a finished transcript against the rubric it was interviewed on.

**Why this is a separate module from `plan.py`.** The plan decides what to ask next and must
answer before the next question is generated, so it lives on the critical path and is
deliberately restricted to string matching. This runs after the session ends, where a second of
latency costs nothing, so it is allowed the model the plan is not. Same rubric, two consumers,
opposite constraints -- one module holding both would have meant one set of trade-offs serving
neither.

**What it will not do, and this is a product decision rather than a technical limit.** There is
no hire recommendation, no overall verdict, no threshold. It produces one rating and its
supporting quotes per competency, plus a weighted total that is explicitly a summary of those
ratings rather than a decision derived from them. A hiring decision made by a model reading a
transcript is a decision nobody can be accountable for, and the person who has to defend it
needs the evidence laid out, not a number to defer to.

**Ratings are labels first and numbers second.** `strong` / `adequate` / `weak` / `no_evidence`
is what the model is asked for; the integer mapping exists only so a weighted total can be
computed. That order matters. Asking a model for "7 out of 10" invites a precision it does not
have, and the number would then be read as calibrated when nothing has calibrated it. The
mapping is a stated convention, declared in `RATINGS` and reported alongside every scorecard, so
a reader can check the arithmetic rather than trust it.

**Refusing to score is a first-class outcome.** With no model configured -- a clean clone, no
credentials -- this records `status="unavailable"` and the reason. The alternative would be a
scripted scorer producing plausible ratings from nothing, and a fabricated score in a hiring
record is the worst thing it is possible to build here.

**Each competency is judged independently, in parallel.** One request per competency rather than
one asking for the whole scorecard: a single call has to hold every competency's criteria at
once and drifts toward giving them all the same rating, and one malformed field would lose the
entire scorecard instead of one row. Parallel because they do not depend on each other and
nothing is waiting.

Nothing here imports a renderer, torch, or a web framework. The model is reached through a
`Judge` callable supplied by the caller, so the tests judge with a function and no network.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from avatar.plan import Competency, InterviewPlan

RATINGS: dict[str, int] = {
    "no_evidence": 0,
    "weak": 1,
    "adequate": 2,
    "strong": 3,
}
"""
The rating vocabulary and its numeric mapping, in one place and reported with every scorecard.

Four levels rather than five, because a five-point scale has a middle and a middle is where
uncertain judgements go to hide. Forcing a choice between `weak` and `adequate` makes the model
commit, and the rationale then has to justify the side it picked.

`no_evidence` is a rating rather than an absence. "We never got to it" and "they answered badly"
are different findings and a reader must be able to tell them apart -- the first is a fault in
the interview, the second in the answer.
"""

MAX_RATING = max(RATINGS.values())

JUDGE_SYSTEM = (
    "You assess one competency from an interview transcript. You are given the competency, "
    "what the interviewer was told to probe, and the transcript. Reply with JSON only: "
    '{"rating": one of "strong" | "adequate" | "weak" | "no_evidence", '
    '"rationale": one or two sentences, '
    '"quotes": up to three short verbatim quotes from the candidate}. '
    "Quote the candidate only, never the interviewer, and never paraphrase inside quotes -- a "
    "quote that does not appear in the transcript makes the whole assessment unusable. Use "
    '"no_evidence" when the transcript does not cover this competency at all; do not infer '
    "ability from an adjacent answer. Judge only what was said, not how fluently it was said."
)
"""
Every clause here defends against a specific way a judge goes wrong.

**"JSON only"**, plus tolerant parsing below. Models wrap JSON in prose or fences however firmly
they are told not to, so this reduces the rate and `_extract_json` absorbs the rest.

**"Quote the candidate only"** -- without it, judges quote the interviewer's own question as
evidence that the candidate demonstrated something, which is circular and reads as convincing.

**"never paraphrase inside quotes"** -- a fabricated quote is worse than no quote. Quotes are
the part of a scorecard a human checks; if those cannot be trusted the ratings cannot either,
which is why `verify_quotes` re-checks them against the transcript rather than taking the
model's word.

**"do not infer ability from an adjacent answer"** -- judges are strongly inclined to rate every
competency `adequate` from a general impression of the candidate, which produces a scorecard
that says the same thing about all six areas and so distinguishes nothing.

**"not how fluently it was said"** -- the transcript comes from an STT engine. Disfluency,
missing punctuation and dropped words are artefacts of transcription, and penalising them would
score the microphone.
"""

Judge = Callable[[str], Awaitable[str]]
"""
One prompt in, the model's raw reply out.

A plain callable rather than a client object, so this module never learns which provider it is
talking to and the tests judge with a function. The prompt is assembled here, which keeps what
the model is asked in the module that has to defend the answer.
"""


@dataclass(frozen=True)
class Verdict:
    """One competency, judged."""

    competency_id: str
    name: str
    rating: str
    score: int
    weight: float
    rationale: str = ""
    quotes: tuple[str, ...] = ()
    unverified_quotes: tuple[str, ...] = ()
    """
    Quotes the model attributed to the candidate that are not in the transcript.

    Kept rather than discarded. Dropping them silently would hide the most important signal
    about whether this scorecard can be trusted at all: a judge that invents evidence is not a
    judge whose ratings mean anything, and a reviewer needs to see that it happened.
    """

    def as_dict(self) -> dict[str, Any]:
        return {
            "competency_id": self.competency_id,
            "name": self.name,
            "rating": self.rating,
            "score": self.score,
            "weight": self.weight,
            "rationale": self.rationale,
            "quotes": list(self.quotes),
            "unverified_quotes": list(self.unverified_quotes),
        }


@dataclass(frozen=True)
class Scorecard:
    """
    A whole session, judged. `status` distinguishes a result from an absence.

    `weighted_score` is `None` unless the session was scored at all. A total computed from
    nothing looks like a low score rather than a missing one, and the difference decides whether
    a reviewer reads the transcript themselves or trusts the summary.
    """

    status: str
    model: str = ""
    reason: str = ""
    verdicts: tuple[Verdict, ...] = ()
    scale: Mapping[str, int] = field(default_factory=lambda: dict(RATINGS))

    @property
    def weighted_score(self) -> float | None:
        """
        The weighted ratings as a fraction of the maximum, or `None` when there is nothing to
        compute from.

        Normalised rather than raw, so two rubrics with different numbers of competencies
        produce comparable figures. Rounded to four places because the inputs are integers 0-3
        and a longer decimal would imply the weights were measured.
        """
        if self.status != "scored" or not self.verdicts:
            return None
        total_weight = sum(v.weight for v in self.verdicts)
        if not total_weight:
            return None
        earned = sum(v.score * v.weight for v in self.verdicts)
        return round(earned / (total_weight * MAX_RATING), 4)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "reason": self.reason,
            "scale": dict(self.scale),
            "max_rating": MAX_RATING,
            "weighted_score": self.weighted_score,
            "verdicts": [v.as_dict() for v in self.verdicts],
            # Stated in the data rather than left to a UI. A consumer must not render this as a
            # decision, and a convention held only in a component is one refactor from being
            # lost.
            "decision": None,
            "note": (
                "Model-generated assessment of a transcript. Ratings summarise the quotes; "
                "they are not a hiring decision, which is a human's."
            ),
        }


def transcript_text(turns: Sequence[Mapping[str, Any]]) -> str:
    """
    The conversation as the judge reads it.

    Interviewer turns are included even though only the candidate is judged: an answer read
    without its question is often unintelligible, and "they never mentioned monitoring" means
    something different depending on whether they were asked. Turns whose transcription failed
    are marked rather than dropped, because a session where the candidate spoke and nothing was
    captured must not read as one where they said nothing.

    **`heard` before `said`, which is the order the turn happened in.** A turn record is opened
    by the candidate finishing an answer and then accumulates the interviewer's reply to it, so
    within one record the answer precedes the question that follows it. This emitted `said`
    first, which inverted every pair: the judge read each interviewer reply *above* the answer
    it was replying to, so read linearly the candidate appeared to ignore every question and
    answer the previous one. Nothing errored and every rating stayed plausible, which is exactly
    why it survived a first review -- the transcript looked like a transcript.

    The consequence is that a turn's question introduces the *next* turn's answer, so the reply
    that closes the interview has no answer beneath it. That is a true rendering of the record
    rather than a gap to paper over.
    """
    lines: list[str] = []
    for turn in turns:
        answered = str(turn.get("heard") or "").strip()
        if answered:
            if turn.get("transcribed", True):
                lines.append(f"Candidate: {answered}")
            else:
                lines.append(f"Candidate: [speech detected, not transcribed] {answered}")
        asked = str(turn.get("said") or "").strip()
        if asked:
            lines.append(f"Interviewer: {asked}")
    return "\n".join(lines)


def candidate_text(turns: Sequence[Mapping[str, Any]]) -> str:
    """Only what the candidate said, which is what quotes are checked against."""
    return "\n".join(
        str(turn.get("heard") or "").strip()
        for turn in turns
        if str(turn.get("heard") or "").strip()
    )


def build_prompt(competency: Competency, transcript: str, coverage: Mapping[str, Any]) -> str:
    """
    The judge's prompt for one competency.

    Coverage is included, but framed as what the *interview* did rather than as a finding. It is
    passed because "never asked" and "asked three times and got nothing" deserve different
    ratings and the transcript alone makes those hard to separate. It is framed carefully
    because a judge shown `status: evidenced` will simply agree with it, which would make the
    scorecard an expensive restatement of the string matching it exists to improve on.
    """
    asked = coverage.get("asked", 0)
    hits = [str(hit) for hit in (coverage.get("signals_hit") or [])]
    facts = [f"The interviewer probed this area {asked} time(s)."]
    if hits:
        facts.append(
            "Rubric signal terms that appeared in the candidate's answers: "
            + ", ".join(hits)
            + ". A term appearing is not by itself competence; judge the surrounding answer."
        )
    else:
        facts.append(
            "No rubric signal term appeared verbatim. That is not itself a negative finding "
            "-- the candidate may have described the same thing in other words."
        )

    signals = ", ".join(competency.signals) or "(none given)"
    body = transcript or "(empty -- no turns were recorded)"
    return (
        f"Competency: {competency.name}\n"
        f"What the interviewer was told to probe: {competency.probe or '(not specified)'}\n"
        f"What the rubric treats as signals: {signals}\n"
        f"\nHow the interview went for this area:\n- "
        + "\n- ".join(facts)
        + f"\n\nTranscript:\n{body}\n"
    )


def _extract_json(reply: str) -> dict[str, Any]:
    """
    Pull an object out of a reply that may be fenced, prefixed, or chatty.

    Tolerant on purpose. Models wrap JSON in ```json fences or a sentence of preamble however
    firmly they are told not to, and a strict parse would turn a perfectly good judgement into a
    failed one. The outermost brace pair is taken rather than the first, so a nested object does
    not truncate the parse.
    """
    stripped = reply.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _normalise_rating(value: object) -> str:
    """
    Map whatever the model said onto the vocabulary, or `no_evidence`.

    Falling back to the lowest rating rather than a middle one, because an unparseable rating is
    an absence of judgement and must not be recorded as a positive finding. The rationale still
    reaches the reader, so a reviewer can see that the judge said something the scale could not
    hold.
    """
    text = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return text if text in RATINGS else "no_evidence"


def verify_quotes(quotes: Sequence[str], candidate_said: str) -> tuple[list[str], list[str]]:
    """
    Split the model's quotes into those actually present and those not.

    Whitespace-normalised and case-insensitive: a judge that reflows a line break or changes
    capitalisation has not fabricated anything, and flagging that would cry wolf until nobody
    read the flag. Anything beyond that counts as unverified -- the point is to catch invention,
    and invention does not look like a whitespace difference.
    """
    haystack = re.sub(r"\s+", " ", candidate_said).lower()
    verified: list[str] = []
    unverified: list[str] = []
    for quote in quotes:
        text = str(quote or "").strip().strip('"')
        if not text:
            continue
        needle = re.sub(r"\s+", " ", text).lower()
        (verified if needle in haystack else unverified).append(text)
    return verified, unverified


async def judge_competency(
    competency: Competency,
    transcript: str,
    candidate_said: str,
    coverage: Mapping[str, Any],
    judge: Judge,
) -> Verdict:
    """
    Judge one competency. Never raises: a failure becomes a `no_evidence` verdict that says so.

    Contained here rather than at the gather, because one competency's bad reply must not cost
    the other five their verdicts -- and a scorecard silently missing a row is worse than one
    carrying a row that admits it failed.
    """
    prompt = f"{JUDGE_SYSTEM}\n\n{build_prompt(competency, transcript, coverage)}"
    try:
        reply = await judge(prompt)
        parsed = _extract_json(reply)
    except Exception as exc:
        return Verdict(
            competency_id=competency.id,
            name=competency.name,
            rating="no_evidence",
            score=RATINGS["no_evidence"],
            weight=competency.weight,
            rationale=f"could not be assessed: {type(exc).__name__}: {exc}",
        )

    rating = _normalise_rating(parsed.get("rating"))
    raw = parsed.get("quotes") or []
    listed = raw if isinstance(raw, list) else [raw]
    verified, unverified = verify_quotes([str(item) for item in listed], candidate_said)
    return Verdict(
        competency_id=competency.id,
        name=competency.name,
        rating=rating,
        score=RATINGS[rating],
        weight=competency.weight,
        rationale=str(parsed.get("rationale") or "").strip(),
        quotes=tuple(verified),
        unverified_quotes=tuple(unverified),
    )


NO_RUBRIC = "the agent for this session has no rubric, so there is nothing to score against"
NO_TURNS = "no turns were recorded for this session"
NO_JUDGE = (
    "no model is configured to assess the transcript. Set AVATAR_LLM=openai (or anthropic) "
    "with credentials and re-score. A scorecard is not generated without one, because an "
    "invented rating in a hiring record is worse than no rating."
)


async def score_session(
    record: Mapping[str, Any],
    plan: InterviewPlan,
    judge: Judge | None,
    *,
    model: str = "",
) -> Scorecard:
    """
    Score one finished session against the rubric it was interviewed on.

    Three refusals before any model is called, each recording why rather than producing a
    number: no rubric, no turns, no judge. The third matters most -- a clean clone has no
    credentials, and a scorer that invented ratings in that case would put fabricated evidence
    in a hiring record.

    The rubric is passed in rather than read from the record, so a caller can re-score against a
    corrected one. A rubric edited after the interview is a legitimate thing to want; silently
    scoring against a rubric the interview was never conducted on is not, which is why the
    scorecard names the competencies it judged.
    """
    if not plan.active:
        return Scorecard(status="unavailable", reason=NO_RUBRIC)

    turns = list(record.get("turns") or [])
    if not turns:
        return Scorecard(status="unavailable", reason=NO_TURNS)

    if judge is None:
        return Scorecard(status="unavailable", reason=NO_JUDGE)

    transcript = transcript_text(turns)
    said = candidate_text(turns)
    coverage = {
        str(item.get("id")): item
        for item in (record.get("coverage") or {}).get("competencies") or []
    }

    verdicts = await asyncio.gather(
        *(
            judge_competency(
                competency, transcript, said, coverage.get(competency.id, {}), judge
            )
            for competency in plan.competencies
        )
    )
    return Scorecard(status="scored", model=model, verdicts=tuple(verdicts))
