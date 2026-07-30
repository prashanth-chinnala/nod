"""
Tests for the scorecard.

Weighted heavily toward the refusal paths and quote verification, because those are what stop
this module producing a plausible-looking hiring record out of nothing. A scorer that rates
confidently from an empty transcript would pass any test written about its happy path.
"""

from __future__ import annotations

import json

import pytest

from avatar.plan import Competency, InterviewPlan, slug
from avatar.scoring import (
    MAX_RATING,
    RATINGS,
    Scorecard,
    Verdict,
    build_prompt,
    candidate_text,
    judge_competency,
    score_session,
    transcript_text,
    verify_quotes,
)


def competency(name: str, *signals: str, weight: float = 1.0) -> Competency:
    return Competency(
        id=slug(name), name=name, probe=f"probe {name}", signals=signals, weight=weight
    )


def plan(*competencies: Competency) -> InterviewPlan:
    return InterviewPlan(name="Backend", competencies=competencies)


def turn(
    heard: str, said: str, *, transcribed: bool = True, epoch: int = 1
) -> dict[str, object]:
    return {"epoch": epoch, "heard": heard, "said": said, "transcribed": transcribed}


def replying(payload: object) -> object:
    """A judge that always returns the same reply, so a test controls exactly what came back."""

    async def judge(_prompt: str) -> str:
        return payload if isinstance(payload, str) else json.dumps(payload)

    return judge


# -- refusals -------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_rubric_is_unavailable_not_zero() -> None:
    """
    A session whose agent had no rubric has nothing to score against.

    Returning a zero score instead would be indistinguishable from a candidate who answered
    nothing, and the two are opposite findings.
    """
    card = await score_session({"turns": [turn("hello", "hi")]}, InterviewPlan(), replying({}))
    assert card.status == "unavailable"
    assert card.weighted_score is None
    assert "no rubric" in card.reason


@pytest.mark.asyncio
async def test_no_turns_is_unavailable() -> None:
    card = await score_session({"turns": []}, plan(competency("Scale")), replying({}))
    assert card.status == "unavailable"
    assert "no turns" in card.reason


@pytest.mark.asyncio
async def test_no_judge_refuses_rather_than_inventing_ratings() -> None:
    """
    The most important assertion in this file.

    A clean clone has no credentials. A scorer that produced ratings anyway would be writing
    fabricated evidence into a hiring record, and it would look exactly like a real scorecard.
    """
    card = await score_session(
        {"turns": [turn("I sharded it", "tell me about scale")]},
        plan(competency("Scale", "sharding")),
        None,
    )
    assert card.status == "unavailable"
    assert card.verdicts == ()
    assert card.weighted_score is None
    assert "AVATAR_LLM" in card.reason  # says how to fix it, not merely that it failed


# -- judging one competency -----------------------------------------------


@pytest.mark.asyncio
async def test_verdict_carries_rating_rationale_and_verified_quotes() -> None:
    card = await score_session(
        {"turns": [turn("We moved to a partition scheme per tenant.", "how did you scale?")]},
        plan(competency("Scale", "partition")),
        replying(
            {
                "rating": "strong",
                "rationale": "Described a concrete partitioning strategy.",
                "quotes": ["We moved to a partition scheme per tenant."],
            }
        ),
    )
    assert card.status == "scored"
    verdict = card.verdicts[0]
    assert verdict.rating == "strong"
    assert verdict.score == RATINGS["strong"]
    assert verdict.quotes == ("We moved to a partition scheme per tenant.",)
    assert verdict.unverified_quotes == ()


@pytest.mark.asyncio
async def test_fabricated_quotes_are_kept_and_flagged() -> None:
    """
    A judge inventing evidence must be visible, not tidied away.

    Dropping the quote silently would leave a confident rating with no support and no sign that
    anything went wrong — which is the one failure that makes every other rating untrustworthy.
    """
    card = await score_session(
        {"turns": [turn("I read the logs.", "how do you debug?")]},
        plan(competency("Debugging", "profiler")),
        replying(
            {
                "rating": "strong",
                "rationale": "Deep profiling experience.",
                "quotes": ["I attached a flame-graph profiler in production."],
            }
        ),
    )
    verdict = card.verdicts[0]
    assert verdict.quotes == ()
    assert verdict.unverified_quotes == ("I attached a flame-graph profiler in production.",)


@pytest.mark.asyncio
async def test_json_wrapped_in_prose_and_fences_still_parses() -> None:
    """
    Models fence and preface JSON however firmly they are told not to.

    A strict parse would turn a perfectly good judgement into an unassessed competency, so the
    tolerance is deliberate rather than sloppy.
    """
    card = await score_session(
        {"turns": [turn("I sharded it.", "scale?")]},
        plan(competency("Scale", "sharding")),
        replying(
            'Here is my assessment:\n```json\n{"rating": "adequate", "rationale": "ok", '
            '"quotes": []}\n```\nHope that helps.'
        ),
    )
    assert card.verdicts[0].rating == "adequate"


@pytest.mark.asyncio
async def test_unknown_rating_falls_back_to_no_evidence_not_a_middle_value() -> None:
    """
    An unparseable rating is an absence of judgement and must not become a positive finding.

    The rationale still reaches the reader, so a reviewer can see the judge said something the
    scale could not hold.
    """
    card = await score_session(
        {"turns": [turn("I sharded it.", "scale?")]},
        plan(competency("Scale", "sharding")),
        replying({"rating": "excellent, 9/10", "rationale": "very good", "quotes": []}),
    )
    verdict = card.verdicts[0]
    assert verdict.rating == "no_evidence"
    assert verdict.rationale == "very good"


@pytest.mark.asyncio
async def test_rating_is_normalised_across_spacing_and_case() -> None:
    card = await score_session(
        {"turns": [turn("x", "y")]},
        plan(competency("Scale")),
        replying({"rating": "No Evidence", "rationale": "", "quotes": []}),
    )
    assert card.verdicts[0].rating == "no_evidence"


@pytest.mark.asyncio
async def test_a_failing_judge_costs_one_row_not_the_scorecard() -> None:
    """
    One competency's bad reply must not lose the others their verdicts.

    A scorecard silently missing a row is worse than one carrying a row that admits it failed,
    because the missing row is invisible while the total quietly changes.
    """

    async def judge(prompt: str) -> str:
        if "Debugging" in prompt:
            raise RuntimeError("provider exploded")
        return json.dumps({"rating": "strong", "rationale": "good", "quotes": []})

    card = await score_session(
        {"turns": [turn("I sharded it.", "scale?")]},
        plan(competency("Scale", "sharding"), competency("Debugging", "profiler")),
        judge,
    )
    assert card.status == "scored"
    assert len(card.verdicts) == 2
    failed = next(v for v in card.verdicts if v.name == "Debugging")
    assert failed.rating == "no_evidence"
    assert "provider exploded" in failed.rationale


# -- arithmetic -----------------------------------------------------------


def test_weighted_score_is_normalised_so_rubrics_are_comparable() -> None:
    """A rubric with three competencies and one with six must produce comparable figures."""
    card = Scorecard(
        status="scored",
        verdicts=(
            Verdict("a", "A", "strong", RATINGS["strong"], 1.0),
            Verdict("b", "B", "no_evidence", RATINGS["no_evidence"], 1.0),
        ),
    )
    # 3 + 0 earned out of 2 competencies * max 3 = 6 available.
    assert card.weighted_score == pytest.approx(0.5)


def test_weights_actually_shift_the_total() -> None:
    """
    Otherwise the field is decorative, which is the quiet way a configuration option lies.
    """
    heavy = Scorecard(
        status="scored",
        verdicts=(
            Verdict("a", "A", "strong", RATINGS["strong"], 3.0),
            Verdict("b", "B", "no_evidence", RATINGS["no_evidence"], 1.0),
        ),
    )
    assert heavy.weighted_score == pytest.approx(9 / 12)


def test_unavailable_scorecard_has_no_total() -> None:
    assert Scorecard(status="unavailable", reason="x").weighted_score is None


def test_scale_travels_with_the_scorecard() -> None:
    """
    The label-to-number mapping is a convention, so a reader must be able to check the
    arithmetic rather than trust it. Shipping the scale in the record is what makes that
    possible a month later, when the constant may have changed.
    """
    payload = Scorecard(status="scored", verdicts=()).as_dict()
    assert payload["scale"] == RATINGS
    assert payload["max_rating"] == MAX_RATING


def test_scorecard_states_it_is_not_a_decision() -> None:
    """
    A consumer must not render this as a verdict, and a convention held only in a UI component
    is one refactor away from being lost — so it lives in the data.
    """
    payload = Scorecard(status="scored", verdicts=()).as_dict()
    assert payload["decision"] is None
    assert "not a hiring decision" in payload["note"]


def test_scorecard_survives_a_json_round_trip() -> None:
    payload = Scorecard(
        status="scored",
        model="gpt-oss:20b",
        verdicts=(Verdict("a", "A", "weak", 1, 1.0, "thin", ("said it",), ("made up",)),),
    ).as_dict()
    assert json.loads(json.dumps(payload)) == payload


# -- transcript assembly --------------------------------------------------


def test_transcript_includes_the_questions() -> None:
    """
    An answer read without its question is often unintelligible, and "they never mentioned
    monitoring" means something different depending on whether they were asked.
    """
    text = transcript_text([turn("We used Kafka.", "What queue did you use?")])
    assert "Interviewer: What queue did you use?" in text
    assert "Candidate: We used Kafka." in text


def test_transcript_is_in_the_order_the_turn_happened() -> None:
    """
    Within a turn record the answer comes first and the interviewer's reply to it second, which
    is the order the runtime writes them in.

    This emitted `said` before `heard`, inverting every pair -- so the judge read each reply
    above the answer it was replying to, and read linearly the candidate appeared to ignore
    every question and answer the previous one. Nothing errored and the ratings stayed
    plausible, which is precisely why it survived being looked at once: a transcript in the
    wrong order still reads like a transcript.
    """
    text = transcript_text(
        [
            turn("Six years on backend.", "Tell me about scaling a dataset."),
            turn("We partitioned per tenant.", "How did you pick the partition key?"),
        ]
    )
    assert text.splitlines() == [
        "Candidate: Six years on backend.",
        "Interviewer: Tell me about scaling a dataset.",
        "Candidate: We partitioned per tenant.",
        "Interviewer: How did you pick the partition key?",
    ]


def test_untranscribed_speech_is_marked_not_dropped() -> None:
    """
    A session where the candidate spoke and nothing was captured must not read as one where they
    said nothing — that is a broken STT configuration, and it must not be scored as silence.
    """
    text = transcript_text(
        [turn("[1200ms of speech, no transcript]", "Go on?", transcribed=False)]
    )
    assert "not transcribed" in text


def test_quotes_are_checked_against_the_candidate_only() -> None:
    """
    Judges quote the interviewer's own question as evidence the candidate demonstrated
    something, which is circular and reads convincingly. The check is what makes that
    detectable.
    """
    turns = [turn("I read the logs.", "Did you use a profiler?")]
    verified, unverified = verify_quotes(["Did you use a profiler?"], candidate_text(turns))
    assert verified == []
    assert unverified == ["Did you use a profiler?"]


def test_reflowed_whitespace_is_not_treated_as_fabrication() -> None:
    """Crying wolf on a line break would make the flag worthless."""
    verified, unverified = verify_quotes(["We  used\nKAFKA."], "We used Kafka.")
    assert verified == ["We  used\nKAFKA."]
    assert unverified == []


# -- the prompt -----------------------------------------------------------


def test_prompt_reports_coverage_without_handing_over_the_verdict() -> None:
    """
    Coverage is context about what the interview did, not a finding to agree with.

    A judge shown `status: evidenced` simply agrees, which would make the scorecard an expensive
    restatement of the string matching it exists to improve on.
    """
    prompt = build_prompt(
        competency("Scale", "sharding"),
        "Interviewer: scale?\nCandidate: we did sharding",
        {"asked": 2, "signals_hit": ["sharding"], "status": "evidenced"},
    )
    assert "probed this area 2 time(s)" in prompt
    assert "sharding" in prompt
    assert "not by itself competence" in prompt
    assert "evidenced" not in prompt  # the status itself is withheld


def test_prompt_says_a_missing_signal_is_not_a_negative_finding() -> None:
    """The candidate may have described the same thing in different words."""
    prompt = build_prompt(competency("Scale", "sharding"), "transcript", {"asked": 1})
    assert "not itself a negative finding" in prompt


@pytest.mark.asyncio
async def test_prompt_reaches_the_judge_with_the_system_instruction() -> None:
    """The instructions defend against specific judge failures; none may be lost in wiring."""
    seen: list[str] = []

    async def judge(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"rating": "weak", "rationale": "", "quotes": []})

    await judge_competency(competency("Scale", "sharding"), "transcript", "said", {}, judge)
    assert "Quote the candidate only" in seen[0]
    assert "Competency: Scale" in seen[0]


@pytest.mark.asyncio
async def test_every_call_failing_is_unavailable_not_a_zero_score() -> None:
    """
    The worst outcome this module can produce, and it did.

    A failed judge call becomes `no_evidence` with score 0 and the competency's weight intact,
    so a run where every call failed returned `status="scored"` with `weighted_score: 0.0` — and
    the report rendered "Weighted score 0%" beside "Judged by <model>". An unmeasured zero in a
    hiring record is precisely what the module docstring calls the worst thing it is possible to
    build, and the file already asserted the opposite invariant: a total computed from nothing
    must look missing, not low.

    One provider 429 is enough to reach it.
    """

    async def broken(_prompt: str) -> str:
        raise RuntimeError("429 insufficient_quota")

    card = await score_session(
        {"turns": [turn("I sharded it.", "scale?")]},
        plan(competency("Scale", "sharding"), competency("Debugging", "profiler")),
        broken,
    )
    assert card.status == "unavailable"
    assert card.weighted_score is None
    assert "429" in card.reason


@pytest.mark.asyncio
async def test_a_partial_failure_withholds_the_total() -> None:
    """
    Three of six failing halves the number with nothing at the scorecard level saying so.

    The per-competency rows survive — one bad reply must not cost the others their verdicts —
    but the aggregate is withheld, because a total that silently counts failures as genuine
    zeros is worse than no total. A reader comparing two candidates cannot see that one was
    scored against a working judge and the other was not.
    """

    async def flaky(prompt: str) -> str:
        if "Debugging" in prompt:
            raise RuntimeError("provider exploded")
        return json.dumps({"rating": "strong", "rationale": "good", "quotes": []})

    card = await score_session(
        {"turns": [turn("I sharded it.", "scale?")]},
        plan(competency("Scale", "sharding"), competency("Debugging", "profiler")),
        flaky,
    )
    assert card.status == "scored"  # the successful row is still worth having
    assert card.weighted_score is None  # but the total is not reportable
    assert [v.assessed for v in card.verdicts].count(False) == 1


@pytest.mark.asyncio
async def test_a_real_no_evidence_still_counts_toward_the_total() -> None:
    """
    The distinction the `assessed` flag exists for.

    A judge that looked and found nothing is a measurement; a judge that never answered is not.
    Both produce `no_evidence`, so without the flag the fix above would suppress the total for
    every honest scorecard containing an unprobed competency — which is most of them.
    """
    card = await score_session(
        {"turns": [turn("I read the logs.", "how do you debug?")]},
        plan(competency("Debugging", "profiler")),
        replying({"rating": "no_evidence", "rationale": "not discussed", "quotes": []}),
    )
    assert card.status == "scored"
    assert card.verdicts[0].assessed is True
    assert card.weighted_score == 0.0  # a measured zero, and reportable as one
