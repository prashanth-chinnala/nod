"""
Write tools. Short on purpose, and not one of them changes a rubric or a score.

**What "write" means here.** Every tool in this module records a *proposal* and returns its id.
None of them edits a rubric, a scorecard, a rating or a weight. A human applies a proposal
through the console, and that act is recorded separately with their name on it. So the trail
always answers two questions a single `updated_at` cannot: what did a model suggest, and what
did a person decide.

That is not caution for its own sake. It is the difference between an assistant that helps a
hiring decision and one that makes hiring decisions nobody can be accountable for — and it is
the position you want to be in when a rejected candidate asks why: a versioned rubric, a
verified quote, and a named human.

**The one loop that must never close.** `request_rescore` and `propose_rubric_change` both exist
to improve the *instrument*. Neither can adjust an individual's score, and nothing in this
package routes manager behaviour into a candidate's assessment. Two loops, never joined:

    candidate score      <- transcript + rubric + anchors. Nothing else, ever.
    instrument quality   <- aggregated human overrides -> rubric wording, weights, anchors

If managers consistently override the judge on one competency, the rubric or the prompt is
miscalibrated: fix it for everyone, going forward. Do not nudge that candidate's number. Same
information, and it improves the measurement instead of contaminating it. Feeding who gets hired
back into who scores well is the mechanism behind Amazon's abandoned recruiting tool, and it is
what the bias-audit requirements in NYC LL144 and the EU AI Act exist to catch.

**Explicit labels are signal; implicit behaviour is not.** A manager writing "this rating is
wrong, she did demonstrate X, here is the quote" is deliberate, attributable and reviewable —
good evaluation data, and `add_note` captures it. Dwell time, click counts and hire outcomes are
none of those things, and there is deliberately no tool for them.
"""

from __future__ import annotations

from typing import Any

from avatar.store import NotFound, store
from langchain_core.tools import tool

from assistant.audit import record


def _exists(collection: str, record_id: str) -> bool:
    try:
        store.get(collection, record_id)
    except NotFound:
        return False
    return True


@tool
def flag_for_review(session_id: str, reason: str, actor: str = "unknown") -> dict[str, Any]:
    """Mark a session as needing a human to look at it, with the reason.

    Use for things a person must judge: an unverified quote in the scorecard, a competency rated
    without ever being asked, an interview whose pipeline misbehaved, a contradiction against
    another candidate. State what you saw, not what should be done about it.
    """
    if not _exists("sessions", session_id):
        return {"error": f"no session {session_id}"}
    if not reason.strip():
        return {"error": "a flag with no reason is not reviewable; say what you saw"}
    action = record(
        "flagged_for_review",
        target=session_id,
        summary=reason.strip()[:300],
        actor=actor,
        detail={"session_id": session_id},
    )
    return {"action_id": action["id"], "status": "proposed", "target": session_id}


@tool
def add_note(
    session_id: str, note: str, competency: str = "", actor: str = "unknown"
) -> dict[str, Any]:
    """Attach a human's explicit correction or observation to a session.

    This is for a deliberate statement — "the rating is wrong, he did demonstrate X, here is the
    quote" — which is reviewable evaluation data. Record it verbatim; do not summarise away the
    quote, because the quote is what makes the correction arguable rather than a vibe.

    A note never changes a rating. It is evidence that the rating may be miscalibrated.
    """
    if not _exists("sessions", session_id):
        return {"error": f"no session {session_id}"}
    if not note.strip():
        return {"error": "an empty note records nothing"}
    action = record(
        "note_added",
        target=session_id,
        summary=(f"[{competency}] " if competency else "") + note.strip()[:300],
        actor=actor,
        detail={"session_id": session_id, "competency": competency, "note": note.strip()},
    )
    return {"action_id": action["id"], "status": "proposed", "target": session_id}


@tool
def request_rescore(session_id: str, reason: str, actor: str = "unknown") -> dict[str, Any]:
    """Propose that a session be assessed again, with why.

    Records a request; it does not run the scorer. Re-scoring overwrites an assessment, so it is
    a human's call — and a scorecard that can be re-rolled until it reads well is not evidence.
    Legitimate reasons: the rubric was corrected, the previous run came back unavailable, the
    judge model changed, a quote failed verification.
    """
    if not _exists("sessions", session_id):
        return {"error": f"no session {session_id}"}
    action = record(
        "rescore_requested",
        target=session_id,
        summary=f"re-score: {reason.strip()[:280]}",
        actor=actor,
        detail={"session_id": session_id, "reason": reason.strip()},
    )
    return {
        "action_id": action["id"],
        "status": "proposed",
        "note": "A human applies this from the session's report page. The scorer has not run.",
    }


@tool
def propose_rubric_change(
    rubric_id: str,
    competency: str,
    change: str,
    evidence: str,
    actor: str = "unknown",
) -> dict[str, Any]:
    """Draft a change to a rubric, with the evidence that motivates it.

    This is the correct destination for a pattern seen across sessions — a competency nobody
    probes, a signal list that never matches how candidates actually talk, ratings that
    contradict each other on comparable answers. The change applies to everyone going forward,
    which is what makes it a calibration fix rather than a re-judgement of one person.

    `evidence` is required and must cite sessions or quotes. A proposed change with no evidence
    is an opinion, and the reviewer cannot evaluate it.
    """
    if not _exists("rubrics", rubric_id):
        return {"error": f"no rubric {rubric_id}"}
    if not evidence.strip():
        return {
            "error": (
                "cite the sessions or quotes that motivate this; a change with no evidence "
                "cannot be reviewed"
            )
        }
    action = record(
        "rubric_change_proposed",
        target=rubric_id,
        summary=f"{competency}: {change.strip()[:240]}",
        actor=actor,
        detail={
            "rubric_id": rubric_id,
            "competency": competency,
            "change": change.strip(),
            "evidence": evidence.strip(),
        },
    )
    return {
        "action_id": action["id"],
        "status": "proposed",
        "note": "Nothing changed. A human reviews and applies this in the console.",
    }


@tool
def propose_calibration_anchor(
    rubric_id: str,
    competency: str,
    rating: str,
    quote: str,
    session_id: str,
    actor: str = "unknown",
) -> dict[str, Any]:
    """Propose a verified quote as an example of what a given rating sounds like.

    An anchor describes the *bar*, not a person: future assessments compare an answer to an
    exemplar answer. That is what makes it safe where learning from who got hired is not — it
    anchors the scale rather than modelling a candidate, and promotion is a deliberate human act
    rather than something harvested from outcomes.

    The quote must be one that already passed verification against that session's transcript. Do
    not paraphrase it and do not compose one: an invented anchor would silently mis-calibrate
    every assessment that follows.
    """
    if not _exists("rubrics", rubric_id):
        return {"error": f"no rubric {rubric_id}"}
    if not _exists("sessions", session_id):
        return {"error": f"no session {session_id}"}

    # Checked here rather than trusted, for the same reason `verify_quotes` exists: an anchor is
    # load-bearing for every future assessment, so one that was never actually said is worse
    # than no anchor. This confirms the quote appears among the *verified* quotes of that
    # session's
    # scorecard, not merely somewhere in the transcript.
    session = store.get("sessions", session_id)
    verified: list[str] = []
    for verdict in (session.get("scoring") or {}).get("verdicts") or []:
        verified.extend(verdict.get("quotes") or [])
    normalised = " ".join(quote.split()).lower()
    if not any(normalised in " ".join(v.split()).lower() for v in verified):
        return {
            "error": (
                "that quote is not among the verified quotes on this session's scorecard. An "
                "anchor has to be something the candidate demonstrably said."
            ),
            "verified_quotes_available": verified[:10],
        }

    action = record(
        "anchor_promotion_proposed",
        target=rubric_id,
        summary=f"{competency} = {rating}: {quote.strip()[:200]}",
        actor=actor,
        detail={
            "rubric_id": rubric_id,
            "competency": competency,
            "rating": rating,
            "quote": quote.strip(),
            "source_session": session_id,
        },
    )
    return {
        "action_id": action["id"],
        "status": "proposed",
        "note": (
            "Nothing changed. A human promotes this in the console, which is the gate that "
            "separates calibration from laundering last cycle's preferences into this one."
        ),
    }


WRITE_TOOLS = [
    flag_for_review,
    add_note,
    request_rescore,
    propose_rubric_change,
    propose_calibration_anchor,
]
"""
The complete set of things this assistant can do to stored state, and all five only propose.

Enumerated so this list is readable as the answer to "what can it change". Note what is absent
and will stay absent: no tool sets a rating, edits a weight, applies its own proposal, or
records anything derived from how a manager behaved rather than what they said.
"""
