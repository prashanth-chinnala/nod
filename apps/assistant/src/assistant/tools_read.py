"""
Read tools. Everything here is free, leaves no trace, and cannot change stored state.

**Why reads and writes are separate modules and separate lists.** Not tidiness — it means the
graph is handed two explicitly named collections, so granting a write is a visible edit rather
than a consequence of a decorator ending up in the wrong file. `tools_write.py` is where
anything with a side effect lives, and it is short on purpose.

**Where the value is.** The interesting tools here are not the ones that fetch a record — a
human can open the console for that. They are `compare_candidates`, `consistency_audit` and
`coverage_gaps`: questions that require holding several sessions side by side, which people are
measurably bad at and which is where an inconsistent rubric actually shows up. Finding a
contradiction in your own instrument is the highest-value thing this assistant does.

**One separation is load-bearing and never crossed.** `interview_quality` reports on the
*pipeline* — barge-ins, dropped frames, failed transcription, turn latency. That is deliberately
a different question from how the candidate did, because a candidate interrupted six times or
transcribed as silence scores low on our machinery rather than on competence, and conflating the
two is how a good candidate gets lost. No tool here mixes them into one number.
"""

from __future__ import annotations

from typing import Any

from avatar.store import NotFound, store
from langchain_core.tools import tool

MAX_ROWS = 50
"""
Ceiling on rows returned to the model.

Not for cost: an unbounded list quietly becomes the whole context window, and the failure is a
model that answers from the first twenty records and sounds equally confident. Truncation is
reported in the payload so the model can say the list was cut rather than presenting it as all.
"""

QUOTE_CHARS = 400


def _sessions() -> list[dict[str, Any]]:
    return store.list("sessions")


def _turn_count(session: dict[str, Any]) -> int:
    return len(session.get("turns") or [])


@tool
def list_sessions(scored_only: bool = False) -> dict[str, Any]:
    """List interview sessions, newest first, with their state.

    Use this to find a session id before fetching anything about it. Set scored_only when the
    question is about assessments rather than about which interviews happened.

    Returns id, agent, when it started, whether it ended, turn count, and scoring status.
    """
    rows = []
    for session in _sessions():
        scoring = session.get("scoring") or {}
        if scored_only and scoring.get("status") != "scored":
            continue
        rows.append(
            {
                "id": session["id"],
                "agent_id": session.get("agent_id"),
                "started_at": session.get("started_at"),
                "ended": bool(session.get("ended_at")),
                "turns": _turn_count(session),
                "scoring_status": scoring.get("status", "none"),
                "weighted_score": scoring.get("weighted_score"),
            }
        )
    return {"sessions": rows[:MAX_ROWS], "total": len(rows), "truncated": len(rows) > MAX_ROWS}


@tool
def get_transcript(session_id: str) -> dict[str, Any]:
    """Fetch what was actually said in one interview, in order.

    Each turn has the candidate's answer and the interviewer's reply to it. A turn marked
    transcribed=false means the candidate spoke and no words were captured — that is a
    transcription failure, not silence, and must not be read as the candidate saying nothing.
    """
    try:
        session = store.get("sessions", session_id)
    except NotFound:
        return {"error": f"no session {session_id}"}
    return {
        "session_id": session_id,
        "turns": [
            {
                "epoch": turn.get("epoch"),
                "candidate_said": turn.get("heard", ""),
                "interviewer_asked": turn.get("said", ""),
                "transcribed": turn.get("transcribed", True),
                "interrupted": turn.get("interrupted", False),
            }
            for turn in (session.get("turns") or [])
        ],
    }


@tool
def get_scorecard(session_id: str) -> dict[str, Any]:
    """Fetch the assessment for one interview: per-competency rating, rationale, and quotes.

    The quotes have been checked against the transcript. Any listed under unverified_quotes were
    attributed to the candidate by the judge but do NOT appear in what they said — if that list
    is non-empty, say so prominently and treat the whole scorecard as unreliable.

    There is no hire recommendation in here by design. Do not supply one.
    """
    try:
        session = store.get("sessions", session_id)
    except NotFound:
        return {"error": f"no session {session_id}"}
    scoring = session.get("scoring") or {}
    if scoring.get("status") != "scored":
        return {
            "session_id": session_id,
            "status": scoring.get("status", "none"),
            "reason": scoring.get("reason", "this session has not been scored"),
        }
    return {
        "session_id": session_id,
        "status": "scored",
        "model": scoring.get("model"),
        "weighted_score": scoring.get("weighted_score"),
        "scale": scoring.get("scale"),
        "verdicts": scoring.get("verdicts"),
        "note": scoring.get("note"),
    }


@tool
def get_coverage(session_id: str) -> dict[str, Any]:
    """What the interview actually probed, competency by competency.

    Distinct from the scorecard and often more useful. `asked: 0` with a `no_evidence` rating
    means the interview never covered that area — a gap in the interview, not a finding about
    the candidate. `exhausted` means it was probed to its limit and produced no signal.
    """
    try:
        session = store.get("sessions", session_id)
    except NotFound:
        return {"error": f"no session {session_id}"}
    coverage = session.get("coverage") or {}
    return {
        "session_id": session_id,
        "plan": coverage.get("plan"),
        "complete": coverage.get("complete"),
        "competencies": coverage.get("competencies") or [],
    }


@tool
def interview_quality(session_id: str) -> dict[str, Any]:
    """How well the *pipeline* performed for one interview — not how the candidate did.

    Use this before drawing any conclusion from a low score. A candidate interrupted repeatedly,
    or whose speech was never transcribed, or who waited many seconds for every reply, was
    assessed under conditions that depress answers. Report this as a separate finding; never
    fold it into a judgement about the person.
    """
    try:
        session = store.get("sessions", session_id)
    except NotFound:
        return {"error": f"no session {session_id}"}
    turns = session.get("turns") or []
    totals = [t["perceived_total_ms"] for t in turns if t.get("perceived_total_ms") is not None]
    untranscribed = [t.get("epoch") for t in turns if not t.get("transcribed", True)]
    return {
        "session_id": session_id,
        "turns": len(turns),
        "interrupted_turns": sum(1 for t in turns if t.get("interrupted")),
        "untranscribed_turns": untranscribed,
        "stale_artifacts_dropped": session.get("stale_dropped", 0),
        "worst_end_to_end_ms": max(totals) if totals else None,
        "median_end_to_end_ms": sorted(totals)[len(totals) // 2] if totals else None,
        "recording": (session.get("recording") or {}).get("status", "unknown"),
        "note": (
            "These are pipeline measurements. A session with untranscribed turns was scored on "
            "words the model never received."
        ),
    }


@tool
def compare_candidates(session_ids: list[str]) -> dict[str, Any]:
    """Put several candidates side by side, evidence against evidence.

    Aligns them by competency so the comparison is between what each actually said, not between
    two numbers. Prefer this over fetching scorecards one at a time — the whole point is the
    alignment.

    Sessions that were never scored are listed separately rather than treated as low scores.
    """
    aligned: dict[str, list[dict[str, Any]]] = {}
    unscored: list[str] = []
    for session_id in session_ids[:MAX_ROWS]:
        try:
            session = store.get("sessions", session_id)
        except NotFound:
            unscored.append(f"{session_id} (not found)")
            continue
        scoring = session.get("scoring") or {}
        if scoring.get("status") != "scored":
            unscored.append(f"{session_id} ({scoring.get('status', 'never scored')})")
            continue
        for verdict in scoring.get("verdicts") or []:
            aligned.setdefault(verdict["name"], []).append(
                {
                    "session_id": session_id,
                    "rating": verdict["rating"],
                    "score": verdict["score"],
                    "rationale": verdict.get("rationale", "")[:QUOTE_CHARS],
                    "quotes": verdict.get("quotes") or [],
                    "unverified_quotes": verdict.get("unverified_quotes") or [],
                }
            )
    return {
        "by_competency": aligned,
        "unscored": unscored,
        "note": "Compare the quotes, not the numbers. The numbers summarise the quotes.",
    }


@tool
def consistency_audit(competency: str = "") -> dict[str, Any]:
    """Find places where the instrument contradicts itself across sessions.

    Groups every rating for a competency with the evidence behind it, so a case where one
    candidate scored higher on weaker evidence becomes visible. This is the highest-value check
    available: humans are poor at spotting it, and an inconsistent rubric silently mis-ranks
    everyone.

    Pass a competency name to narrow, or leave empty for all. What comes back is grouped
    evidence — judge whether the ratings are defensible; do not assume a disagreement is an
    error.
    """
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for session in _sessions():
        scoring = session.get("scoring") or {}
        if scoring.get("status") != "scored":
            continue
        for verdict in scoring.get("verdicts") or []:
            name = verdict["name"]
            if competency and competency.lower() not in name.lower():
                continue
            grouped.setdefault(name, {}).setdefault(verdict["rating"], []).append(
                {
                    "session_id": session["id"],
                    "quotes": verdict.get("quotes") or [],
                    "rationale": verdict.get("rationale", "")[:QUOTE_CHARS],
                }
            )
    return {
        "by_competency_and_rating": grouped,
        "note": (
            "Within one competency, read the evidence across ratings. Two candidates given "
            "different ratings on comparable answers is a rubric problem, not a candidate "
            "problem."
        ),
    }


@tool
def coverage_gaps() -> dict[str, Any]:
    """Which competencies are being skipped across the whole pipeline.

    A competency that is never probed means the rubric is being ignored in practice — the
    interview is not asking what someone decided it should ask. That is a finding about the
    system, and it is invisible from any single session.
    """
    probed: dict[str, int] = {}
    unprobed: dict[str, int] = {}
    for session in _sessions():
        for item in (session.get("coverage") or {}).get("competencies") or []:
            bucket = probed if item.get("asked", 0) > 0 else unprobed
            bucket[item["name"]] = bucket.get(item["name"], 0) + 1
    never = sorted(name for name in unprobed if name not in probed)
    return {
        "probed_counts": probed,
        "never_probed_counts": unprobed,
        "never_probed_at_all": never,
        "note": (
            "A competency in never_probed_at_all was on a rubric for every session it "
            "appears in and asked about in none of them."
        ),
    }


@tool
def action_history(target_id: str) -> dict[str, Any]:
    """Every change the assistant has proposed about one session, rubric or competency.

    Use this before proposing something, so the same change is not proposed twice, and to answer
    "has anyone already flagged this".
    """
    from assistant.audit import history

    return {"target": target_id, "actions": history(target_id)}


READ_TOOLS = [
    list_sessions,
    get_transcript,
    get_scorecard,
    get_coverage,
    interview_quality,
    compare_candidates,
    consistency_audit,
    coverage_gaps,
    action_history,
]
"""
The read-only set, named explicitly.

Enumerated rather than collected by introspection, so adding a tool to this module does not
grant it to the assistant by accident -- and so this list can be read as the answer to "what can
it see".
"""
