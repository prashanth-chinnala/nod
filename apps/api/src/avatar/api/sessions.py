"""
Conversation records: what was said, what it cost, and what got interrupted.

**Append-only, deliberately.** There is no PATCH and no DELETE. A conversation record that can
be edited is not evidence, and evidence is the entire reason this collection exists — every
latency figure in `PROCESS.md` traces back to observed turns, and a record that could be revised
after the fact would make those figures unciteable. Turns are appended; a session is ended once.

**Why the per-turn shape mirrors the telemetry rather than the UI.** A turn stores the four
stage timings the runtime already emits (`llm_ttft_ms`, `tts_first_audio_ms`, `first_frame_ms`,
`perceived_total_ms`) plus `heard`, `said`, and whether the transcriber produced anything.
Storing a pre-computed "total" instead would lose the breakdown, and the breakdown is the
finding: none of the three dominant terms is the renderer.

**`transcribed` is stored separately from `heard` on purpose.** An empty transcript is not the
same as a silent candidate, and conflating them hides the failure that took a day to find — the
interviewer asking a plausible question that ignores the answer, because no words ever reached
it. A session where `transcribed` is false on every turn is a broken STT configuration, and that
must be visible in a list rather than needing a log grep.
"""

from __future__ import annotations

import contextlib
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from avatar.store import NotFound, now_iso, store

COLLECTION = "sessions"
ID_PREFIX = "sess"

router = APIRouter(prefix="/sessions", tags=["sessions"])


class Turn(BaseModel):
    """
    One exchange. Every timing is optional because a turn can be cut off before it happens.

    A barge-in during `THINKING` produces a turn with an LLM timing and nothing after it, and
    that is a real record worth keeping rather than an incomplete one worth rejecting —
    interrupted turns are the ones most worth counting.
    """

    model_config = ConfigDict(extra="forbid")

    epoch: Annotated[int, Field(ge=0)]
    heard: str = ""
    said: str = ""
    transcribed: bool = False
    llm_ttft_ms: float | None = Field(default=None, ge=0)
    tts_first_audio_ms: float | None = Field(default=None, ge=0)
    first_frame_ms: float | None = Field(default=None, ge=0)
    perceived_total_ms: float | None = Field(default=None, ge=0)
    interrupted: bool = False
    silent: bool = False


class SessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str | None = None
    candidate_id: str | None = None
    """
    Who is being interviewed. Optional, and legitimately so.

    A session with no candidate is a real state -- every smoke test, every scripted demo, and
    every interview recorded before candidates existed. The usual way to set this is
    `POST /candidates/{id}/interview`, which also moves the candidate to `invited`; this field
    exists so a session can still be minted directly without that lifecycle step.
    """


def _not_found(session_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"no session {session_id!r}"
    )


def _load(session_id: str) -> dict[str, Any]:
    try:
        return store.get(COLLECTION, session_id)
    except NotFound as exc:
        raise _not_found(session_id) from exc


@router.get("")
async def list_sessions() -> list[dict[str, Any]]:
    return store.list(COLLECTION)


def new_session(agent_id: str | None, candidate_id: str | None = None) -> dict[str, Any]:
    """
    The initial shape of a session record. The only definition of it.

    Exported because `/candidates/{id}/interview` mints sessions too, and an earlier version of
    it built the record by hand with only the two ids -- so an invited interview arrived with no
    `turns`, no `started_at` and no counters, while a directly-created one had all five. Nothing
    errored; the console read the missing keys positionally and rendered `undefined` where a
    number belonged, which is the same defect this file already guards against by initialising
    every timing on a turn. Two writers of one shape will always drift, so there is one writer.
    """
    return {
        "agent_id": agent_id,
        "candidate_id": candidate_id,
        "started_at": now_iso(),
        "ended_at": None,
        "turns": [],
        "stale_dropped": 0,
        "frames_repeated": 0,
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session(body: SessionCreate) -> dict[str, Any]:
    """Called by the runtime when a candidate connects, not by an operator."""
    if body.candidate_id:
        # Rejected here rather than at session start. The file store has no foreign keys, so an
        # unknown id would otherwise produce an interview with no briefing and no error
        # anywhere.
        try:
            store.get("candidates", body.candidate_id)
        except NotFound:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"no candidate {body.candidate_id!r}",
            ) from None
    return store.create(COLLECTION, ID_PREFIX, new_session(body.agent_id, body.candidate_id))


@router.get("/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    return _load(session_id)


class Attendance(BaseModel):
    """
    What the person joining says about themselves, before the interview starts.

    `confirmed_name` is what they typed, not what we expected. Storing the expected name would
    make a mismatch invisible, and a mismatch is the only interesting thing this endpoint can
    record.
    """

    model_config = ConfigDict(extra="forbid")

    confirmed_name: Annotated[str, Field(min_length=1, max_length=160)]
    consented_to_recording: bool = False
    user_agent: Annotated[str, Field(max_length=400)] = ""
    timezone: Annotated[str, Field(max_length=80)] = ""


@router.post("/{session_id}/attendance")
async def record_attendance(session_id: str, body: Attendance) -> dict[str, Any]:
    """
    Record who says they are sitting this interview. **This does not verify identity.**

    **What this is, stated precisely, because the wrong reading of it is dangerous.** There is
    no authentication anywhere in this system; the interview link is the whole credential. So
    nothing here can prove a person is who they claim. What it does is capture an explicit,
    timestamped attestation — a name they typed, a recording consent, the browser and timezone
    they joined from — and put that on the record, so "who sat this interview" has a documented
    answer rather than an assumed one.

    That distinction is carried through to the report, which says "attested, not verified".
    Labelling this as identity verification would be worse than not having it: a hiring decision
    made partly on the belief that identity was checked, when it was not, is a specific and
    foreseeable harm.

    **Why the typed name is stored rather than compared.** A mismatch is the only signal
    available here, and it is only visible if both names survive. If this endpoint compared and
    stored a boolean, a candidate typing a colleague's name would be indistinguishable from a
    typo, and a reviewer would have nothing to look at. So it records, the report shows both,
    and a human decides whether the difference matters.

    Re-joining overwrites, because the last attestation describes the session that actually
    happened. Every attestation is kept in `history` though, since several joins under different
    names is itself worth seeing.
    """
    record = _load(session_id)
    if record.get("ended_at"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"session {session_id!r} ended at {record['ended_at']}",
        )

    expected = ""
    candidate_id = str(record.get("candidate_id") or "")
    if candidate_id:
        try:
            expected = str(store.get("candidates", candidate_id).get("name") or "")
        except NotFound:
            expected = ""

    typed = body.confirmed_name.strip()
    entry = {
        "confirmed_name": typed,
        "expected_name": expected,
        # Compared here so the console does not have to reimplement the comparison, but both
        # names are kept above so a reviewer can judge for themselves. Case- and
        # space-insensitive: a candidate typing "aparna rao" is not a discrepancy worth flagging
        # to a human.
        "matches_expected": bool(expected) and typed.casefold() == expected.casefold(),
        "consented_to_recording": body.consented_to_recording,
        "user_agent": body.user_agent[:400],
        "timezone": body.timezone,
        "attested_at": now_iso(),
        # Always false, and present rather than omitted so a reader of the record does not have
        # to infer that identity was unverified from the absence of a field. There is no
        # mechanism in this system that could set it true.
        "verified": False,
    }
    previous = record.get("attendance") or {}
    history = list(previous.get("history") or [])
    if previous.get("attested_at"):
        history.append({k: v for k, v in previous.items() if k != "history"})

    return store.update(
        COLLECTION, session_id, {"attendance": {**entry, "history": history}}
    )


@router.post("/{session_id}/turns", status_code=status.HTTP_201_CREATED)
async def append_turn(session_id: str, body: Turn) -> dict[str, Any]:
    """
    Append one turn. Appending to an ended session is rejected.

    Otherwise a late-arriving turn from a socket that has already closed would land after
    `ended_at`, and the record would claim a conversation continued past its own end.
    """
    record = _load(session_id)
    if record.get("ended_at"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"session {session_id!r} ended at {record['ended_at']}",
        )
    turns = list(record.get("turns", []))
    turns.append(body.model_dump())
    return store.update(COLLECTION, session_id, {"turns": turns})


@router.get("/{session_id}/rtc")
async def rtc_credentials(session_id: str) -> dict[str, Any]:
    """
    A LiveKit URL and join token for the candidate in this session's room.

    The room name is derived from the session id rather than stored, so there is one room per
    session by construction and no way for two sessions to collide in one — which would put two
    candidates in the same interview.

    Minted per request rather than at session creation, so a token's lifetime is tied to opening
    the page and not to how long ago the link was sent. The link itself is still not a
    credential; this endpoint is where that would be fixed, by requiring one.

    Returns `available: false` rather than a 500 when LiveKit is not configured. WebRTC is
    opt-in and a clean clone has no SFU, so the client must be able to fall back to the
    WebSocket transport rather than treat it as an outage.

    **The room is created here, before the token is minted, and that ordering is the whole
    point.** LiveKit auto-creates a room for whoever arrives first, and an auto-created room
    carries no egress configuration — so recording depends on the room existing *with* its
    config before anybody joins. `transport/livekit.py` also calls `ensure_room`, and for a
    while that was the only call, which was wrong in a way that produced no error: the
    candidate's browser fetches this token and connects immediately, while the agent's transport
    is still starting, so the browser won a race by about a second and the SFU logged
    `CreateRoom` on a room that already existed. Every interview reported `requested` and no
    file was ever written.

    This endpoint is the correct gate because it is the last thing that happens before a
    participant can join: no token without a room. The transport's call stays, idempotent, for
    the case where no browser ever asks — a headless or socket-only session.
    """
    _load(session_id)
    try:
        from avatar.transport.livekit import AGENT_IDENTITY, credentials, room_token

        url, _, _ = credentials()
    except Exception as exc:
        return {"available": False, "detail": str(exc)}

    room = f"session-{session_id}"

    from avatar.transport.recording import ensure_room

    # Awaited, not backgrounded: the token in this response admits the bearer to the room, so
    # returning before the room exists would reopen the race this call exists to close.
    recording = await ensure_room(room)
    with contextlib.suppress(NotFound):  # _load above already proved the session exists
        store.update(COLLECTION, session_id, {"recording": recording})

    return {
        "available": True,
        "url": url,
        "room": room,
        "token": room_token(room, f"candidate-{session_id}", name="Candidate"),
        # The constant, not the string. Whoever publishes the avatar must claim exactly this,
        # and under worker delivery that is a different process from this one -- so a literal
        # here and a literal there is two chances to disagree about the one value both sides
        # need to share.
        "agent_identity": AGENT_IDENTITY,
        # Reported back so the client knows whether it is being recorded from the same response
        # that lets it join, rather than having to ask separately.
        "recording": recording,
    }


# **There is deliberately no DELETE here**, and an earlier version of this file added one before
# `test_there_is_no_way_to_edit_or_delete_a_session` explained why: immutability is the property
# that makes these records evidence, and that test asserts against the route table rather than
# for a 405 precisely so the absence reads as a decision instead of an oversight.
#
# The need that motivated the attempt was real -- "start from a clean install" was impossible
# without psql, because demo and smoke-test sessions accumulated forever. That is a development
# operation, not a product one, and it belongs in `scripts/seed_demo.py --reset`, which goes to
# the store directly and says so. Adding a product endpoint to serve a maintenance need would
# have traded the evidence property for a convenience available another way.


def _advance_candidate(record: dict[str, Any]) -> None:
    """
    Move the candidate to `interviewed` when one of their interviews finishes.

    Here rather than in the candidates router because this is the moment the system can
    *observe* the transition; asking an operator to mark it by hand would make the status a
    description of who remembered to click rather than of what happened. Never advances
    backwards -- a candidate already `reviewed` stays reviewed if they sit a second interview,
    since a human has looked at them and that fact is not undone by more evidence arriving.
    """
    candidate_id = str(record.get("candidate_id") or "")
    if not candidate_id:
        return
    try:
        candidate = store.get("candidates", candidate_id)
        if str(candidate.get("status") or "") in ("new", "invited"):
            store.update("candidates", candidate_id, {"status": "interviewed"})
    except NotFound:
        # The candidate was deleted while their interview ran. The session survives by design;
        # there is simply no status left to advance.
        return
    except Exception as exc:  # pragma: no cover - bookkeeping must not fail ending a session
        print(f"sessions: could not advance candidate {candidate_id}: {exc}", flush=True)


@router.post("/{session_id}/end")
async def end_session(session_id: str, background: BackgroundTasks) -> dict[str, Any]:
    """
    Mark a session finished, then score it in the background. Idempotent on both counts.

    A socket can close twice — a client disconnect racing a server shutdown — and the second
    call must not move the timestamp, because session duration is derived from it. The early
    return also stops a second close from re-scoring, which would spend a model call to
    overwrite an identical scorecard.

    Scoring is queued rather than awaited, and that is the whole architectural point: the
    response to "the interview is over" must not wait on six model calls. The scorecard appears
    on the record when it is ready, and `GET /sessions/{id}` reports `scoring` as `pending`
    until then, so a reader can tell "not scored yet" from "could not be scored".
    """
    record = _load(session_id)
    if record.get("ended_at"):
        return record
    _advance_candidate(record)
    updated = store.update(
        COLLECTION, session_id, {"ended_at": now_iso(), "scoring": {"status": "pending"}}
    )
    background.add_task(run_scoring, session_id)
    return updated


@router.post("/{session_id}/score")
async def score_now(session_id: str, background: BackgroundTasks) -> dict[str, Any]:
    """
    Re-score a session on demand.

    Exists for the cases the automatic pass cannot cover: a rubric corrected after the fact, a
    scorecard that came back `unavailable` because credentials were missing at the time, or a
    judge model changed. Re-scoring overwrites, deliberately — a history of scorecards would
    invite picking the flattering one, and the turns it was derived from are append-only and
    unchanged, so any scorecard can be reproduced from them.

    Does not require the session to have ended. Scoring a live interview mid-way is a legitimate
    thing for an operator to do — it is how you check the rubric is working before running
    twenty candidates through it.
    """
    _load(session_id)
    store.update(COLLECTION, session_id, {"scoring": {"status": "pending"}})
    background.add_task(run_scoring, session_id)
    return {"session_id": session_id, "scoring": {"status": "pending"}}


async def run_scoring(session_id: str) -> None:
    """
    Assess one session and write the scorecard onto its record.

    Every failure is written rather than raised. This runs detached from any request, so an
    exception here would land in a server log nobody is reading and the record would sit on
    `pending` for ever — indistinguishable from a scorer that is still working. A stored reason
    is the only form of this failure anyone will ever see.

    Imports are local to keep the scoring path out of the module import graph for a process that
    only serves CRUD, and to keep this router free of a dependency on the runtime.
    """
    from avatar.agent_config import resolve_agent
    from avatar.judge_openai import build_judge
    from avatar.plan import InterviewPlan
    from avatar.scoring import Scorecard, score_session

    try:
        record = store.get(COLLECTION, session_id)
    except NotFound:
        return  # deleted between queueing and running; nothing to write to

    try:
        agent_id = record.get("agent_id")
        plan = resolve_agent(str(agent_id)).plan if agent_id else InterviewPlan()
        built = build_judge()
        judge, model = built if built else (None, "")
        card = await score_session(record, plan, judge, model=model)  # type: ignore[arg-type]
    except Exception as exc:
        card = Scorecard(
            status="unavailable",
            reason=f"scoring failed: {type(exc).__name__}: {exc}",
        )

    try:
        store.update(COLLECTION, session_id, {"scoring": card.as_dict()})
    except Exception:  # pragma: no cover - a failed write must not crash a background task
        return
