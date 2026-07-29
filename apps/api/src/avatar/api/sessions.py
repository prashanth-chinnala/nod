"""
Conversation records: what was said, what it cost, and what got interrupted.

**Append-only, deliberately.** There is no PATCH and no DELETE. A conversation record that
can be edited is not evidence, and evidence is the entire reason this collection exists —
every latency figure in `PROCESS.md` traces back to observed turns, and a record that could
be revised after the fact would make those figures unciteable. Turns are appended; a session
is ended once.

**Why the per-turn shape mirrors the telemetry rather than the UI.** A turn stores the four
stage timings the runtime already emits (`llm_ttft_ms`, `tts_first_audio_ms`,
`first_frame_ms`, `perceived_total_ms`) plus `heard`, `said`, and whether the transcriber
produced anything. Storing a pre-computed "total" instead would lose the breakdown, and the
breakdown is the finding: none of the three dominant terms is the renderer.

**`transcribed` is stored separately from `heard` on purpose.** An empty transcript is not the
same as a silent candidate, and conflating them hides the failure that took a day to find —
the interviewer asking a plausible question that ignores the answer, because no words ever
reached it. A session where `transcribed` is false on every turn is a broken STT
configuration, and that must be visible in a list rather than needing a log grep.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, status
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


class SessionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str | None = None


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


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_session(body: SessionCreate) -> dict[str, Any]:
    """Called by the runtime when a candidate connects, not by an operator."""
    return store.create(
        COLLECTION,
        ID_PREFIX,
        {
            "agent_id": body.agent_id,
            "started_at": now_iso(),
            "ended_at": None,
            "turns": [],
            "stale_dropped": 0,
            "frames_repeated": 0,
        },
    )


@router.get("/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    return _load(session_id)


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
    """
    _load(session_id)
    try:
        from avatar.transport.livekit import credentials, room_token

        url, _, _ = credentials()
    except Exception as exc:
        return {"available": False, "detail": str(exc)}

    room = f"session-{session_id}"
    return {
        "available": True,
        "url": url,
        "room": room,
        "token": room_token(room, f"candidate-{session_id}", name="Candidate"),
        "agent_identity": "avatar-agent",
    }


@router.post("/{session_id}/end")
async def end_session(session_id: str) -> dict[str, Any]:
    """
    Mark a session finished. Idempotent: the first `ended_at` is the one that stands.

    A socket can close twice — a client disconnect racing a server shutdown — and the second
    call must not move the timestamp, because session duration is derived from it.
    """
    record = _load(session_id)
    if record.get("ended_at"):
        return record
    return store.update(COLLECTION, session_id, {"ended_at": now_iso()})
