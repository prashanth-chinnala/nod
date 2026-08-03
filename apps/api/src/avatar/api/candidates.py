"""
Candidates — the person on the other side of the interview, and the resume that briefs it.

**What this closes.** A session named an agent and nothing else, so the system could say which
interviewer conducted an interview and not who sat in it. That is enough to prove the mechanics
and not enough to hire with: no comparing two people, no giving one person two interviews with
different interviewers, and nothing to tell the interviewer about who they are talking to.

**Why the resume changes the interview rather than only the record.** A resume is the only cheap
source of what is worth probing about a specific person. `resolve_for_session` appends a
briefing to the agent's system prompt when a session names a candidate — so the same agent asks
a data engineer about late-arriving events and a backend engineer about ordering guarantees,
without an operator configuring anything. The framing that goes with it matters more than the
text: see `avatar.resume.briefing`, which labels every line as an unverified claim precisely so
the interviewer probes it instead of reciting it.

**Why creating an interview lives here and not on `/sessions`.** `POST
/candidates/{id}/interview` picks an agent, mints the session, moves the candidate to `invited`
and returns the link. Doing it from the sessions router would work and would put the candidate's
lifecycle in two places; the status transition and the session creation are one operation and
belong together.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field

from avatar.store import NotFound, store

router = APIRouter(prefix="/candidates", tags=["candidates"])

COLLECTION = "candidates"
ID_PREFIX = "cand"

STATUSES = ("new", "invited", "interviewed", "reviewed")
"""
The lifecycle, advanced by the API rather than typed by an operator.

`new` on create, `invited` when an interview is minted, `interviewed` when one of their sessions
ends, `reviewed` when a human has looked at the report. Only the last is set by hand, because it
is the only one that describes a human action the system cannot observe.
"""


class CandidateCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=160)]
    email: Annotated[str, Field(max_length=254)] = ""
    role: Annotated[str, Field(max_length=160)] = ""
    notes: Annotated[str, Field(max_length=2000)] = ""
    agent_id: str | None = None
    """The interviewer an operator intends. Not binding — `/interview` may override it."""


class CandidatePatch(BaseModel):
    """
    What an operator may change after the fact.

    `resume_text` is absent on purpose: it is derived from the uploaded file, and letting it be
    patched independently would make the stored briefing disagree with the stored document with
    no way to tell which is right. Re-upload to change it.
    """

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1, max_length=160)] | None = None
    email: Annotated[str, Field(max_length=254)] | None = None
    role: Annotated[str, Field(max_length=160)] | None = None
    notes: Annotated[str, Field(max_length=2000)] | None = None
    agent_id: str | None = None
    status: str | None = None


class InterviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str | None = None
    """Falls back to the candidate's own `agent_id`. One of the two must be set."""


def _not_found(candidate_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"no candidate {candidate_id!r}"
    )


def _check_agent(agent_id: str | None) -> None:
    """
    Reject an unknown agent id at the boundary rather than at session start.

    The file store has no foreign keys, so without this an operator can point a candidate at an
    agent that does not exist and only find out when the candidate opens the link and the
    runtime falls back to its environment default — an interview conducted by the wrong
    interviewer, with no error anywhere.
    """
    if not agent_id:
        return
    try:
        store.get("agents", agent_id)
    except NotFound:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"no agent {agent_id!r}, so an interview with it could not be conducted",
        ) from None


@router.get("")
async def list_candidates() -> list[dict[str, Any]]:
    return store.list(COLLECTION)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_candidate(body: CandidateCreate) -> dict[str, Any]:
    _check_agent(body.agent_id)
    return store.create(
        COLLECTION,
        ID_PREFIX,
        {
            "name": body.name.strip(),
            "email": body.email.strip(),
            "role": body.role.strip(),
            "notes": body.notes.strip(),
            "agent_id": body.agent_id,
            "status": "new",
            # Every resume field initialised, including the ones that stay empty. The console
            # reads these positionally and an absent key renders as "undefined" where a dash
            # belongs.
            "resume_filename": None,
            "resume_path": None,
            "resume_text": None,
            "resume_chars": None,
            "resume_pages": None,
            "resume_truncated": False,
            "resume_error": None,
        },
    )


@router.get("/{candidate_id}")
async def get_candidate(candidate_id: str) -> dict[str, Any]:
    try:
        return store.get(COLLECTION, candidate_id)
    except NotFound as exc:
        raise _not_found(candidate_id) from exc


@router.patch("/{candidate_id}")
async def patch_candidate(candidate_id: str, body: CandidatePatch) -> dict[str, Any]:
    patch = body.model_dump(exclude_unset=True)
    if "status" in patch and patch["status"] not in STATUSES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"status must be one of {', '.join(STATUSES)}",
        )
    if "agent_id" in patch:
        _check_agent(patch["agent_id"])
    for key in ("name", "email", "role", "notes"):
        if key in patch and isinstance(patch[key], str):
            patch[key] = patch[key].strip()
    try:
        return store.update(COLLECTION, candidate_id, patch)
    except NotFound as exc:
        raise _not_found(candidate_id) from exc


@router.delete("/{candidate_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_candidate(candidate_id: str) -> None:
    """
    Remove the candidate and their resume file. Their interviews survive.

    The resume goes because it is a real person's document and a delete that leaves it on disk
    did not do what it says. The sessions stay, with `candidate_id` set to null by the schema: a
    transcript is evidence of something that happened, and a hiring process that destroys its
    own evidence when a row is tidied is worse than one that keeps an orphan.
    """
    from avatar import media

    try:
        record = store.get(COLLECTION, candidate_id)
    except NotFound as exc:
        raise _not_found(candidate_id) from exc

    raw = record.get("resume_path")
    if raw:
        root = media.MEDIA_ROOT.resolve()
        path = Path(str(raw)).resolve()
        # Confined to the media directory, for the same reason face deletion is: "unlink
        # whatever path this row holds" is a sentence worth never writing literally.
        if path.is_relative_to(root):
            path.unlink(missing_ok=True)
        else:
            print(
                f"candidates: refusing to delete {path} for {candidate_id}: outside {root}",
                flush=True,
            )
    store.delete(COLLECTION, candidate_id)


@router.post("/{candidate_id}/resume")
async def upload_resume(
    candidate_id: str,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT or MD")],
) -> dict[str, Any]:
    """
    Attach a resume and extract its text now, not at session start.

    Extraction happens here so that a bad parse is visible in the console before anyone is
    interviewed against it. Doing it lazily would put a PDF parse inside the session-start
    latency budget and would surface a scanned-image resume as a strange first question rather
    than as an error next to an upload button.

    A failed extraction still keeps the file and records the reason. The operator uploaded
    something real; throwing it away because this process could not read it would lose the only
    copy and tell them nothing.
    """
    from avatar import media, resume

    try:
        record = store.get(COLLECTION, candidate_id)
    except NotFound as exc:
        raise _not_found(candidate_id) from exc

    raw = await file.read()
    try:
        stored = resume.store_resume(raw, file.filename or "resume.pdf")
    except resume.ResumeUnreadable as exc:
        # Rejected before anything is written: an unsupported suffix, an empty body, or a file
        # over the resume ceiling. Distinct from a file that stored fine and would not parse,
        # which is recorded on the record below rather than refused.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from None

    # The previous resume, if any, goes now that a new one has landed. Keeping both would leave
    # a file no row points at, which is how a media directory becomes unattributable.
    previous = record.get("resume_path")
    if previous and Path(str(previous)) != stored:
        old = Path(str(previous)).resolve()
        if old.is_relative_to(media.MEDIA_ROOT.resolve()):
            old.unlink(missing_ok=True)

    patch: dict[str, Any] = {
        "resume_filename": file.filename or stored.name,
        "resume_path": str(stored),
    }
    try:
        extracted = resume.extract(raw, file.filename or stored.name)
    except resume.ResumeUnreadable as exc:
        patch |= {
            "resume_text": None,
            "resume_chars": 0,
            "resume_pages": None,
            "resume_truncated": False,
            "resume_error": str(exc),
        }
        updated = store.update(COLLECTION, candidate_id, patch)
        # 200, not an error status: the upload succeeded and the file is kept. The failure is a
        # property of the record now, which the console shows next to the file.
        return updated

    patch |= {
        "resume_text": extracted.text,
        "resume_chars": extracted.chars,
        "resume_pages": extracted.pages,
        "resume_truncated": extracted.truncated,
        "resume_error": None,
    }
    return store.update(COLLECTION, candidate_id, patch)


@router.delete("/{candidate_id}/resume")
async def remove_resume(candidate_id: str) -> dict[str, Any]:
    """Detach and delete the resume, leaving the candidate."""
    from avatar import media

    try:
        record = store.get(COLLECTION, candidate_id)
    except NotFound as exc:
        raise _not_found(candidate_id) from exc

    raw = record.get("resume_path")
    if raw:
        path = Path(str(raw)).resolve()
        if path.is_relative_to(media.MEDIA_ROOT.resolve()):
            path.unlink(missing_ok=True)
    return store.update(
        COLLECTION,
        candidate_id,
        {
            "resume_filename": None,
            "resume_path": None,
            "resume_text": None,
            "resume_chars": None,
            "resume_pages": None,
            "resume_truncated": False,
            "resume_error": None,
        },
    )


@router.get("/{candidate_id}/sessions")
async def candidate_sessions(candidate_id: str) -> list[dict[str, Any]]:
    """
    Every interview this candidate has sat, newest first.

    Filtered here rather than in the console because the console would have to fetch every
    session in the system to do it, which is fine at forty rows and wrong at forty thousand.
    """
    try:
        store.get(COLLECTION, candidate_id)
    except NotFound as exc:
        raise _not_found(candidate_id) from exc

    sessions = [
        record
        for record in store.list("sessions")
        if str(record.get("candidate_id") or "") == candidate_id
    ]
    sessions.sort(key=lambda record: str(record.get("created_at") or ""), reverse=True)
    return sessions


@router.post("/{candidate_id}/interview", status_code=status.HTTP_201_CREATED)
async def create_interview(candidate_id: str, body: InterviewRequest) -> dict[str, Any]:
    """
    Mint an interview for this candidate and return the link to send them.

    One operation, three effects: a session bound to both the candidate and the agent, the
    candidate moved to `invited`, and the chosen agent remembered so the next interview defaults
    to the same interviewer. Splitting these across two calls would let a session exist for a
    candidate still marked `new`, which then reads as a bug in the list view.
    """
    try:
        candidate = store.get(COLLECTION, candidate_id)
    except NotFound as exc:
        raise _not_found(candidate_id) from exc

    agent_id = body.agent_id or candidate.get("agent_id")
    if not agent_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="no agent to interview with: pass agent_id, or set one on the candidate",
        )
    _check_agent(str(agent_id))

    # Built by `sessions.new_session`, not here. Constructing the record inline is what left an
    # invited interview without `turns` or `started_at` while a directly-created one had them.
    from avatar.api.sessions import new_session

    session = store.create("sessions", "sess", new_session(str(agent_id), candidate_id))
    store.update(
        COLLECTION,
        candidate_id,
        {"status": "invited", "agent_id": str(agent_id)},
    )
    return {
        "session_id": session["id"],
        "candidate_id": candidate_id,
        "agent_id": str(agent_id),
        # A path, not a URL: this process does not know the origin the console is served from,
        # and inventing one would produce a link that works on the developer's machine only.
        "interview_path": f"/interview/{session['id']}",
    }


@router.get("/{candidate_id}/resume/file")
async def download_resume(candidate_id: str) -> Any:
    """The original document, so an operator can check what the extractor was given."""
    from fastapi.responses import FileResponse

    from avatar import media

    try:
        record = store.get(COLLECTION, candidate_id)
    except NotFound as exc:
        raise _not_found(candidate_id) from exc

    raw = record.get("resume_path")
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="this candidate has no resume"
        )
    path = Path(str(raw)).resolve()
    if not path.is_relative_to(media.MEDIA_ROOT.resolve()) or not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="the resume file is missing from disk; re-upload it",
        )
    return FileResponse(
        path,
        filename=str(record.get("resume_filename") or path.name),
        media_type="application/octet-stream",
    )
