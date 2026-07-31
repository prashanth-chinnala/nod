"""
Voices — a recording of someone speaking, which an agent can be given to sound like.

**Its own resource rather than a field on the agent**, for the same reason faces, rubrics,
guardrails and lexicons are: one recording is reused across agents, and the console needs a
place to list, audition and delete them. It also keeps the two halves of a persona independent —
a face can be swapped without re-uploading a voice, and the reverse.

**No preparation step, unlike faces.** Cloning is zero-shot: the reference is encoded into a
speaker embedding at first use, cached in-process, and there is no artifact to build offline. So
a voice is usable the moment it validates, and `status` exists only to carry a failure. That
asymmetry is worth noticing rather than smoothing over — it is why this file is much shorter
than `faces.py`.

**What the audition endpoint is for.** A voice that sounds wrong should be discovered before a
candidate hears it, and the only way to know is to listen. `POST /voices/{id}/audition`
synthesises a sentence with the real engine and returns the WAV, so the console can play it.
That costs a model load on a cold process, which is the honest price of finding out early.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, Field

from avatar.store import NotFound, store

router = APIRouter(prefix="/voices", tags=["voices"])

COLLECTION = "voices"
ID_PREFIX = "voice"

AUDITION_TEXT = (
    "Thanks for making the time today. Could you start by telling me about a system you built "
    "that you are proud of?"
)
"""
What the audition speaks.

A real interviewer's opening rather than a pangram: the point is to judge whether this voice can
conduct an interview, and a sentence with interview cadence answers that where "the quick brown
fox" does not.
"""


class VoicePatch(BaseModel):
    """Only the fields an operator may change. `reference_path` is not one of them."""

    name: str | None = Field(default=None, min_length=1, max_length=120)

    model_config = {"extra": "forbid"}


def _not_found(voice_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"no voice {voice_id!r}"
    )


@router.get("")
def list_voices() -> list[dict[str, Any]]:
    return store.list(COLLECTION)


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_voice(
    name: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
) -> dict[str, Any]:
    """
    Create a voice from an uploaded recording.

    Validated by reading it with ffprobe, not by trusting the extension or the browser's content
    type — both are claims the uploader makes. `.webm` in particular is a container that may
    hold
    either audio or video, and a browser's `MediaRecorder` produces audio-only webm, so only the
    probe can say which this is.

    Returns the record plus `warnings`: a sample that is short, quiet or multi-channel will
    clone
    into a voice that works and disappoints, which is a different answer from a refusal and
    belongs
    in front of whoever chose the file.
    """
    from avatar import media

    raw = await file.read()
    try:
        stored = media.store_upload(raw, file.filename or "")
    except media.MediaRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        found = media.probe(stored)
        if found.kind != "audio":
            raise media.MediaRejected(
                f"this is {found.kind}, not audio. A voice needs a recording of someone "
                "speaking — upload video on the Faces screen instead."
            )
        warnings = media.check(found)
    except media.MediaRejected as exc:
        # Rejected uploads do not stay on the disk. Leaving them behind is how a media directory
        # becomes unattributable junk nobody dares delete.
        stored.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    record = store.create(
        COLLECTION,
        ID_PREFIX,
        {
            "name": name.strip() or file.filename or "untitled",
            "reference_path": str(stored),
            "duration_seconds": found.duration,
            "sample_rate": found.sample_rate,
            "channels": found.channels,
            # `ready` immediately: there is no enrollment to wait for. The field exists so a
            # voice
            # whose reference later goes missing can be marked, not to model a queue.
            "status": "ready",
            "failure_reason": None,
        },
    )
    return {**record, "warnings": warnings}


@router.get("/{voice_id}")
def read_voice(voice_id: str) -> dict[str, Any]:
    try:
        return store.get(COLLECTION, voice_id)
    except NotFound as exc:
        raise _not_found(voice_id) from exc


@router.patch("/{voice_id}")
def update_voice(voice_id: str, patch: VoicePatch) -> dict[str, Any]:
    try:
        fields = patch.model_dump(exclude_unset=True)
        if not fields:
            return store.get(COLLECTION, voice_id)
        return store.update(COLLECTION, voice_id, fields)
    except NotFound as exc:
        raise _not_found(voice_id) from exc


@router.delete("/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_voice(voice_id: str) -> None:
    """
    Delete the record and the recording it points at.

    The file goes too, deliberately. A voice is a recording of a real person, so leaving it on
    disk
    after the operator asked for it to be removed is the wrong default — and `faces` not doing
    this
    yet is a known gap rather than the pattern to copy.
    """
    from pathlib import Path

    try:
        record = store.get(COLLECTION, voice_id)
    except NotFound as exc:
        raise _not_found(voice_id) from exc

    for key in ("reference_path",):
        path = record.get(key)
        if path:
            source = Path(str(path))
            source.unlink(missing_ok=True)
            # The trimmed conditioning copy `tts_clone` writes beside it.
            for sibling in source.parent.glob(f"{source.stem}-ref*s.wav"):
                sibling.unlink(missing_ok=True)

    store.delete(COLLECTION, voice_id)


@router.post("/{voice_id}/audition")
async def audition_voice(voice_id: str) -> Response:
    """
    Synthesise one sentence in this voice and return it as a WAV the console can play.

    A thin proxy to the voice sidecar, which owns the model. That indirection is not incidental:
    Chatterbox cannot be installed alongside MuseTalk -- it needs a newer `transformers` than
    the
    renderer pins, and trying it downgraded torch and broke CUDA -- so nothing in this process
    imports it. See `avatar.audio.tts_clone`.

    `async def` is correct here precisely *because* of that: the work happens in another process
    and
    this handler only awaits a socket. The equivalent in-process version had to be sync to stay
    off
    the event loop.

    Failures come back as 422 with the sidecar's own message. Every realistic cause -- the
    service
    not running, no GPU, a reference deleted from under the record -- is something the operator
    can
    act on, and a traceback helps nobody looking at the console.
    """
    import httpx

    from avatar.audio.tts_clone import DEFAULT_SERVICE, SERVICE_ENV

    try:
        record = store.get(COLLECTION, voice_id)
    except NotFound as exc:
        raise _not_found(voice_id) from exc

    reference = str(record.get("reference_path") or "")
    if not reference:
        raise HTTPException(status_code=422, detail="this voice has no reference recording")

    service = os.environ.get(SERVICE_ENV, DEFAULT_SERVICE).rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{service}/audition",
                json={"text": AUDITION_TEXT, "reference_path": reference},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"the voice service at {service} is unreachable: {exc}. Start it with "
                "`.venv-voice/bin/python scripts/voice_service.py`."
            ),
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=422,
            detail=f"the voice service refused this ({response.status_code}): "
            f"{response.text[:200]}",
        )
    return Response(content=response.content, media_type="audio/wav")
