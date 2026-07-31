"""
CRUD for the Face — a reference clip or image, plus the prepared identity artifact made from it.

**Why enrollment is a status field and not a synchronous side effect of create.** Preparing an
identity is the one part of this system that is allowed to be slow: `TalkingHeadRenderer` says
so explicitly, because a real model crops and encodes every frame of the reference clip before a
session can use it. Doing that inside `POST /faces` would make the create call take however long
the clip takes, with no record on disk if it died half way. A face therefore exists first and is
enrolled second, and `status` is what says which of those has happened.

**Why a failed prepare is a 200 and not a 500.** A reference clip with no detectable face, an
unreadable file, a video the decoder refuses — these are the ordinary outcomes of pointing a
model at operator-supplied media, not server faults. Returning a 500 would put them in the error
log next to real bugs and leave the operator with nothing to look at; storing `status="failed"`
with the reason puts the outcome on the row that caused it. The only thing this endpoint treats
as a genuine fault is a face id that does not exist.

**Why `reference_path` cannot be patched.** `status`, `enrollment_ms` and `frame_count` are
findings about one specific clip. Re-pointing a prepared face at different media would leave a
measured number attached to something it was never measured from — and the store's merge drops
`None`, so those fields could not be cleared in the same write (`agents.py` documents the same
limitation). A new reference is a new face; only the label is editable.

**Why records go back to the client exactly as the store wrote them.** Same reason as
`agents.py`: a response model would filter every read through this file's idea of a Face, so a
field written by a newer build would vanish from the console while still sitting on disk. The
store is the authority on what a face is; this module is the authority on what may be written.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from avatar import jobs
from avatar.contracts import RendererConfig
from avatar.renderers import build
from avatar.store import NotFound, store

COLLECTION = "faces"
ID_PREFIX = "face"

router = APIRouter(prefix="/faces", tags=["faces"])

FaceStatus = Literal["queued", "preparing", "ready", "failed"]

PREPARE_RENDERER = os.environ.get("AVATAR_RENDERER", "stub")
"""
Which renderer performs enrollment: whichever one the server is configured to render with.

This was pinned to `"stub"` for the whole time no GPU existed, with a note that the constant
was the one-line swap once a real renderer ran. It has run, so this is that swap.

Reading `AVATAR_RENDERER` rather than taking a separate variable is the point. Enrolling with
one renderer and rendering with another produces an identity artifact of the wrong shape --
MuseTalk's is a dict of latents, masks and cycled frames; the stub's holds a path -- and the
mismatch would not surface at enrollment. It would surface as a failed session, after a
candidate had already joined.

The consequence for anything measured before this commit, stated rather than quietly
corrected: an `enrollment_ms` recorded under the stub is the cost of a no-op, not of an
enrollment, and those rows are still in the database. `frame_count` is the tell -- the stub
reports none.
"""

PREPARABLE: frozenset[str] = frozenset({"queued", "failed"})
"""
The statuses `POST /{id}/prepare` accepts.

`preparing` is excluded because a second run would race the first for the same record and the
loser's result would silently overwrite the winner's — a double-clicked button must not be able
to do that. `ready` is excluded because prepare overwrites a recorded measurement: refusing it
means the `enrollment_ms` on a ready face is always the number produced by the run that made it
ready. Re-enrolling is a delete and a create, which is cheap here.
"""


def _stripped(field_name: str) -> AfterValidator:
    """
    Trim, and refuse what is left if it is empty.

    `min_length=1` alone accepts a single space, which is worse than an empty value: the row
    renders blank and cannot be found by anyone scanning the list for it.
    """

    def validate(value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} must not be blank")
        return stripped

    return AfterValidator(validate)


FaceName = Annotated[str, Field(min_length=1), _stripped("name")]
"""Shared by create and update so the two paths cannot disagree about a legal name."""

ReferencePath = Annotated[str, Field(min_length=1), _stripped("reference_path")]
"""
A path the *server* resolves, not the browser.

Deliberately not checked for existence here. Whether a reference is usable is the renderer's
judgment — a real one may accept a directory of frames, or a URL — and a pre-flight
`Path.exists()` in the router would either duplicate that judgment or contradict it. An unusable
reference surfaces where it is discovered, as `status="failed"` with the reason.
"""


class FaceCreate(BaseModel):
    """
    What a client may send to create a face.

    `status`, `enrollment_ms`, `frame_count` and `failure_reason` are absent on purpose, and
    `extra="forbid"` makes sending them a 422 rather than a silent no-op. Every one of them is a
    finding produced by a prepare run; accepting them from a client would let a face be declared
    ready, with a latency figure, without anything having been measured. That is the one failure
    this project is least allowed to have.
    """

    model_config = ConfigDict(extra="forbid")

    name: FaceName
    reference_path: ReferencePath


class FaceUpdate(BaseModel):
    """
    A partial update. Only the label is editable — see the module docstring.

    `extra="forbid"` is what turns "patch `reference_path`" into a 422 that names the rule,
    instead of a write that quietly succeeds and leaves a measurement attached to media it did
    not come from.
    """

    model_config = ConfigDict(extra="forbid")

    name: FaceName | None = None


def _not_found(face_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"no face with id {face_id!r}"
    )


def _reported_frame_count(identity: object) -> int | None:
    """
    The frame count the identity artifact reports, or `None` if it does not report one.

    Read off the artifact rather than computed here, because counting frames means decoding the
    clip and the renderer has already done that. Neither renderer exposes it yet, so this
    returns `None` today: an operator sees an empty cell rather than a number nothing measured,
    which is the correct outcome under this repo's first standing rule.
    """
    value = getattr(identity, "frame_count", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


@router.get("")
async def list_faces() -> list[dict[str, Any]]:
    """Newest first, as the store orders them — the list view relies on that order."""
    return store.list(COLLECTION)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_face(body: FaceCreate) -> dict[str, Any]:
    """
    Create an unenrolled face.

    The four derived fields are written as explicit nulls rather than left absent, so that a
    reader — the console table, a later migration, a person running `cat` over the data
    directory — never has to distinguish "not measured" from "key missing".
    """
    return store.create(
        COLLECTION,
        ID_PREFIX,
        {
            **body.model_dump(),
            "status": "queued",
            "enrollment_ms": None,
            "frame_count": None,
            "failure_reason": None,
        },
    )


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_face(
    name: Annotated[str, Form()],
    file: Annotated[UploadFile, File()],
    animate: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    """
    Create a face from an uploaded video or image.

    Alongside `POST ""` rather than replacing it: that endpoint takes a path already on the
    server, which is how a scripted demo and the test suite create faces without moving bytes.
    This is the one a person uses.

    **The file is stored before it is validated, and validated by reading it.** ffprobe needs a
    file on disk, and every property that matters -- is it really video, how long, how big -- is
    a claim the uploader cannot be trusted for. A `.mp4` full of PDF passes every check based on
    the filename and then fails inside the renderer, hours later, as a preparation error nobody
    traces back to here. A rejected upload is deleted rather than left to accumulate.

    **An image is expanded into a short clip and the clip becomes the reference.** A photograph
    has no head motion for a reference-driven renderer to borrow, so keeping the two shapes
    separate would push a special case into `prepare_identity`, the frame cycling and the render
    windowing. The warning about it reaches the operator; what it cannot do is invent movement.

    Returns the record plus `warnings` -- things that will work and disappoint, which is a
    different answer from a refusal and belongs in front of whoever chose the file.
    """
    from avatar import media

    raw = await file.read()
    try:
        stored = media.store_upload(raw, file.filename or "")
    except media.MediaRejected as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        found = media.probe(stored)
        warnings = media.check(found)
        reference = media.clip_from_image(stored) if found.kind == "image" else stored
        thumb = media.thumbnail(reference)
    except media.MediaRejected as exc:
        # Nothing usable came of it, so it does not stay on the disk. Leaving rejected uploads
        # behind is how a media directory becomes unattributable junk nobody dares delete.
        stored.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # An image can be animated into a clip with real head motion instead of being expanded into
    # identical frames. Requested per upload rather than applied always: it costs minutes, needs
    # LivePortrait installed, and an operator who wants a deliberately static persona should be
    # able to have one.
    animation: dict[str, Any] = {}
    if animate and found.kind == "image":
        from avatar import animate as animator

        unavailable = animator.available()
        if unavailable:
            # Not fatal. The still already works, so refusing the whole upload because an
            # optional
            # enhancement is unavailable would be the wrong trade -- the operator gets the face
            # they uploaded plus the reason it did not move.
            warnings.append(f"could not animate this photograph: {unavailable}")
        else:
            animation = {"animate_requested": True}

    record = store.create(
        COLLECTION,
        ID_PREFIX,
        {
            "name": name.strip() or file.filename or "untitled",
            "reference_path": str(reference),
            # The original is kept even when a clip was generated from it. It is the only
            # lossless
            # copy of what the operator actually provided, and re-deriving the clip later --
            # at a
            # different length, or with a renderer that can animate a still -- needs it.
            "source_path": str(stored),
            "thumbnail_path": str(thumb),
            "source_kind": found.kind,
            "duration_seconds": found.duration,
            "width": found.width,
            "height": found.height,
            "status": "queued",
            "enrollment_ms": None,
            "frame_count": None,
            "failure_reason": None,
            "animated": False,
            "animation_ms": None,
            "job_started_at": None,
            **animation,
        },
    )
    return {**record, "warnings": warnings}


@router.get("/{face_id}/thumbnail")
async def face_thumbnail(face_id: str) -> FileResponse:
    """
    The preview frame, served as a file.

    Served by the runtime rather than the console because the file lives beside the reference on
    the runtime's disk, and copying it into the web app's public directory would make two copies
    with no way to tell which is current.
    """
    try:
        record = store.get(COLLECTION, face_id)
    except NotFound as exc:
        raise _not_found(face_id) from exc
    path = Path(str(record.get("thumbnail_path") or ""))
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "this face has no preview frame. Faces created from a server-side path do not "
                "get one; upload the reference to generate it."
            ),
        )
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{face_id}")
async def get_face(face_id: str) -> dict[str, Any]:
    try:
        return store.get(COLLECTION, face_id)
    except NotFound as exc:
        raise _not_found(face_id) from exc


@router.patch("/{face_id}")
async def update_face(face_id: str, body: FaceUpdate) -> dict[str, Any]:
    """
    Merge the keys that were sent.

    `exclude_unset` is what makes this a patch rather than a replace: without it an omitted
    `name` would arrive as `None` and overwrite the stored one with nothing. The store now
    applies nulls rather than dropping them, so that failure would be destructive instead of
    merely confusing — which makes `exclude_unset` load-bearing here rather than tidy.
    """
    patch = body.model_dump(exclude_unset=True)
    if not patch:
        # `updated_at` is displayed in the console, and it is the only evidence an operator
        # has about when a face last changed. Bumping it for a request that changes nothing
        # makes that column lie.
        # 422 spelled numerically: Starlette renamed its constant for this code, so either
        # name emits a deprecation warning on one of the versions in the supported range.
        raise HTTPException(
            status_code=422, detail="empty patch: send at least one field to change"
        )
    try:
        return store.update(COLLECTION, face_id, patch)
    except NotFound as exc:
        raise _not_found(face_id) from exc


@router.delete("/{face_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_face(face_id: str) -> None:
    """
    Hard delete: the record, every file it points at, and any identity prepared from it.

    **The media goes too, and that is the correction here.** A face is a photograph or a video
    of a real person. Leaving it on disk after the operator asked for it to be removed means the
    delete button quietly did not do what it says -- and it means the only record of whose face
    it is has just been deleted, so what remains is unattributable biometric data that nobody
    can safely clean up later. `voices` already did this; `faces` not doing it was a known gap.

    Four things exist per face and all four go: the reference the renderer reads, the original
    upload it was derived from, the thumbnail, and the LivePortrait output directory if the
    photograph was animated. The prepared identity in the renderer's cache goes with them.

    **Still deliberately not cascading to agents.** An agent may hold this face id. Rewriting
    those agents to keep the data tidy would change a configuration nobody asked to change, and
    a dangling id that surfaces as "missing face" at session start is the smaller, louder
    problem. Files are different from foreign keys: nothing reads a file by accident.
    """
    from avatar import media

    try:
        record = store.get(COLLECTION, face_id)
    except NotFound as exc:
        raise _not_found(face_id) from exc

    # Confined to the media directory before anything is unlinked. These paths come from a
    # record rather than from a request, so this is not the front line -- but "delete whatever
    # path this row contains" is a sentence worth never writing, and one hand-edited JSON file
    # is the whole distance between it and being literally true.
    root = media.MEDIA_ROOT.resolve()
    for key in ("reference_path", "source_path", "thumbnail_path"):
        raw = record.get(key)
        if not raw:
            continue
        path = Path(str(raw)).resolve()
        if not path.is_relative_to(root):
            print(f"faces: refusing to delete {path} for {face_id}: outside {root}", flush=True)
            continue
        path.unlink(missing_ok=True)
        # LivePortrait writes its frames into `<stem>-animated/` beside the source it animated,
        # and that directory is the largest thing on disk per face. Leaving it would make the
        # delete look like it worked while reclaiming almost none of the space.
        animated = path.parent / f"{path.stem}-animated"
        if animated.is_dir() and animated.resolve().is_relative_to(root):
            shutil.rmtree(animated, ignore_errors=True)

    # The prepared identity too, if this process holds one. It is about a gigabyte of that
    # person's frames and latents and would otherwise stay resident until restart -- with the
    # record that named whose face it was already gone.
    #
    # Imported here rather than at module scope: this module is part of the API surface and has
    # to stay importable with no renderer installed, and `musetalk` is a renderer module.
    try:
        from avatar.renderers.musetalk import forget_identity

        for key in ("reference_path", "source_path"):
            if record.get(key):
                forget_identity(str(record[key]))
    except ImportError:
        pass

    store.delete(COLLECTION, face_id)


@router.post("/{face_id}/prepare", status_code=status.HTTP_202_ACCEPTED)
def prepare_face(face_id: str) -> dict[str, Any]:
    """
    Start enrollment. Returns 202 immediately; poll the record for the outcome.

    **202 rather than a completed 200**, because the work takes minutes -- 126s for a 550-frame
    reference, and another ~124s when a photograph is animated first. Holding the connection
    open for
    that failed in three ways at once: a proxy or browser timed out and showed an error for work
    that
    was running fine, nothing reported progress, and a process killed midway left a row claiming
    `preparing` for ever, which `PREPARABLE` refuses -- so the face became permanently
    unenrollable
    and the only fix was deleting it. `avatar.jobs` handles all three, including reaping stale
    rows at
    startup.

    **`preparing` is claimed before the renderer is touched**, with a timestamp. The timestamp
    is what
    makes recovery possible: without it a stuck row is indistinguishable from a live one.

    Animation, when the upload asked for it, happens *before* enrollment and in the same job.
    Enrolling first would cache an identity built from the still and silently discard the motion
    just
    generated.
    """
    try:
        record = store.get(COLLECTION, face_id)
    except NotFound as exc:
        raise _not_found(face_id) from exc

    current = str(record.get("status", "queued"))
    if current not in PREPARABLE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"face {face_id!r} is {current!r}; prepare accepts "
                f"{sorted(PREPARABLE)}. Delete and recreate the face to re-enroll."
            ),
        )
    jobs.claim(store, COLLECTION, face_id)

    reference = str(record["reference_path"])
    wants_animation = bool(record.get("animate_requested")) and not record.get("animated")

    def enroll() -> dict[str, Any]:
        """
        Animate if asked, then enroll. Runs on a worker thread; raises to fail the job.

        Ordered this way because enrollment must see the final frames: preparing the still and
        then
        animating it would cache an identity built from a photograph and quietly ignore the
        motion
        that was just generated.
        """
        patch: dict[str, Any] = {}
        source = reference

        if wants_animation:
            from avatar import animate as animator

            produced = animator.animate(Path(str(record["source_path"] or reference)))
            source = str(produced.clip)
            patch |= {
                "reference_path": source,
                "animated": True,
                "animation_ms": produced.ms,
                "source_kind": "video",
                "duration_seconds": round(produced.seconds, 2),
            }

        # Timed from before `build` on purpose: constructing the renderer is where a real one
        # loads
        # its weights, and that cost is genuinely part of the first enrollment. Timing only
        # `prepare_identity` would report a number smaller than the wait the operator sat
        # through.
        started = time.perf_counter()
        renderer = build(RendererConfig(name=PREPARE_RENDERER))
        identity = renderer.prepare_identity(source)
        return patch | {
            "status": "ready",
            "enrollment_ms": round((time.perf_counter() - started) * 1000),
            "frame_count": _reported_frame_count(identity),
        }

    jobs.submit(store, COLLECTION, face_id, enroll, label=f"enroll-{face_id}")
    return store.get(COLLECTION, face_id)
