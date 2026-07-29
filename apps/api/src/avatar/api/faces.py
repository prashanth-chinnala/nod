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

import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from avatar.contracts import RendererConfig
from avatar.renderers import build
from avatar.store import NotFound, store

COLLECTION = "faces"
ID_PREFIX = "face"

router = APIRouter(prefix="/faces", tags=["faces"])

FaceStatus = Literal["queued", "preparing", "ready", "failed"]

PREPARE_RENDERER = "stub"
"""
Which renderer performs enrollment.

Pinned to the stub rather than read from `AVATAR_RENDERER`, and that is the honest state of this
milestone: no GPU exists yet, so the queue, the status transitions and the failure path have to
be buildable and testable against a renderer that needs nothing. `build` is the one-line swap —
when the real renderer runs, this constant is what changes, and every transition around it has
already been exercised.

The consequence, stated rather than hidden: an `enrollment_ms` recorded today is the cost of the
stub's no-op, not of a real enrollment. It is a real measurement of the wrong thing.
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
    Hard delete, and deliberately not cascading.

    An agent may reference this face id. Rewriting those agents to keep the data tidy would
    change a configuration nobody asked to change; a dangling id that shows up as "missing face"
    at session start is the smaller and more visible problem.
    """
    try:
        store.delete(COLLECTION, face_id)
    except NotFound as exc:
        raise _not_found(face_id) from exc


@router.post("/{face_id}/prepare")
def prepare_face(face_id: str) -> dict[str, Any]:
    """
    Run enrollment: `queued`/`failed` → `preparing` → `ready` or `failed`.

    **Declared `def`, not `async def`, and that is load-bearing.** `prepare_identity` is
    synchronous and allowed to be slow. In an `async def` handler it would run on the event
    loop, which is the same loop serving live WebSocket sessions — one enrollment would stall
    every conversation in progress. A sync handler is dispatched to a threadpool instead, so the
    cost lands on a worker thread.

    **`preparing` is written before the renderer is touched** so that a process killed mid-
    enrollment leaves evidence of an attempt rather than a record that still claims to be
    queued. The cost of that choice, stated because it is real: nothing reaps a `preparing` row
    afterwards, so a crashed enrollment leaves a face that `PREPARABLE` will not accept again
    and has to be deleted. A reaper belongs with a real job queue, which this is not.
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
    store.update(COLLECTION, face_id, {"status": "preparing"})

    # Timed from before `build` on purpose: constructing the renderer is where a real one
    # loads its weights, and that cost is genuinely part of the first enrollment. Timing only
    # `prepare_identity` would report a number smaller than the wait the operator sat through.
    started = time.perf_counter()
    try:
        renderer = build(RendererConfig(name=PREPARE_RENDERER))
        identity = renderer.prepare_identity(str(record["reference_path"]))
    except Exception as exc:
        # Broad by intention. Everything a renderer can raise on operator-supplied media is
        # this endpoint's normal failure outcome, and the alternative — enumerating the
        # exception types of a model this project has not run yet — would be a guess that
        # turns into a 500 the first time it is wrong. `build` is inside the block too, so a
        # renderer that cannot even be constructed still leaves a diagnosable record instead
        # of a face stuck in `preparing`.
        return store.update(
            COLLECTION,
            face_id,
            {"status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"},
        )
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    # `failure_reason` is explicitly cleared, not left behind. It used to survive a later
    # successful run because the store dropped nulls, so a face could read `ready` while still
    # carrying the reason it failed two attempts ago -- harmless only for as long as every
    # reader remembered to check `status` first. Now that a null can be written, the record can
    # simply stop contradicting itself.
    return store.update(
        COLLECTION,
        face_id,
        {
            "status": "ready",
            "failure_reason": None,
            "enrollment_ms": elapsed_ms,
            "frame_count": _reported_frame_count(identity),
        },
    )
