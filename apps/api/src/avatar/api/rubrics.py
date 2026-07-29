"""
CRUD for the Rubric — the competency plan an interview is run against.

**Why competency ids are derived, not supplied.** Coverage is keyed by id and stored on session
records, so an id has to outlive an edit. A positional id would be silently reassigned the
moment an operator dragged a competency up the list, and every past session's coverage would
then point at the wrong area — a data corruption with no error and no symptom until someone read
a report. Slugging the name gives an id that is stable across reordering and readable in a
report, and duplicates are rejected here because two competencies sharing a key would share
coverage.

**Why an empty `signals` list is allowed but warned about in the response.** A rubric written
before anyone has decided what evidence looks like is a legitimate draft, and refusing to save
it would push the operator to invent signals to satisfy a validator. But a competency with no
signals can never be evidenced — it will be probed `max_turns` times and reported as exhausted —
so the API says so rather than letting it be discovered from a confusing report weeks later.

**Why records go back exactly as the store wrote them.** Same reason as `agents.py`: a response
model would filter reads through this file's idea of a Rubric, so a field written by a newer
build would vanish from the console while still sitting on disk, and a partial deploy would look
like data loss.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from avatar.plan import DEFAULT_MAX_TURNS, DEFAULT_MIN_SIGNALS, slug
from avatar.store import NotFound, store

COLLECTION = "rubrics"
ID_PREFIX = "rubric"

router = APIRouter(prefix="/rubrics", tags=["rubrics"])


def _stripped_name(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("name must not be blank")
    return stripped


RubricName = Annotated[str, Field(min_length=1), AfterValidator(_stripped_name)]


class CompetencyIn(BaseModel):
    """
    One competency as an operator writes it.

    `probe` is deliberately prose and deliberately not a question. The runtime tells the
    model to probe this area and write its own sentence; storing a literal question would
    make the interview a script, and a script cannot follow up on an answer — which is the
    only reason a live interviewer beats a form.
    """

    model_config = ConfigDict(extra="forbid")

    name: RubricName
    probe: str = ""
    signals: list[str] = Field(default_factory=list)
    """
    Terms that count as the candidate having engaged with this area.

    Matched whole-word and case-insensitively by `avatar.knowledge.augment.terms_present`, which
    handles `C++`, `.NET` and `Node.js` correctly — the naive word-boundary regex does not, and
    those are exactly the signals an engineering rubric is written in.
    """
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1, le=20)
    min_signals: int = Field(default=DEFAULT_MIN_SIGNALS, ge=1, le=20)
    weight: float = Field(default=1.0, gt=0, le=10)
    """
    Contribution to the scorecard, not to the running order.

    `gt=0` rather than `ge=0`: a zero-weight competency would still be asked about, consuming
    turns from a fixed-length interview, and then count for nothing. An operator who wants that
    means to delete it.
    """

    @model_validator(mode="after")
    def _min_signals_must_be_reachable(self) -> CompetencyIn:
        """
        Refuse a bar the signal list cannot clear.

        `min_signals=3` against two signals is not a strict rubric, it is a competency that can
        never be evidenced — it will be probed to exhaustion every time and reported as a dead
        end. The operator meant something achievable, so this fails at the form rather than
        producing a confusing report. An empty signal list is exempt: that is a draft, handled
        below.
        """
        distinct = {signal.strip() for signal in self.signals if signal.strip()}
        if distinct and self.min_signals > len(distinct):
            raise ValueError(
                f"min_signals ({self.min_signals}) exceeds the number of distinct signals "
                f"({len(distinct)}) for {self.name!r}; this competency could never be evidenced"
            )
        return self


class RubricCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: RubricName
    description: str = ""
    competencies: list[CompetencyIn] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_must_be_unique(self) -> RubricCreate:
        """
        Two competencies whose names slug to the same id would share one coverage record.

        Caught here because the failure is invisible downstream: the second would appear in the
        console, never accumulate its own evidence, and inherit the first's status.
        """
        seen: dict[str, str] = {}
        for competency in self.competencies:
            key = slug(competency.name)
            if key in seen:
                raise ValueError(
                    f"{competency.name!r} and {seen[key]!r} both reduce to the id {key!r}; "
                    "they would share one coverage record. Rename one."
                )
            seen[key] = competency.name
        return self


class RubricUpdate(BaseModel):
    """
    A partial update. `competencies` is replaced whole rather than merged.

    Whole, because the uniqueness rule spans the list: merging one competency would let a rename
    collide with a sibling that was not sent, and the check would have to move somewhere it
    could be forgotten. Sending the list keeps `RubricCreate`'s validator the single authority.
    """

    model_config = ConfigDict(extra="forbid")

    name: RubricName | None = None
    description: str | None = None
    competencies: list[CompetencyIn] | None = None

    @model_validator(mode="after")
    def _ids_must_be_unique(self) -> RubricUpdate:
        if self.competencies is None:
            return self
        seen: dict[str, str] = {}
        for competency in self.competencies:
            key = slug(competency.name)
            if key in seen:
                raise ValueError(
                    f"{competency.name!r} and {seen[key]!r} both reduce to the id {key!r}; "
                    "they would share one coverage record. Rename one."
                )
            seen[key] = competency.name
        return self


def _not_found(rubric_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"no rubric with id {rubric_id!r}"
    )


def _stored(body: RubricCreate | RubricUpdate) -> dict[str, Any]:
    """
    Shape a payload for the store, stamping each competency with its derived id.

    Stamped at write time rather than derived on read so the id is in the data a report cites.
    Deriving it on every read would work until someone renamed a competency, at which point old
    session coverage would key off an id no longer computable from the current name.
    """
    payload = body.model_dump(exclude_unset=True)
    if payload.get("competencies") is not None:
        payload["competencies"] = [
            {**competency, "id": slug(str(competency["name"]))}
            for competency in payload["competencies"]
        ]
    return payload


def _warnings(record: dict[str, Any]) -> list[str]:
    """
    Things that will not fail but will disappoint. Reported, not enforced.

    A rubric with no competencies or with signal-less competencies is a legitimate draft; it is
    also one that cannot do its job, and the gap between "saved" and "working" is exactly where
    a feature gets quietly mistrusted.
    """
    notes: list[str] = []
    competencies = record.get("competencies") or []
    if not competencies:
        notes.append("no competencies yet, so this rubric will not steer an interview")
    blind = [
        str(c.get("name"))
        for c in competencies
        if not [s for s in (c.get("signals") or []) if str(s).strip()]
    ]
    if blind:
        notes.append(
            "no signals for "
            + ", ".join(blind)
            + " — these can never be evidenced and will be reported as exhausted after "
            "max_turns probes"
        )
    return notes


@router.get("")
async def list_rubrics() -> list[dict[str, Any]]:
    return store.list(COLLECTION)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_rubric(body: RubricCreate) -> dict[str, Any]:
    record = store.create(COLLECTION, ID_PREFIX, _stored(body))
    return {**record, "warnings": _warnings(record)}


@router.get("/{rubric_id}")
async def get_rubric(rubric_id: str) -> dict[str, Any]:
    try:
        record = store.get(COLLECTION, rubric_id)
    except NotFound as exc:
        raise _not_found(rubric_id) from exc
    return {**record, "warnings": _warnings(record)}


@router.patch("/{rubric_id}")
async def update_rubric(rubric_id: str, body: RubricUpdate) -> dict[str, Any]:
    try:
        record = store.update(COLLECTION, rubric_id, _stored(body))
    except NotFound as exc:
        raise _not_found(rubric_id) from exc
    return {**record, "warnings": _warnings(record)}


@router.delete("/{rubric_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rubric(rubric_id: str) -> None:
    """
    Hard delete, and deliberately not cascading — the same call as `agents.py`.

    A session record holds the coverage it measured, not a live reference to the rubric, so
    deleting one does not damage a past interview's report. An agent still pointing at it will
    refuse to start, which is the loud failure `agent_config` prefers to a silent downgrade.
    """
    try:
        store.delete(COLLECTION, rubric_id)
    except NotFound as exc:
        raise _not_found(rubric_id) from exc
