"""
CRUD for tools — functions the agent may call mid-interview.

A tool record is two things at once, and the validation here exists because they pull in
opposite directions:

  * a **schema handed to the model**. `name` and `parameters_schema` are emitted verbatim
    into the tool definition the LLM sees, so a name the model cannot legally emit is not a
    cosmetic problem — it is a tool that never fires, with no error anywhere.
  * a **call inside a conversational turn**. Every field under `kind`/`url`/`timeout_ms`
    is a latency and reachability decision, and a bad one shows up as a stalled interview
    rather than as a 500.

So the rules below reject at the boundary rather than defaulting something plausible.
A tool misconfigured here fails silently mid-conversation, which is the one failure mode
this resource can produce and the hardest to diagnose after the fact.

Per-tool measured p95 is the other half of this story (ROADMAP §3.4) and is not here:
nothing has executed a tool yet, so there is no number to show.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from avatar.store import NotFound, Store, store

COLLECTION = "tools"
ID_PREFIX = "tool"

NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
"""
The identifier the model emits to call this tool.

Lowercase snake_case and nothing else, because this string travels into the function
schema sent to the provider: a space, a dot, or a leading digit is either rejected by the
provider or produces a name the model paraphrases instead of emitting exactly. Either way
the call never lands, and the only visible symptom is an interviewer that declines to use
a tool it was given.
"""

NAME_MAX_LENGTH = 64
"""Both major providers cap function names at 64 characters. Longer is rejected upstream."""

TIMEOUT_MIN_MS = 1
TIMEOUT_MAX_MS = 5000
DEFAULT_TIMEOUT_MS = 1500
"""
Hard bounds on the per-call deadline.

A tool call inserts a round trip *inside* a conversational turn that already measures
2.7-5.8s (PROCESS.md §3.4), so an unbounded timeout is a hung interview: the candidate
sits in front of a silent, frozen avatar with no way to tell whether it is thinking or
dead. 5000ms is already indefensible as a product decision and is set as the outer wall,
not a recommendation. `0` and negatives are rejected rather than clamped — a zero deadline
would make the tool a guaranteed no-op, which is worse than the misconfiguration it came
from because it looks configured.
"""


def get_store() -> Store:
    """
    Indirection so tests can bind a `Store(tmp_path)` via `dependency_overrides`.

    Without it the routes close over the process-wide default and a test run writes into
    whatever `AVATAR_DATA_DIR` points at — i.e. someone's real console data.
    """
    return store


StoreDep = Annotated[Store, Depends(get_store)]

Kind = Literal["http", "builtin"]


def _require_url_for_http(kind: Kind, url: str | None) -> None:
    """
    An `http` tool with no endpoint is a silent no-op mid-conversation.

    Shared by create and update because the invariant belongs to the stored record, not to
    one request shape: a PATCH that flips `kind` to "http" on a record that never had a url
    breaks it just as thoroughly as a bad POST. Whitespace counts as absent — `" "` is
    exactly as uncallable as `None`, and is what a half-filled form submits.
    """
    if kind == "http" and not (url or "").strip():
        raise ValueError("kind='http' requires a url; a tool with no endpoint can never fire")


class ToolCreate(BaseModel):
    """
    A new tool.

    `extra="forbid"` on purpose: a misspelled field name would otherwise be dropped in
    silence, and "I set the timeout and it ignored me" is the resulting bug report.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=NAME_PATTERN, max_length=NAME_MAX_LENGTH)
    description: str = ""
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    kind: Kind
    url: str | None = None
    timeout_ms: int = Field(default=DEFAULT_TIMEOUT_MS, ge=TIMEOUT_MIN_MS, le=TIMEOUT_MAX_MS)
    enabled: bool = True

    @model_validator(mode="after")
    def _check_reachable(self) -> ToolCreate:
        _require_url_for_http(self.kind, self.url)
        return self


class ToolPatch(BaseModel):
    """
    A partial update. Omitted fields are left alone.

    Note what this cannot express: clearing a field back to null. `Store.update` drops
    `None` values from the patch, so `{"url": null}` is a no-op rather than a deletion.
    That is the safer default here — an accidental null in a form submission must not
    quietly strip the endpoint off a working http tool — and a real clear is a delete and
    recreate. Written down because the alternative is discovering it as a bug.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, pattern=NAME_PATTERN, max_length=NAME_MAX_LENGTH)
    description: str | None = None
    parameters_schema: dict[str, Any] | None = None
    kind: Kind | None = None
    url: str | None = None
    timeout_ms: int | None = Field(default=None, ge=TIMEOUT_MIN_MS, le=TIMEOUT_MAX_MS)
    enabled: bool | None = None


router = APIRouter(prefix="/tools", tags=["tools"])


def _load(data: Store, tool_id: str) -> dict[str, Any]:
    try:
        return data.get(COLLECTION, tool_id)
    except NotFound:
        # 404 rather than a bare KeyError traceback: an unknown id is a routine client
        # mistake (stale tab, deleted record), not a server fault.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no tool {tool_id!r}"
        ) from None


@router.get("")
def list_tools(data: StoreDep) -> list[dict[str, Any]]:
    """Newest first, per `Store.list`. The console's table renders this order verbatim."""
    return data.list(COLLECTION)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_tool(body: ToolCreate, data: StoreDep) -> dict[str, Any]:
    return data.create(COLLECTION, ID_PREFIX, body.model_dump())


@router.get("/{tool_id}")
def get_tool(tool_id: str, data: StoreDep) -> dict[str, Any]:
    return _load(data, tool_id)


@router.patch("/{tool_id}")
def update_tool(tool_id: str, body: ToolPatch, data: StoreDep) -> dict[str, Any]:
    """
    Merge a patch, re-checking the reachability invariant against the *merged* record.

    Validating the patch alone would let `{"kind": "http"}` land on a record with no url —
    a tool that passes validation at write time and dead-ends at call time.
    """
    current = _load(data, tool_id)
    patch = body.model_dump(exclude_unset=True, exclude_none=True)
    merged = {**current, **patch}
    try:
        _require_url_for_http(merged.get("kind", "builtin"), merged.get("url"))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    return data.update(COLLECTION, tool_id, patch)


@router.delete("/{tool_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_tool(tool_id: str, data: StoreDep) -> None:
    try:
        data.delete(COLLECTION, tool_id)
    except NotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no tool {tool_id!r}"
        ) from None
