"""
CRUD for pronunciation lexicons — per-term overrides applied to text before synthesis.

This is the cheapest quality win the orchestration layer has. An engineering interview is
full of words every TTS voice mangles — nginx, PostgreSQL, Kubernetes, Kafka, the
candidate's own surname — and each one makes the interviewer sound like it has never worked
in the field. The fix is a string substitution, so it belongs where the text is: upstream of
the synthesiser, which means it works whichever voice `AVATAR_TTS` resolved to and needs no
model change, no vendor feature, and no re-render.

The substitution rules below are the entire subtlety, and both were bugs before they were
rules. See `apply_lexicon`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from avatar.store import NotFound, Store, store

COLLECTION = "pronunciations"
ID_PREFIX = "lex"

WORD_LEFT = r"(?<!\w)"
WORD_RIGHT = r"(?!\w)"
"""
Whole-word edges as lookarounds rather than `\\b`.

`\\b` is defined relative to the pattern's own first and last characters, so it silently
stops working for the terms most likely to need an override: `\\bC\\+\\+\\b` never matches
`C++` because the boundary after `+` requires a word character to its left. A negative
lookahead for `\\w` asks the question that actually matters — "is the neighbouring character
part of a word?" — and gets `C++`, `.NET`, and `node.js` right for free.
"""


def get_store() -> Store:
    """
    Indirection so tests can bind a `Store(tmp_path)` via `dependency_overrides`.

    Without it the routes close over the process-wide default and a test run writes into
    whatever `AVATAR_DATA_DIR` points at — i.e. someone's real console data.
    """
    return store


StoreDep = Annotated[Store, Depends(get_store)]

Trimmed = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
"""
A required string, stored without surrounding whitespace.

Stripping before the length check is what makes `" "` a validation error rather than a
stored value. A term of one space would compile into a pattern that matches every gap in
every sentence; a `say` of one space would delete the word it was supposed to fix. Both are
what a half-filled row in the console's entry editor submits.
"""


class Entry(BaseModel):
    """One override: say `term` as `say`."""

    model_config = ConfigDict(extra="forbid")

    term: Trimmed
    say: Trimmed


def _reject_duplicate_terms(entries: list[Entry]) -> list[Entry]:
    """
    Two entries for the same term, differing only in case, are rejected.

    Matching is case-insensitive, so `Kafka` and `kafka` are the same rule with two answers
    and which one wins depends on list order — a lexicon that appears to be configured and
    produces the other pronunciation about half the time someone edits it. Rejecting at the
    boundary is the only place this is cheap to explain.
    """
    seen: set[str] = set()
    for entry in entries:
        key = entry.term.casefold()
        if key in seen:
            raise ValueError(
                f"duplicate term {entry.term!r}: matching is case-insensitive, so this "
                "would be two answers for one rule"
            )
        seen.add(key)
    return entries


class LexiconCreate(BaseModel):
    """
    A new lexicon.

    `extra="forbid"` on purpose: `{"entry": [...]}` for `{"entries": [...]}` would otherwise
    be dropped in silence and create an empty lexicon that reports success, and "I added ten
    terms and nothing changed" is the resulting bug report.
    """

    model_config = ConfigDict(extra="forbid")

    name: Trimmed
    entries: list[Entry] = Field(default_factory=list)

    @field_validator("entries")
    @classmethod
    def _check_terms(cls, entries: list[Entry]) -> list[Entry]:
        return _reject_duplicate_terms(entries)


class LexiconPatch(BaseModel):
    """
    A partial update. Omitted fields are left alone.

    `entries` replaces wholesale rather than merging: the console edits the whole list at
    once, and a merge would need a per-entry identity the entries do not have. A patch
    cannot clear `name` back to null — `Store.update` drops `None` — which is the safer
    default for a form submission that lost a field.
    """

    model_config = ConfigDict(extra="forbid")

    name: Trimmed | None = None
    entries: list[Entry] | None = None

    @field_validator("entries")
    @classmethod
    def _check_terms(cls, entries: list[Entry] | None) -> list[Entry] | None:
        return None if entries is None else _reject_duplicate_terms(entries)


class ApplyRequest(BaseModel):
    """
    Text to run the lexicon over.

    Empty text is allowed rather than rejected: the console's live preview calls this on
    every keystroke, and 422-ing an empty box would make clearing the field look like a
    server error.
    """

    model_config = ConfigDict(extra="forbid")

    text: str


def apply_lexicon(entries: Sequence[Mapping[str, Any]], text: str) -> str:
    """
    Rewrite `text` with every override applied, case-insensitively and whole-word only.

    Two defects this shape exists to prevent, both of which the obvious implementation has:

    1. **Substring matching.** A plain `str.replace` of `Kafka` → `KAFF-ka` turns
       `Kafkaesque` into `KAFF-kaesque`, so the fix for one word breaks every word that
       contains it. Hence the whole-word lookarounds.

    2. **Re-substituting inside a replacement.** Looping over the entries and replacing each
       in turn re-scans text that a previous entry already produced: with `nginx` →
       `engine ex` and `ex` → `eks` in the same lexicon, `nginx` comes out as `engine eks`.
       One pass with a single alternation is the fix — `re.sub` resumes after each match in
       the *input*, so a replacement is never read again.

    Longest term first, because `re` alternation is leftmost-first-alternative: with `SQL`
    ahead of `PostgreSQL`, the shorter rule would claim the tail of the longer one.
    """
    pairs = [
        (str(entry["term"]), str(entry["say"]))
        for entry in entries
        if entry.get("term") and entry.get("say") is not None
    ]
    if not pairs or not text:
        return text

    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    say_for = {term.casefold(): say for term, say in pairs}
    pattern = re.compile(
        WORD_LEFT + "(?:" + "|".join(re.escape(term) for term, _ in pairs) + ")" + WORD_RIGHT,
        re.IGNORECASE,
    )

    # `say_for.get(..., matched)` rather than `[...]`: `re.IGNORECASE` and `str.casefold`
    # disagree on a handful of non-ASCII characters, and a lexicon containing one must leave
    # the word alone, not 500 the request that was trying to read it.
    return pattern.sub(lambda m: say_for.get(m.group(0).casefold(), m.group(0)), text)


router = APIRouter(prefix="/pronunciations", tags=["pronunciations"])


def _load(data: Store, lexicon_id: str) -> dict[str, Any]:
    try:
        return data.get(COLLECTION, lexicon_id)
    except NotFound:
        # 404 rather than a bare KeyError traceback: an unknown id is a routine client
        # mistake (stale tab, deleted record), not a server fault.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no lexicon {lexicon_id!r}"
        ) from None


@router.get("")
def list_lexicons(data: StoreDep) -> list[dict[str, Any]]:
    """Newest first, per `Store.list`. The console's table renders this order verbatim."""
    return data.list(COLLECTION)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_lexicon(body: LexiconCreate, data: StoreDep) -> dict[str, Any]:
    return data.create(COLLECTION, ID_PREFIX, body.model_dump())


@router.get("/{lexicon_id}")
def get_lexicon(lexicon_id: str, data: StoreDep) -> dict[str, Any]:
    return _load(data, lexicon_id)


@router.patch("/{lexicon_id}")
def update_lexicon(lexicon_id: str, body: LexiconPatch, data: StoreDep) -> dict[str, Any]:
    """404s before writing, so a PATCH against a deleted lexicon cannot resurrect it."""
    _load(data, lexicon_id)
    return data.update(
        COLLECTION, lexicon_id, body.model_dump(exclude_unset=True, exclude_none=True)
    )


@router.delete("/{lexicon_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_lexicon(lexicon_id: str, data: StoreDep) -> None:
    try:
        data.delete(COLLECTION, lexicon_id)
    except NotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no lexicon {lexicon_id!r}"
        ) from None


@router.post("/{lexicon_id}/apply")
def apply_to_text(lexicon_id: str, body: ApplyRequest, data: StoreDep) -> dict[str, str]:
    """
    What the synthesiser would receive, given this lexicon.

    Exposed as an endpoint rather than left as a library call so the console's preview runs
    the same code the TTS path will. A JavaScript reimplementation in the browser would
    reassure the operator about substitutions the server does not actually make — and this
    function's whole value is that its two edge cases are gnarlier than they look.
    """
    record = _load(data, lexicon_id)
    return {"text": apply_lexicon(record.get("entries") or [], body.text)}
