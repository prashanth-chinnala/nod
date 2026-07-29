"""
The attributed record every write goes through.

**Why writes are recorded and reads are not.** An assistant that can quietly change a rubric or
a score is an audit problem wearing a helpful face. The asymmetry is deliberate and it is the
whole security model here: reading is free and leaves no trace, and anything that changes stored
state produces a row saying what changed, who asked for it, that it came via the assistant, and
when.

**Why the assistant proposes and never commits.** Nothing in this package mutates a rubric or a
scorecard. A write tool records a *proposal* against the target and returns its id; a human
applies it through the console, and applying it records who did that. So the trail distinguishes
two things that a single "updated_at" cannot: what a model suggested, and what a person decided.
That distinction is also the legal position — a rejected candidate asking why gets a versioned
rubric, a verified quote, and a named human decision.

**On the actor field, honestly.** There is no authentication anywhere in this product yet, so
`actor` is whatever the caller claims. That makes this an audit trail in shape but not yet in
force: it will tell you a change was proposed via the assistant and cannot yet prove who.
Recording the field now means the trail is complete from the day auth exists rather than
starting then, and the gap is stated here rather than left for someone to assume otherwise.
"""

from __future__ import annotations

from typing import Any, Literal

from avatar.store import Store, now_iso, store

COLLECTION = "assistant_actions"

Kind = Literal[
    "rubric_change_proposed",
    "rescore_requested",
    "flagged_for_review",
    "note_added",
    "anchor_promotion_proposed",
]
"""
Every kind of write the assistant can make, enumerated.

A closed set rather than a free string, because this list *is* the answer to "what can this
thing do to my data" -- and a reviewer must be able to read that answer without grepping the
tool definitions. Adding a capability means adding a member here, which is a deliberately
visible act.
"""

Status = Literal["proposed", "applied", "rejected"]
"""
`proposed` is the only status this package ever writes.

`applied` and `rejected` exist so the console can close the loop, and they are set by whatever
records the human's decision -- never from here. An assistant that could mark its own proposal
applied would make the trail worthless.
"""


def record(
    kind: Kind,
    *,
    target: str,
    summary: str,
    actor: str,
    detail: dict[str, Any] | None = None,
    data: Store | None = None,
) -> dict[str, Any]:
    """
    Write one attributed action and return it, id included.

    `target` is the id of the thing the action is about -- a session, a rubric, a competency --
    so the console can show a resource's history without scanning every action. `summary` is one
    line a human reads in a list; `detail` carries the machine-readable proposal, kept separate
    so the summary never has to be parsed.

    `via` is hardcoded rather than a parameter. The point of the field is to distinguish an
    assistant-originated change from a hand-made one, and a caller that could set it could erase
    exactly the distinction the record exists to preserve.
    """
    return (data or store).create(
        COLLECTION,
        "act",
        {
            "kind": kind,
            "target": target,
            "summary": summary,
            "actor": actor,
            "via": "assistant",
            "status": "proposed",
            "detail": detail or {},
            "proposed_at": now_iso(),
            # Set when a human decides, by the console -- not here. Present as null so a reader
            # can see the decision is outstanding rather than infer it from an absent key.
            "decided_at": None,
            "decided_by": None,
        },
    )


def history(target: str, *, data: Store | None = None) -> list[dict[str, Any]]:
    """
    Every assistant action about one target, newest first.

    Filtered in Python rather than by a query, which is fine at this scale and is the same
    trade-off `avatar.store` documents for itself. It becomes wrong at a few thousand actions,
    and that is the point at which the store should be a database rather than this function
    being cleverer.
    """
    return [
        action
        for action in (data or store).list(COLLECTION)
        if action.get("target") == target
    ]
