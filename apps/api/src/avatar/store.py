"""
A JSON-file-backed store for console resources.

**Why not a database.** Every resource here is a handful of small documents edited by one
operator: agents, faces, knowledge bases, tools, guardrails, pronunciations. Postgres would
add a service to run, a migration tool, and a connection pool to tune, in exchange for
guarantees nothing here needs. A directory of JSON files is inspectable with `cat`, diffable
in git, and survives a process restart — which is the entire requirement.

**What this deliberately is not.** No concurrent-writer safety beyond a whole-file atomic
replace, no transactions across collections, no query language. The moment two operators
edit simultaneously, or a collection grows past a few thousand rows, this should be replaced
rather than extended. Writing that down now is cheaper than discovering the boundary later:
the `Store` surface is small enough that swapping the backing for SQL is a contained change.

Atomic replace matters even at this size. A half-written JSON file is a corrupted resource
that fails on read forever, and a crash mid-write is exactly when it would happen — so every
write goes to a temporary file in the same directory and is renamed over the target, which
POSIX guarantees is atomic.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATA_ROOT = Path(os.environ.get("AVATAR_DATA_DIR", "data"))
"""
Where resources live. Overridable so tests get a tmp_path and never touch real data.

Relative by default, resolved against the working directory rather than the package, because
the data belongs to the deployment and not to the installed library.
"""


def now_iso() -> str:
    """UTC, ISO-8601, second precision. Sorts lexicographically, which list views rely on."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    """
    A prefixed, URL-safe identifier: `agent_3f9a1c2b`.

    Prefixed on purpose. A bare UUID in a log line or a bug report tells you nothing about
    what it refers to, and mixing up a face id and an agent id is the kind of mistake that
    is silent until something renders the wrong persona.
    """
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class NotFound(KeyError):
    """Raised when an id does not exist. Mapped to a 404 by the router."""


class Store:
    """
    One directory per collection, one JSON file per record.

    Chosen over a single file per collection so that two resources being edited never
    contend for the same file, and so a corrupt record cannot take its whole collection down
    with it.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else DATA_ROOT

    def _dir(self, collection: str) -> Path:
        path = self.root / collection
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _file(self, collection: str, record_id: str) -> Path:
        # Guard against a traversal through an id. These ids are generated, but this
        # function is one URL parameter away from user input and the check is one line.
        if "/" in record_id or record_id.startswith("."):
            raise NotFound(record_id)
        return self._dir(collection) / f"{record_id}.json"

    # -- reads --------------------------------------------------------------

    def get(self, collection: str, record_id: str) -> dict[str, Any]:
        path = self._file(collection, record_id)
        try:
            data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise NotFound(record_id) from exc
        return data

    def list(self, collection: str) -> list[dict[str, Any]]:
        """
        Every record, newest first.

        A corrupt or half-written file is skipped rather than raising: one bad record must
        not make a whole page 500. It stays on disk to be inspected.
        """
        records: list[dict[str, Any]] = []
        for path in sorted(self._dir(collection).glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        records.sort(key=lambda r: str(r.get("created_at", "")), reverse=True)
        return records

    def iter_all(self, collection: str) -> Iterator[dict[str, Any]]:
        yield from self.list(collection)

    # -- writes -------------------------------------------------------------

    def create(self, collection: str, prefix: str, body: dict[str, Any]) -> dict[str, Any]:
        record = dict(body)
        record["id"] = new_id(prefix)
        record["created_at"] = record["updated_at"] = now_iso()
        self._write(collection, record)
        return record

    def update(self, collection: str, record_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """
        Merge a partial update. `id` and `created_at` are immutable.

        A PATCH that could rewrite an id would let one resource silently become another, and
        `created_at` moving would break the ordering every list view depends on.
        """
        record = self.get(collection, record_id)
        record.update({k: v for k, v in patch.items() if v is not None})
        record["id"] = record_id
        record["updated_at"] = now_iso()
        self._write(collection, record)
        return record

    def delete(self, collection: str, record_id: str) -> None:
        try:
            self._file(collection, record_id).unlink()
        except FileNotFoundError as exc:
            raise NotFound(record_id) from exc

    def _write(self, collection: str, record: dict[str, Any]) -> None:
        """
        Atomic replace: write a sibling temp file, fsync, rename over the target.

        The rename is the atomic step POSIX guarantees. Without it, a crash between opening
        the target and finishing the write leaves a truncated JSON file that fails to parse
        on every subsequent read — a resource permanently broken by an unlucky moment.
        """
        target = self._file(collection, record["id"])
        payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=target.parent, prefix=".tmp-", delete=False
        )
        try:
            with handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, target)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise


store = Store()
"""Process-wide default. Tests construct their own against a tmp_path."""
