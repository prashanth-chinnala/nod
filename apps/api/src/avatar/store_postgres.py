"""
The same six-method store, backed by PostgreSQL. Selected with `AVATAR_STORE=postgres`.

`store.py` names the conditions under which it should be replaced rather than extended: two
writers at once, or a collection past a few thousand rows. The first one arrived. This is the
replacement, and it is deliberately not a rewrite of the file store -- that keeps working, stays
the default, and a clean clone still runs with no services.

**The three things this fixes, in the order they bite.**

*Concurrency.* Every file-store write is read-modify-write: `update` loads the record, merges
the patch in Python, and rewrites the whole file. Three writers touch one live session -- the
recording setup, the coverage snapshot, and the turn flush -- and today they are safe only by
accident of being one process on one event loop. `uvicorn --workers 2` ends the accident with no
error message and no traceback, just a turn that is not there. Here `update` is one statement
and the merge happens in the engine.

*Referential integrity.* `agents.rubric_id` is a plain string on disk, so deleting a rubric an
agent still points at succeeds, and the failure surfaces when a candidate joins and the session
will not start. Foreign keys move that failure to the operator who asked for the delete, while
there is still a form open.

*Appending a turn.* The file store rewrites all N turns to append the (N+1)th, and two appends
at once both read the same array so the second silently drops the first. Turns are their own
table here, so a write is one row.

**What it costs, stated rather than implied.** A service to run and a schema to apply by hand
(`psql -f migrations/001_initial.sql`; there is no migration tool, and that file says why). And
this is the *synchronous* psycopg driver called from `async def` routers, so a store call holds
the event loop for the length of the query. That is not a regression -- file reads held it too
-- but it is a lower ceiling than the rest of this design suggests, and the fix is psycopg's
async connection or a threadpool, which is a change to the callers rather than to this file.

**The rule this module is written to.** The routers, the console and 666 tests were written
against what the file store returns, so the observable contract is whatever `store.py` does, not
whatever the schema would prefer. That is why `get` hands back `doc` untouched, why nothing
projects `agent_name` or a turn's `seq` into a record, and why an id containing a slash raises
`NotFound` here even though there is no path to traverse. Where the two backends cannot agree,
the divergence is named at the method that causes it instead of being smoothed over.

**Not included, on purpose.** No connection pool: `psycopg_pool` is a second dependency, and one
connection per process is what a sync driver on one event loop can use anyway. No retry on a
dropped connection -- the call that hits it fails and the next one reconnects. No
authentication; there is none anywhere in this product yet, and row-level security is where it
would go.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

from avatar.store import NotFound, Store, new_id, now_iso

DSN_ENV = "AVATAR_DATABASE_URL"
FALLBACK_DSN_ENV = "DATABASE_URL"


def dsn() -> str:
    """
    The connection string: `AVATAR_DATABASE_URL`, then `DATABASE_URL`, then nothing.

    "Nothing" is a real answer rather than an error. An empty conninfo makes libpq apply its own
    resolution -- `PGHOST`, `PGDATABASE`, `PGUSER`, the local socket -- which is exactly what
    `psql` with no arguments does. So on a machine where `psql` already works, this works, and
    nobody has to write a URL that restates the defaults they are already using.

    `DATABASE_URL` is honoured second because every managed Postgres sets it, and requiring an
    `AVATAR_`-prefixed copy of a variable the platform already injected is the kind of friction
    that gets solved by pasting a credential somewhere worse.
    """
    return os.environ.get(DSN_ENV) or os.environ.get(FALLBACK_DSN_ENV) or ""


class UnknownCollection(ValueError):
    """
    Raised for a collection this schema has no table for.

    A `ValueError` and not a `NotFound`, deliberately: a `NotFound` becomes a 404, which reads
    as "no such record" and hides a misspelled collection name behind an ordinary-looking
    response.
    """


class SessionEnded(RuntimeError):
    """
    Raised by `append_turn` when the session is already closed.

    Distinct from `NotFound` because the router that will eventually call it has to tell the two
    apart: a missing session is a 404 and an ended one is a 409, and the message for each is the
    only thing that tells an operator whether they have the wrong id or the wrong moment.
    """


class _Column(NamedTuple):
    """
    One typed column that mirrors a key in `doc`.

    `key` is the doc key that supplies the value, and it is not always the column name --
    `sessions.agent_name` is bound by `agent_id`, because the name is looked up from the agent
    rather than sent by the client. `insert` and `assign` are the two SQL shapes that column
    needs, kept as literal fragments so the statement built from them is readable in a log.

    Named placeholders (`%(agent_id)s`) rather than positional ones, so one value can appear
    twice in a statement -- which `agent_id` does, once for the column and once inside the
    subquery that finds the name.
    """

    column: str
    key: str
    insert: str
    assign: str


def _plain(column: str) -> _Column:
    """A column that is a doc key by the same name and nothing more -- the usual case."""
    return _Column(column, column, f"%({column})s", f"{column} = %({column})s")


_AGENT_NAME = "(select name from agents where id = %(agent_id)s)"


_TYPED: dict[str, tuple[_Column, ...]] = {
    "faces": (_plain("name"),),
    "voices": (_plain("name"),),
    "rubrics": (_plain("name"),),
    "guardrails": (_plain("name"),),
    "pronunciations": (_plain("name"),),
    "knowledge": (_plain("name"),),
    "tools": (_plain("name"),),
    "agents": (
        _plain("name"),
        _plain("face_id"),
        _plain("voice_ref_id"),
        _plain("rubric_id"),
        _plain("guardrail_id"),
        _plain("pronunciation_id"),
    ),
    "sessions": (
        _plain("agent_id"),
        # `coalesce(..., agent_name)` is what keeps the name after the agent is gone. Setting
        # `agent_id` to null -- detaching, or the FK's own ON DELETE SET NULL -- makes the
        # subquery empty, and without the coalesce the session list would start showing
        # "(deleted agent)" for interviews that have a perfectly good name on record. Repointing
        # a session at a different agent still overwrites it, because then the subquery finds
        # one.
        _Column(
            "agent_name",
            "agent_id",
            _AGENT_NAME,
            f"agent_name = coalesce({_AGENT_NAME}, agent_name)",
        ),
        # Cast explicitly: the doc carries an ISO-8601 string from `now_iso()`, and Postgres is
        # the parser rather than `datetime.fromisoformat`, so a malformed value fails naming
        # itself instead of raising a `ValueError` two frames from the caller.
        _Column(
            "ended_at",
            "ended_at",
            "%(ended_at)s::timestamptz",
            "ended_at = %(ended_at)s::timestamptz",
        ),
    ),
    "assistant_actions": (_plain("target"),),
}
"""
Every collection this schema has a table for, and the typed columns each one mirrors.

Doubles as the allowlist `_table` matches against -- see there for why that matters.
"""


class _Link(NamedTuple):
    """One join table: `key` is the doc key holding the ids, `column` the referencing column."""

    table: str
    column: str
    key: str


_LINKS: dict[str, tuple[_Link, ...]] = {
    "agents": (
        _Link("agent_knowledge_bases", "knowledge_id", "knowledge_base_ids"),
        _Link("agent_tools", "tool_id", "tool_ids"),
    ),
}
"""
The many-to-many attachments, which exist for the foreign key and nothing else.

`doc.knowledge_base_ids` stays the ordering authority and is what every reader gets back, so
these tables are never read here -- only rewritten, so that deleting a knowledge base an agent
uses fails. That is why there is no `position` column and why nothing reconstructs the list from
these rows.
"""

_TURN_COLUMNS: tuple[str, ...] = (
    "epoch",
    "heard",
    "said",
    "transcribed",
    "llm_ttft_ms",
    "tts_first_audio_ms",
    "first_frame_ms",
    "perceived_total_ms",
    "interrupted",
)
"""
The turn columns, in the order `Turn` declares its fields.

The order is cosmetic -- dict equality ignores it -- but a session record read out of this
backend and one read out of the file store are compared by eye often enough that having the keys
land in the same sequence is worth one tuple. `seq` is absent on purpose: the file store never
wrote one, and a key that exists on only one backend is how two backends stop being
interchangeable.
"""

APPEND_ATTEMPTS = 3
"""
How many times `append_turn` retries a `seq` collision.

Two concurrent appends can compute the same `max(seq) + 1`, and the primary key rejects one of
them. Retrying is the whole improvement over the file store, which resolves the same race by
overwriting one turn with the other and returning 201. Three attempts because each retry reads a
committed `max(seq)`, so a loop that needs a fourth is not contending -- it is broken.
"""


def _table(collection: str) -> str:
    """
    The table for a collection, and the allowlist that makes interpolating its name safe.

    SQL cannot parameterise an identifier, so every statement in this module formats a table
    name into the string. That is only safe because the name is matched against `_TYPED` first
    rather than passed through. Every caller is a module constant (`COLLECTION = "agents"`), so
    this should never fire -- but "should never" next to string formatting into SQL is not a
    combination worth trusting.

    A misspelled collection is a hard error here and silence in the file store, which invents
    the directory and returns an empty list for ever. This cannot invent a table, and it should
    not want to: an exception naming the typo is the more useful outcome of the two.
    """
    if collection not in _TYPED:
        known = ", ".join(sorted(_TYPED))
        raise UnknownCollection(f"no table for collection {collection!r}; known: {known}")
    return collection


def _guard(record_id: str) -> None:
    """
    The file store's traversal guard, kept even though nothing here touches a path.

    Not defensive theatre: the two backends have to answer the same way for the same input, and
    `get("agents", "../secrets")` raising `NotFound` is part of what the file store's callers --
    and its tests -- were written against. Dropping the check would make this backend return a
    404 where the other one does too, right up until someone relies on the difference.
    """
    if "/" in record_id or record_id.startswith("."):
        raise NotFound(record_id)


def _jsonb(value: Mapping[str, Any]) -> Any:
    """
    Wrap a dict so psycopg sends it as `jsonb` rather than as text.

    Imported here rather than at module scope for the same reason as the driver itself: a
    process that never selects this backend must not pay for the import.
    """
    from psycopg.types.json import Jsonb

    return Jsonb(dict(value))


def _validated_turns(value: Any) -> list[dict[str, Any]]:
    """
    Refuse a turn carrying a key the table has no column for.

    `turns` is the one nested structure with no `doc` to fall back on, so an unrecognised key
    would be dropped in silence -- a stored turn missing a field somebody wrote, discovered when
    a report needs it. `Turn` is `extra="forbid"` at the API boundary for exactly this reason;
    this is the same rule at the other end, for the writer that does not go through it
    (`server.py` assembles the dict by hand from telemetry events).

    What it deliberately does not do is fill in absent keys. Restating `Turn`'s defaults here
    would put a second copy of them in a module that must not import the router, so a turn with
    no `heard` is a not-null violation rather than an empty string. Both writers today set all
    nine keys, and the one that stops doing so should hear about it.
    """
    turns = [dict(turn) for turn in value]
    for turn in turns:
        unknown = sorted(set(turn) - set(_TURN_COLUMNS))
        if unknown:
            raise ValueError(
                f"turn has no column for {unknown}; the turns table is closed, so this key "
                "would be stored nowhere. Add the column to the schema or drop the key."
            )
    return turns


def _read_turns(conn: Any, session_ids: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """
    Every turn of every named session, grouped, in one query.

    One query rather than one per session because `list("sessions")` is the console's index
    page: a query per row is the N+1 that makes a list view slow enough to notice at exactly the
    point the data is worth looking at.
    """
    grouped: dict[str, list[dict[str, Any]]] = {session_id: [] for session_id in session_ids}
    if not session_ids:
        return grouped
    columns = ", ".join(_TURN_COLUMNS)
    rows = conn.execute(
        f"select session_id, {columns} from turns "
        "where session_id = any(%s) order by session_id, seq",
        (list(session_ids),),
    ).fetchall()
    for row in rows:
        grouped[row[0]].append(dict(zip(_TURN_COLUMNS, row[1:], strict=True)))
    return grouped


def _insert_turns(
    conn: Any, session_id: str, turns: Sequence[Mapping[str, Any]], *, start: int
) -> None:
    """Insert turns with `seq` running from `start`. The caller owns the transaction."""
    if not turns:
        return
    columns = ", ".join(("session_id", "seq", *_TURN_COLUMNS))
    placeholders = ", ".join(["%s"] * (2 + len(_TURN_COLUMNS)))
    rows = [
        (session_id, start + offset, *(turn.get(name) for name in _TURN_COLUMNS))
        for offset, turn in enumerate(turns)
    ]
    with conn.cursor() as cursor:
        cursor.executemany(f"insert into turns ({columns}) values ({placeholders})", rows)


def _sync_turns(conn: Any, session_id: str, turns: Sequence[Mapping[str, Any]]) -> None:
    """
    Make the child table match a whole `turns` list, appending when that is what happened.

    The six-method contract cannot express "append one turn" -- `update` receives the entire
    list, because that is what the file store's callers build and pass. So this reads how many
    rows exist and, when the incoming list is longer, inserts only the extra ones. That is the
    O(n^2) fix: appending the seventeenth turn writes one row instead of seventeen.

    **The trade-off, named.** A longer list is treated as an append, so an edit to an earlier
    turn *in the same patch* is not applied. Detecting that would mean reading every stored turn
    back to compare it, which is the per-append O(n) this table exists to remove. Sessions are
    append-only by design -- `sessions.py` has no PATCH and no DELETE, and says why -- so
    nothing in the product edits a turn in place. A shorter or equal-length list is not an
    append and is replaced whole, which is the slow path and the correct one.

    Still one statement short of race-free: two writers that each read the record and pass n+1
    turns can both compute the same tail. `append_turn` below is the version without that hole,
    and switching the router to it is a change to `sessions.py`, not to this file.
    """
    row = conn.execute(
        "select count(*) from turns where session_id = %s", (session_id,)
    ).fetchone()
    stored = int(row[0]) if row is not None else 0
    if len(turns) > stored:
        _insert_turns(conn, session_id, turns[stored:], start=stored)
        return
    conn.execute("delete from turns where session_id = %s", (session_id,))
    _insert_turns(conn, session_id, turns, start=0)


def _write_links(
    conn: Any, table: str, record_id: str, source: Mapping[str, Any], *, only_present: bool
) -> None:
    """
    Rewrite an agent's attachment rows from the doc keys that hold them.

    Delete-then-insert rather than a diff, because the list is a handful of ids and a diff would
    be more code to get wrong for no measurable gain. `select distinct` because the doc may
    legitimately hold the same id twice -- the API does not forbid it -- and the join table's
    primary key does; the duplicate belongs in `doc`, which is what readers get back, not here.

    `only_present` is what keeps a patch a patch: on update, an absent `tool_ids` key means
    "leave the tools alone", and clearing the rows because the key was not sent would detach
    every tool from an agent whose name was being corrected.

    The caller owns the transaction, and must have one. This is the single place where rule 7's
    "one statement" is not achievable: the doc merge and these rows are two writes, so they need
    a transaction to stop a crash between them from leaving an agent whose doc lists a knowledge
    base that no longer protects it from deletion.
    """
    for link in _LINKS.get(table, ()):
        if only_present and link.key not in source:
            continue
        ids = [str(value) for value in (source.get(link.key) or [])]
        conn.execute(f"delete from {link.table} where agent_id = %s", (record_id,))
        conn.execute(
            f"insert into {link.table} (agent_id, {link.column}) "
            "select distinct %s, value from unnest(%s::text[]) as t(value)",
            (record_id, ids),
        )


class PostgresStore(Store):
    """
    The `Store` surface over the schema in `migrations/001_initial.sql`.

    **Why it subclasses the file store instead of implementing a Protocol.** Every annotation
    that mentions a store in this repo and in `apps/assistant` says `Store` -- four router
    dependencies, `audit.record(data=...)`, and the module-level singleton. A Protocol would be
    the tidier description of "the six methods", and it would mean editing all of those for a
    type checker's benefit, in two apps, to express something the subclass already expresses.
    The inheritance is real: the six methods have the same signatures and the same contract.

    What it costs is that the file-specific helpers come along, and a `PostgresStore` that
    reached one would quietly write JSON next to a database that is supposed to be
    authoritative. So `_dir`, `_file` and `_write` are overridden to raise, and `__init__` never
    sets `root` -- there is deliberately nothing here that could be mistaken for a data
    directory.

    **One connection, opened on first use, re-opened after a fork.** Lazy so that constructing
    the store -- which happens at import, for the singleton -- cannot fail on a database that is
    not up yet; the failure belongs to the first request instead of to the import. The pid check
    exists because `uvicorn --workers 2` forks, and a connection inherited across a fork is one
    socket with two processes talking over it, which corrupts both sessions in ways that read
    like data loss. The child abandons the inherited object without closing it: closing sends a
    terminate down the socket the parent is still using.

    **One lock, held for a whole operation, and it is not optional.** Half the routers are `def`
    rather than `async def` -- `knowledge`, `tools`, `guardrails` and `pronunciations` all take
    their store as a dependency -- so FastAPI runs them in a threadpool and two requests really
    do call this object from two threads. A psycopg connection is thread-safe for one statement
    at a time, but a `transaction()` block is per-connection and not per-thread: two threads
    entering one raises `OutOfOrderTransactionNesting`, and the near miss is worse than the
    error -- a plain `execute` from another thread lands *inside* whatever transaction is open,
    so its write disappears if that transaction rolls back. Both were reproduced with eight
    threads on one store before this lock existed.

    The cost, stated plainly: store access is serialised within a process, so the concurrency
    this whole design buys is across processes and not across threads. That is the right shape
    for a sync driver -- the alternatives are a connection per thread, which multiplies
    `max_connections` by the threadpool size, or psycopg's async connection, which is a change
    to every caller. Two stores are two connections and two locks, which is what makes
    `uvicorn --workers 2` the thing that actually goes faster.
    """

    def __init__(self, connection_string: str | None = None) -> None:
        self.dsn = connection_string if connection_string is not None else dsn()
        self._conn: Any = None
        self._pid: int | None = None
        # Reentrant because `iter_all` goes through `list`, and a second acquisition on the same
        # thread must not deadlock on a store that is otherwise correct.
        self._lock = threading.RLock()

    # -- connection ---------------------------------------------------------

    def _connection(self) -> Any:
        """
        The live connection, opening or replacing it if needed.

        `autocommit=True` so a single statement is a single round trip with no explicit COMMIT,
        and so a failed statement does not leave the connection in an aborted transaction that
        rejects every subsequent query -- which is how a sync driver in a long-lived process
        turns one bad write into an outage. The multi-statement paths open their own transaction
        explicitly, which is visible at the point where atomicity is actually required.
        """
        try:
            import psycopg
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "AVATAR_STORE=postgres needs the psycopg driver, which is an optional extra: "
                "pip install -e '.[postgres]'. Unset AVATAR_STORE to use the file store."
            ) from exc

        pid = os.getpid()
        conn = self._conn
        if conn is not None:
            if self._pid != pid:
                conn = self._conn = None  # inherited across a fork; abandon, do not close
            elif conn.closed or conn.broken:
                conn = self._conn = None
        if conn is None:
            conn = self._conn = psycopg.connect(self.dsn, autocommit=True)
            self._pid = pid
        return conn

    def close(self) -> None:
        """Close the connection if there is one. Not on the `Store` contract; tests want it."""
        with self._lock:
            if self._conn is not None and self._pid == os.getpid():
                self._conn.close()
            self._conn = None

    # -- reads --------------------------------------------------------------

    def get(self, collection: str, record_id: str) -> dict[str, Any]:
        """
        The record as `doc` holds it, plus `turns` for a session.

        `doc` is returned untouched: the typed columns are not overlaid onto it. For all but one
        column that choice is invisible, since each is written from the same patch that writes
        its doc key. The exception is `sessions.agent_id` after its agent is deleted -- the
        column is null and the doc still names the agent that ran the interview. That
        disagreement is the dangling id `delete_agent` chose over rewriting history, and
        overlaying the column would erase exactly what it kept.
        """
        with self._lock:
            table = _table(collection)
            _guard(record_id)
            conn = self._connection()
            row = conn.execute(
                f"select doc from {table} where id = %s", (record_id,)
            ).fetchone()
            if row is None:
                raise NotFound(record_id)
            record: dict[str, Any] = row[0]
            if table == "sessions":
                record["turns"] = _read_turns(conn, [record_id])[record_id]
            return record

    def list(self, collection: str) -> list[dict[str, Any]]:
        """
        Every record, newest first -- ordered by the doc's `created_at`, not by the column.

        Ordering on `doc->>'created_at'` rather than the timestamptz column is what makes this
        list identical to the file store's, which sorts on that same string. The column is the
        tie-break: it has microsecond resolution where the doc key has one second, so records
        written in the same second come back in insertion order. The file store's answer there
        is filename order, which is generated-hex order, which is arbitrary -- so within one
        second the two backends can disagree, and only one of them is defensible.

        The cost of ordering on an expression: the `created_at desc` index does not serve it, so
        this is a sort over the collection. Right at nine collections of a few hundred rows, and
        the fix when it stops being right is an index on `(doc->>'created_at')` -- a migration,
        not a change here.

        Nothing skips corrupt records the way the file store does, because there is no such
        state: `jsonb` cannot hold a half-written document.
        """
        with self._lock:
            table = _table(collection)
            conn = self._connection()
            rows = conn.execute(
                f"select doc from {table} "
                "order by doc->>'created_at' desc nulls last, created_at desc"
            ).fetchall()
            records: list[dict[str, Any]] = [row[0] for row in rows]
            if table == "sessions":
                turns = _read_turns(conn, [str(record["id"]) for record in records])
                for record in records:
                    record["turns"] = turns[str(record["id"])]
            return records

    def iter_all(self, collection: str) -> Iterator[dict[str, Any]]:
        """
        Every record, as an iterator. Reads them all first, exactly as the file store does.

        A server-side cursor would make this stream, and would also make it the one method whose
        behaviour differs between backends -- a caller holding a cursor open across an `await`
        is a different failure mode from a caller holding a list. Identical behaviour is worth
        more than the memory today; a named cursor is the change when a collection outgrows RAM.
        """
        yield from self.list(collection)

    # -- writes -------------------------------------------------------------

    def create(self, collection: str, prefix: str, body: dict[str, Any]) -> dict[str, Any]:
        """
        Insert one record, its attachment rows, and any turns it arrived with.

        The returned record is assembled here rather than read back, so it is the same object
        the file store would have returned, key for key. `created_at` and `updated_at` are
        written into the doc by `now_iso()` -- second precision, the string every list view
        sorts on -- and the columns take the database's own `now()`. They agree to the second,
        and the column is the finer of the two, which is why `list` uses it as its tie-break.

        One transaction always, even for the collections that have no child rows to write. A
        branch to save two round trips on a write that happens at operator pace is a second code
        path to keep correct, and this one is short enough to read.
        """
        with self._lock:
            table = _table(collection)
            record = dict(body)
            record["id"] = new_id(prefix)
            record["created_at"] = record["updated_at"] = now_iso()
            # Turns live in their own table and the schema forbids the key in `doc`, so it
            # comes out of the body before the insert and goes back on afterwards.
            turns = _validated_turns(record.pop("turns", [])) if table == "sessions" else []

            typed = _TYPED[table]
            columns = ", ".join(("id", "doc", *(column.column for column in typed)))
            values = ", ".join(("%(_id)s", "%(_doc)s", *(column.insert for column in typed)))
            params: dict[str, Any] = {"_id": record["id"], "_doc": _jsonb(record)}
            for column in typed:
                params[column.key] = record.get(column.key)

            conn = self._connection()
            with conn.transaction():
                conn.execute(f"insert into {table} ({columns}) values ({values})", params)
                _write_links(conn, table, str(record["id"]), record, only_present=False)
                _insert_turns(conn, str(record["id"]), turns, start=0)
            if table == "sessions":
                record["turns"] = turns
            return record

    def update(self, collection: str, record_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """
        Merge a partial update in the database, in one statement.

        `doc = doc || patch` is the whole point. jsonb concatenation shallow-merges, which is
        what `record.update(patch)` does in the file store -- *including* setting a key to null
        rather than dropping it, which that store was specifically fixed to do, because dropping
        it made "not sent" and "explicitly cleared" the same request and left a console picker
        that appeared to detach a rubric and did nothing.

        Doing it server-side is what closes the race the file store leaves open: three writers
        touch one live session, and read-modify-write means the last one to finish overwrites
        what the others merged. A single statement has no window to lose them in.

        Where a patch key mirrors a typed column, the column is set in the same statement, so
        there is no instant at which the column and the doc disagree about which rubric an agent
        uses.

        `id` is dropped from the patch, matching the file store, which merges it and then
        re-pins it. The schema's `doc->>'id' = id` CHECK would catch a merged-in id anyway; a
        record that served itself under one id while claiming another is not a failure worth
        leaving to a constraint.

        `created_at` is dropped too, and that one is the single place these two backends do not
        agree, so it is stated rather than left to be found. `Store.update`'s docstring says
        both keys are immutable and its code enforces only the id, so a patch carrying
        `created_at` moves it on the file store and does not here. This is the stricter half and
        the defensible one: `created_at` is what `list` orders on and what the column holds as
        the true insert time, so honouring such a patch would reorder the console and put the
        two copies permanently out of step. Nothing an operator can do reaches it -- every patch
        model is `extra="forbid"` and none declares the field -- so the callers that make the
        difference visible are the ones that bypass the routers: the assistant and the scripts.
        """
        with self._lock:
            table = _table(collection)
            _guard(record_id)
            immutable = ("id", "created_at")
            merged = {k: v for k, v in patch.items() if k not in immutable}
            turns = (
                _validated_turns(merged.pop("turns"))
                if table == "sessions" and "turns" in merged
                else None
            )
            merged["updated_at"] = now_iso()

            assignments = ["doc = doc || %(_patch)s", "updated_at = now()"]
            params: dict[str, Any] = {"_id": record_id, "_patch": _jsonb(merged)}
            for column in _TYPED[table]:
                if column.key in patch:
                    assignments.append(column.assign)
                    params[column.key] = patch[column.key]
            statement = (
                f"update {table} set {', '.join(assignments)} where id = %(_id)s returning doc"
            )

            conn = self._connection()
            with conn.transaction():
                row = conn.execute(statement, params).fetchone()
                if row is None:
                    raise NotFound(record_id)
                record: dict[str, Any] = row[0]
                _write_links(conn, table, record_id, patch, only_present=True)
                if turns is not None:
                    _sync_turns(conn, record_id, turns)
            if table == "sessions":
                # A patch that carried turns already knows what they are -- returning the
                # list it passed saves reading every turn back to answer with the same
                # thing. Any other patch has to read them, because the contract returns the
                # whole record: the write is O(1) now, the response is still O(turns), and
                # only the caller can fix that.
                record["turns"] = (
                    turns if turns is not None else _read_turns(conn, [record_id])[record_id]
                )
            return record

    def delete(self, collection: str, record_id: str) -> None:
        """
        Hard delete, with the foreign keys deciding what is allowed.

        Deleting a rubric, face, guardrail, pronunciation, knowledge base or tool that an agent
        still references now fails -- ON DELETE RESTRICT -- where the file store succeeded and
        left the agent pointing at nothing. That is the change this schema exists for, and it is
        also the one behaviour difference an operator will see: psycopg raises
        `ForeignKeyViolation`, the router does not catch it, so the console gets a 500 naming
        the constraint instead of a 409 explaining it. Mapping it belongs in the routers.

        Deleting an agent still works: its attachment rows cascade, and the sessions it ran keep
        their transcripts with a null column and the agent id still in the doc.
        """
        with self._lock:
            table = _table(collection)
            _guard(record_id)
            cursor = self._connection().execute(
                f"delete from {table} where id = %s", (record_id,)
            )
            if cursor.rowcount == 0:
                raise NotFound(record_id)

    def append_turn(self, session_id: str, turn: Mapping[str, Any]) -> int:
        """
        Append one turn in one statement, and return the `seq` it was given.

        **Not part of the `Store` contract, and nothing calls it yet.** It is here because it is
        the write the schema was designed around, and without it the design's central claim is
        unreachable through the six methods: `update` still takes the whole list, so two writers
        can still compute the same tail. This takes one turn and lets the engine decide.

        Two races close here. `seq` comes from `max(seq) + 1` inside the insert, so a collision
        is a primary key violation and a retry rather than the file store's silent overwrite.
        And the insert selects from `sessions` with `ended_at is null`, so a turn arriving from
        a socket that has already closed inserts nothing -- where checking the doc first and
        inserting second lets a close and an append that arrive together both pass the check.
        That guard is the only reason `ended_at` is a column as well as a doc key.

        Zero rows means one of two things and the caller needs to know which, so the reason is
        looked up afterwards. That second query is on the failure path only, and a wrong answer
        there costs an error message, not a turn.
        """
        with self._lock:
            import psycopg

            values = ", ".join(f"%({name})s" for name in _TURN_COLUMNS)
            statement = (
                f"insert into turns ({', '.join(('session_id', 'seq', *_TURN_COLUMNS))}) "
                "select s.id, "
                "coalesce((select max(t.seq) + 1 from turns t where t.session_id = s.id), 0), "
                f"{values} from sessions s "
                "where s.id = %(_id)s and s.ended_at is null "
                "returning seq"
            )
            _validated_turns([turn])
            params: dict[str, Any] = {name: turn.get(name) for name in _TURN_COLUMNS}
            params["_id"] = session_id

            conn = self._connection()
            for attempt in range(APPEND_ATTEMPTS):
                try:
                    row = conn.execute(statement, params).fetchone()
                except psycopg.errors.UniqueViolation:
                    if attempt == APPEND_ATTEMPTS - 1:
                        raise
                    continue  # another writer took this seq; read a fresh max and try again
                if row is not None:
                    return int(row[0])
                ended = conn.execute(
                    "select ended_at from sessions where id = %s", (session_id,)
                ).fetchone()
                if ended is None:
                    raise NotFound(session_id)
                raise SessionEnded(f"session {session_id!r} ended at {ended[0]}")
            raise AssertionError("unreachable: the loop above either returns or raises")

    # -- the file store's own helpers, which must never run here ------------

    def _no_files(self) -> NoReturn:
        """
        The three inherited helpers that touch the filesystem, disabled.

        Reaching one would mean a `Store` method exists that this class forgot to override, and
        the symptom would be JSON files appearing beside a database that is meant to be
        authoritative -- a split-brain nobody notices until the two disagree. Failing loudly
        names the bug as a bug in this class rather than as a configuration problem.
        """
        raise RuntimeError(
            "PostgresStore reached the file store's path helpers, which means a Store method "
            "was not overridden. This is a bug in store_postgres.py, not a misconfiguration."
        )

    def _dir(self, collection: str) -> Path:
        self._no_files()

    def _file(self, collection: str, record_id: str) -> Path:
        self._no_files()

    def _write(self, collection: str, record: dict[str, Any]) -> None:
        self._no_files()
