#!/usr/bin/env python3
"""
Copy the JSON file store into PostgreSQL, once.

    pip install -e ".[postgres]"
    createdb nod
    psql -v ON_ERROR_STOP=1 -d nod -f migrations/001_initial.sql
    AVATAR_POSTGRES_DSN=postgresql:///nod python scripts/migrate_to_postgres.py

**Why a script and not a `Store` method.** This runs once per deployment, by a person, at a
moment when both backends exist and neither is authoritative yet. Putting it behind the `Store`
interface would mean every process that imports the store carries the import path of a one-off,
and the six-method surface would grow a seventh that no router ever calls.

**Why it does not apply the schema itself.** `migrations/001_initial.sql` is applied with
`psql -f` on purpose -- there is no migration framework here and the file says why. A script
that quietly created tables when they were missing would make "which schema is in this database"
unanswerable, because the answer would depend on which of two things ran last. So a missing
table is an error with the command to fix it, not a thing this repairs.

**What it does with a reference that no longer exists.** The file store never enforced one, so
some records point at ids that are gone -- `faces/` is empty on disk while an agent may still
name a face. The database will not accept those, and there are three possible responses: abort,
drop the record, or drop the reference. This drops the reference: the typed column is left null,
the warning names the record and the field, and the `doc` keeps the id it always had. Aborting
would make the migration impossible without editing historical data by hand, and dropping the
record would lose an agent because something it merely mentioned was deleted. Nulling the column
while the doc remembers is not a compromise invented here -- it is exactly the state the schema
already defines for `sessions.agent_id` after an agent is deleted, and the reader is already
required to tolerate it.

**Four properties, each because of a specific way this goes wrong:**

*Dependency order.* Leaf resources, then agents, then the join tables, then sessions, then
turns. Any other order and the foreign keys reject rows that are perfectly valid, and the error
would look like bad data rather than a bad sequence.

*One transaction.* A migration that half-succeeded is the worst outcome available: the database
looks populated, so the next person points the server at it, and the missing half surfaces as
individual 404s weeks later. Either all of it is there or none of it is.

*Idempotent.* Every insert upserts and turns are replaced per session, so a re-run converges on
the files rather than colliding with itself. The cost, stated plainly: this is a one-way import,
so a re-run silently discards anything written directly to the database since the last one. That
is the right default for a migration and the wrong one for a sync; if this ever needs to be a
sync, it needs a different name.

*Refuses a non-empty database.* Overwriting live rows with a snapshot of a file store somebody
forgot was stale is the mistake this prevents, and it is unrecoverable. `--force` allows it,
which is the point of asking.

Deliberately not here: no batching, no `COPY`, no progress bar. The whole data set is a few
hundred small documents and one `INSERT` per row is legible in a way a `COPY` stream is not. If
this ever takes longer than it takes to read the output, that is the point to revisit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from avatar.config import load_env

try:
    import psycopg
    from psycopg import sql
    from psycopg.types.json import Jsonb
except ModuleNotFoundError:  # pragma: no cover - operator error, not a code path
    sys.exit("needs the postgres extra: pip install -e '.[postgres]'")


DSN_ENV = "AVATAR_POSTGRES_DSN"
DEFAULT_DSN = "postgresql:///nod"
"""
Everything except the database name is left to libpq: local socket, current user, no password.

That is what a Homebrew or Debian default install gives you, so the common case needs no
configuration at all, and anything else -- a host, a password, a different name -- is a full DSN
in `AVATAR_POSTGRES_DSN`. A DSN carrying a password belongs in `.env.development`, which is
gitignored, which is why `load_env()` runs before this is read.
"""

DATA_DIR_ENV = "AVATAR_DATA_DIR"
DEFAULT_DATA_DIR = "data"
"""
Mirrors `store.DATA_ROOT` rather than importing it, and the reason is import order.

`store.DATA_ROOT` is evaluated when `avatar.store` is imported, so it would freeze before
`load_env()` had a chance to set `AVATAR_DATA_DIR` from a file -- pointing this at the wrong
directory and reporting a successful migration of nothing. Two constants that must agree is a
smaller problem than a silently empty run.
"""

LEAF_COLLECTIONS = ("faces", "rubrics", "guardrails", "pronunciations", "knowledge", "tools")
"""
Referenced by agents, referencing nothing. Insert order within the group does not matter; being
ahead of `agents` does.
"""

AGENT_REFERENCES = {
    "face_id": "faces",
    "rubric_id": "rubrics",
    "guardrail_id": "guardrails",
    "pronunciation_id": "pronunciations",
}

AGENT_LINKS = (
    ("knowledge_base_ids", "knowledge", "agent_knowledge_bases", "knowledge_id"),
    ("tool_ids", "tools", "agent_tools", "tool_id"),
)
"""(doc key, referenced collection, join table, referencing column) for the two m2m pairs."""

TURN_COLUMNS = (
    "epoch",
    "heard",
    "said",
    "transcribed",
    "interrupted",
    "llm_ttft_ms",
    "tts_first_audio_ms",
    "first_frame_ms",
    "perceived_total_ms",
)
"""
Every key a stored turn may carry. The `turns` table has no `doc` column, so a key that is not
in this tuple has nowhere to land -- hence the warning in `insert_turns` rather than a silent
drop. It is `Turn`'s field list; `extra="forbid"` on the model is what makes it complete.
"""

TABLES = (
    *LEAF_COLLECTIONS,
    "agents",
    "agent_knowledge_bases",
    "agent_tools",
    "sessions",
    "turns",
    "assistant_actions",
)
"""Insert order, and the order the report prints in. Also the list the emptiness check walks."""


@dataclass
class Report:
    """
    Rows written per table, and everything that was not quite right.

    Warnings accumulate rather than printing as they happen, so the count-per-table summary is
    not buried in them. They are printed after the commit -- a warning about a transaction that
    then rolled back would be describing something that never existed.
    """

    counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def wrote(self, table: str, rows: int = 1) -> None:
        self.counts[table] = self.counts.get(table, 0) + rows

    def warn(self, message: str) -> None:
        self.warnings.append(message)


# -- reading the file store -------------------------------------------------


def load_collection(root: Path, collection: str) -> list[dict[str, Any]]:
    """
    Every record in one collection directory, sorted by filename.

    Read directly rather than through `Store.list`, which is the one place this deliberately
    does not reuse the existing code. `Store.list` skips a file that fails to parse, so that one
    corrupt record cannot 500 a whole page -- correct for serving, wrong here: the file store
    keeps the file and the database would not, so "skipped" would mean "lost". An unreadable
    file stops the run instead, with the path, before anything is written.

    Sorted by filename for a deterministic order, not by `created_at`. Nothing downstream cares
    -- ordering comes from the `created_at` column -- but a stable order makes two runs' output
    diffable, which is how you tell a data change from a code change.
    """
    records: list[dict[str, Any]] = []
    for path in sorted((root / collection).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            sys.exit(f"{path}: cannot read ({exc}). Fix or move it; nothing has been written.")
        if not isinstance(record, dict):
            sys.exit(f"{path}: contains {type(record).__name__}, not a JSON object.")
        # The store addresses a record by filename and also writes the id into the body. If
        # those disagree, the record is served under an id it does not claim, and the schema's
        # `doc->>'id' = id` check would reject it with a constraint name and no path. Say which
        # file, and stop -- guessing which of the two ids is real would rename somebody's data.
        if record.get("id") != path.stem:
            sys.exit(
                f"{path}: body says id={record.get('id')!r} but the filename says "
                f"{path.stem!r}. The store reads it by filename; fix one of them."
            )
        records.append(record)
    return records


def required_instant(record: dict[str, Any], key: str, where: str, report: Report) -> datetime:
    """
    Parse `created_at` / `updated_at`, falling back to now with a warning.

    Preserving these is not cosmetic. `created_at` is what every list view sorts on, so letting
    the column default to `now()` would stamp every record with the migration time and collapse
    the console's ordering into whatever order this script happened to insert in -- a data loss
    that looks like a UI bug.
    """
    parsed = _instant(record.get(key))
    if parsed is not None:
        return parsed
    report.warn(f"{where}: {key}={record.get(key)!r} is not a timestamp; used the current time")
    return datetime.now(UTC)


def optional_instant(
    record: dict[str, Any], key: str, where: str, report: Report
) -> datetime | None:
    """A nullable timestamp. Absent and null are ordinary; present and unparseable is not."""
    raw = record.get(key)
    if raw is None:
        return None
    parsed = _instant(raw)
    if parsed is None:
        report.warn(f"{where}: {key}={raw!r} is not a timestamp; column left null")
    return parsed


def _instant(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def required_name(record: dict[str, Any], where: str, report: Report) -> str:
    """
    A non-null `name` for the tables that have one.

    Every create model refuses a blank name, so this should never fire. When it does, the empty
    string is the honest answer: inventing one from the id would put a value in the console that
    nobody typed, and a row findable only by id is a smaller problem than a row lying about what
    it is called.
    """
    name = record.get("name")
    if isinstance(name, str) and name.strip():
        return name
    report.warn(f"{where}: name={name!r}; stored as empty, so it will list as a blank row")
    return ""


def _number(raw: object) -> float | None:
    """A stage timing, or null. `bool` is excluded: it is an `int`, and it is not a latency."""
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        return None
    return float(raw)


# -- writing ----------------------------------------------------------------

LEAF_UPSERT = sql.SQL(
    """
    insert into {} (id, name, doc, created_at, updated_at)
    values (%s, %s, %s, %s, %s)
    on conflict (id) do update set
        name       = excluded.name,
        doc        = excluded.doc,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at
    """
)
"""
`do update`, not `do nothing`: a re-run should leave the database matching the files, and
`do nothing` would make a second run appear to succeed while silently keeping stale rows.

The table name is composed with `sql.Identifier` rather than interpolated into the string. The
names come from a constant in this file so there is no injection surface today -- the reason is
that a query built by `%`-formatting is a pattern that gets copied to somewhere the names are
not constants.
"""

AGENT_UPSERT = """
    insert into agents
        (id, name, face_id, rubric_id, guardrail_id, pronunciation_id, doc,
         created_at, updated_at)
    values (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    on conflict (id) do update set
        name             = excluded.name,
        face_id          = excluded.face_id,
        rubric_id        = excluded.rubric_id,
        guardrail_id     = excluded.guardrail_id,
        pronunciation_id = excluded.pronunciation_id,
        doc              = excluded.doc,
        created_at       = excluded.created_at,
        updated_at       = excluded.updated_at
"""

SESSION_UPSERT = """
    insert into sessions (id, agent_id, agent_name, ended_at, doc, created_at, updated_at)
    values (%s, %s, %s, %s, %s, %s, %s)
    on conflict (id) do update set
        agent_id   = excluded.agent_id,
        agent_name = excluded.agent_name,
        ended_at   = excluded.ended_at,
        doc        = excluded.doc,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at
"""

TURN_INSERT = """
    insert into turns
        (session_id, seq, epoch, heard, said, transcribed, interrupted,
         llm_ttft_ms, tts_first_audio_ms, first_frame_ms, perceived_total_ms)
    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

ACTION_UPSERT = """
    insert into assistant_actions (id, target, doc, created_at, updated_at)
    values (%s, %s, %s, %s, %s)
    on conflict (id) do update set
        target     = excluded.target,
        doc        = excluded.doc,
        created_at = excluded.created_at,
        updated_at = excluded.updated_at
"""


def existing_ids(cur: psycopg.Cursor[Any], table: str) -> set[str]:
    """
    The ids a foreign key will actually accept, read back after inserting that table.

    Read from the database rather than computed from the files, and the difference only shows up
    on a `--force` re-run: if the database holds a rubric that has since been deleted from disk,
    the foreign key accepts a reference to it and this must not warn about a reference that is
    fine. The constraint is the authority on what dangles, so ask it.
    """
    cur.execute(sql.SQL("select id from {}").format(sql.Identifier(table)))
    return {str(row[0]) for row in cur.fetchall()}


def insert_leaves(cur: psycopg.Cursor[Any], root: Path, report: Report) -> None:
    for collection in LEAF_COLLECTIONS:
        for record in load_collection(root, collection):
            where = f"{collection}/{record['id']}"
            cur.execute(
                LEAF_UPSERT.format(sql.Identifier(collection)),
                (
                    record["id"],
                    required_name(record, where, report),
                    Jsonb(record),
                    required_instant(record, "created_at", where, report),
                    required_instant(record, "updated_at", where, report),
                ),
            )
            report.wrote(collection)


def insert_agents(cur: psycopg.Cursor[Any], root: Path, report: Report) -> None:
    """
    Agents and their attachment rows, in one pass.

    The join tables are written here rather than in a second pass over the same records because
    they are meaningless without the agent row that precedes them, and a pass boundary between
    the two would be a place for the ordering to be got wrong later.
    """
    known = {name: existing_ids(cur, name) for name in LEAF_COLLECTIONS}

    for record in load_collection(root, "agents"):
        agent_id = record["id"]
        where = f"agents/{agent_id}"

        columns: dict[str, str | None] = {}
        for column, collection in AGENT_REFERENCES.items():
            target = record.get(column)
            if target and target not in known[collection]:
                report.warn(
                    f"{where}: {column} -> {target!r} is not in {collection}; column left "
                    f"null, doc keeps the id"
                )
                target = None
            columns[column] = target

        cur.execute(
            AGENT_UPSERT,
            (
                agent_id,
                required_name(record, where, report),
                columns["face_id"],
                columns["rubric_id"],
                columns["guardrail_id"],
                columns["pronunciation_id"],
                Jsonb(record),
                required_instant(record, "created_at", where, report),
                required_instant(record, "updated_at", where, report),
            ),
        )
        report.wrote("agents")
        insert_links(cur, record, known, report)


def insert_links(
    cur: psycopg.Cursor[Any],
    record: dict[str, Any],
    known: dict[str, set[str]],
    report: Report,
) -> None:
    """
    One agent's knowledge-base and tool attachments.

    Deleted and rewritten rather than upserted, because the doc's array is the authority: an id
    removed from `knowledge_base_ids` since the last run must lose its row, and an upsert alone
    would leave it attached for ever. The delete is scoped to this agent, so a re-run cannot
    touch anybody else's attachments.

    `on conflict do nothing` covers a duplicate id within one array. Nothing writes one, but the
    array is a list and the join table's key is a set, and the difference should not abort a
    migration.
    """
    for key, collection, table, column in AGENT_LINKS:
        cur.execute(
            sql.SQL("delete from {} where agent_id = %s").format(sql.Identifier(table)),
            (record["id"],),
        )
        statement = sql.SQL(
            "insert into {table} (agent_id, {column}) values (%s, %s) on conflict do nothing"
        ).format(table=sql.Identifier(table), column=sql.Identifier(column))
        for target in record.get(key) or []:
            if target not in known[collection]:
                report.warn(
                    f"agents/{record['id']}: {key} contains {target!r}, which is not in "
                    f"{collection}; attachment skipped, doc keeps the id"
                )
                continue
            cur.execute(statement, (record["id"], target))
            report.wrote(table)


def insert_sessions(cur: psycopg.Cursor[Any], root: Path, report: Report) -> None:
    """
    Sessions and their turns.

    `agent_name` is denormalised from the agent that is there now. For a session whose agent was
    deleted before this ran there is no name left to copy, and that is a real, permanent hole --
    the column exists so that a *future* delete does not create one, and it cannot recover the
    ones already made. Warned rather than passed over silently, because "(deleted agent)" in a
    report is a thing somebody will ask about.
    """
    cur.execute("select id, name from agents")
    agent_names = {str(row[0]): row[1] for row in cur.fetchall()}

    for record in load_collection(root, "sessions"):
        session_id = record["id"]
        where = f"sessions/{session_id}"

        agent_id = record.get("agent_id")
        if agent_id and agent_id not in agent_names:
            report.warn(
                f"{where}: agent_id -> {agent_id!r} is not in agents; column left null and "
                f"agent_name unrecoverable, doc keeps the id"
            )
            agent_id = None

        # Turns live in the `turns` table and the schema has a CHECK forbidding the key here, so
        # this is not merely tidiness -- leaving it in would abort the transaction.
        doc = {key: value for key, value in record.items() if key != "turns"}

        cur.execute(
            SESSION_UPSERT,
            (
                session_id,
                agent_id,
                agent_names.get(str(agent_id)) if agent_id else None,
                optional_instant(record, "ended_at", where, report),
                Jsonb(doc),
                required_instant(record, "created_at", where, report),
                required_instant(record, "updated_at", where, report),
            ),
        )
        report.wrote("sessions")
        insert_turns(cur, session_id, record.get("turns") or [], report)


def insert_turns(
    cur: psycopg.Cursor[Any],
    session_id: str,
    turns: list[dict[str, Any]],
    report: Report,
) -> None:
    """
    Explode one session's turn array into rows, `seq` being the array index.

    `seq` is arrival order and the array's order *is* arrival order, so the index is the right
    value and not a convenience. `epoch` is copied across as stored and is deliberately not used
    as the key: `sessions/sess_b391f845` in this repo's own data holds epochs `[0, 0, 1]`,
    because a turn the state machine refused leaves the counter where it was. An epoch-keyed
    table would drop one of those two, and it would be the interrupted one.

    Deleted and re-inserted per session for the same reason as the join tables, with one extra:
    `seq` is positional, so a session whose array got shorter would keep the tail of the
    previous run as rows that no file mentions.
    """
    cur.execute("delete from turns where session_id = %s", (session_id,))
    for seq, turn in enumerate(turns):
        stray = sorted(set(turn) - set(TURN_COLUMNS))
        if stray:
            # There is no `doc` column on `turns` to put these in, so they are being dropped.
            # Loudly, because the alternative is a field that existed on disk and does not exist
            # after the migration, with nothing in the output to say so.
            report.warn(
                f"sessions/{session_id} turn {seq}: no column for {', '.join(stray)}; dropped"
            )
        cur.execute(
            TURN_INSERT,
            (
                session_id,
                seq,
                int(turn.get("epoch") or 0),
                str(turn.get("heard") or ""),
                str(turn.get("said") or ""),
                bool(turn.get("transcribed")),
                bool(turn.get("interrupted")),
                _number(turn.get("llm_ttft_ms")),
                _number(turn.get("tts_first_audio_ms")),
                _number(turn.get("first_frame_ms")),
                _number(turn.get("perceived_total_ms")),
            ),
        )
        report.wrote("turns")


def insert_actions(cur: psycopg.Cursor[Any], root: Path, report: Report) -> None:
    """
    The audit trail. `target` is polymorphic and has no foreign key, so nothing here can dangle.

    It can, however, point at something already gone, and that is fine and expected -- an action
    about a deleted session is still a record of what the assistant did. There is nothing to
    check and nothing to warn about, which is worth saying so the absence of a check does not
    read as one that was forgotten.
    """
    for record in load_collection(root, "assistant_actions"):
        where = f"assistant_actions/{record['id']}"
        target = record.get("target")
        if not isinstance(target, str) or not target:
            # `target not null` in the schema, and an audit row that does not say what it is
            # about is not an audit row. Skipped rather than stored with an empty target,
            # which would show up in `history()` for nothing.
            report.warn(f"{where}: target={target!r}; record skipped")
            continue
        cur.execute(
            ACTION_UPSERT,
            (
                record["id"],
                target,
                Jsonb(record),
                required_instant(record, "created_at", where, report),
                required_instant(record, "updated_at", where, report),
            ),
        )
        report.wrote("assistant_actions")


def migrate(cur: psycopg.Cursor[Any], root: Path) -> Report:
    """Dependency order, top to bottom. Reordering these four calls breaks the foreign keys."""
    report = Report()
    insert_leaves(cur, root, report)
    insert_agents(cur, root, report)
    insert_sessions(cur, root, report)
    insert_actions(cur, root, report)
    return report


# -- driving it -------------------------------------------------------------


def row_counts(cur: psycopg.Cursor[Any]) -> dict[str, int]:
    """Rows per table before anything is written. Also proves the schema is there."""
    counts: dict[str, int] = {}
    for table in TABLES:
        cur.execute(sql.SQL("select count(*) from {}").format(sql.Identifier(table)))
        row = cur.fetchone()
        counts[table] = int(row[0]) if row else 0
    return counts


def print_report(report: Report, dsn: str) -> None:
    width = max(len(table) for table in TABLES)
    print(f"\n--- migrated into {dsn} ---")
    for table in TABLES:
        print(f"  {table:<{width}}  {report.counts.get(table, 0)}")
    if not report.warnings:
        return
    print(f"\n--- {len(report.warnings)} warning(s) ---")
    for warning in report.warnings:
        print(f"  {warning}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy the JSON file store in data/ into PostgreSQL."
    )
    parser.add_argument("--dsn", help=f"libpq connection string (default: ${DSN_ENV})")
    parser.add_argument(
        "--data-dir", type=Path, help=f"file store root (default: ${DATA_DIR_ENV})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="migrate into a database that already has rows, overwriting matching ids",
    )
    args = parser.parse_args(argv)

    # Same loader the server uses, so a DSN in `.env.development` works here with no `source`
    # step -- and so a DSN with a password in it stays in a gitignored file.
    load_env()
    dsn = args.dsn or os.environ.get(DSN_ENV) or DEFAULT_DSN
    root = args.data_dir or Path(os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR))

    if not root.is_dir():
        sys.exit(f"no file store at {root} -- run from apps/api, or pass --data-dir.")

    try:
        # psycopg opens a transaction on the first statement and this `with` commits it on a
        # clean exit, rolls it back on any exception, `SystemExit` included. That is the whole
        # implementation of "a partial migration cannot exist" -- there is no cleanup path to
        # get wrong, because there is nothing to clean up.
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            try:
                occupied = row_counts(cur)
            except psycopg.errors.UndefinedTable:
                sys.exit(
                    f"{dsn} has no schema. Apply it first:\n"
                    f"  psql -v ON_ERROR_STOP=1 -d <db> -f migrations/001_initial.sql"
                )
            if any(occupied.values()) and not args.force:
                listing = ", ".join(f"{t}={n}" for t, n in occupied.items() if n)
                sys.exit(
                    f"{dsn} is not empty ({listing}).\n"
                    "Refusing to overwrite it with a snapshot of the file store. Re-run with "
                    "--force if that is what you want, or use an empty database."
                )
            report = migrate(cur, root)
    except psycopg.OperationalError as exc:
        sys.exit(f"cannot connect to {dsn!r}: {exc}\nSet {DSN_ENV} or pass --dsn.")
    except psycopg.errors.IntegrityError as exc:
        # A constraint the file store never had. Nothing was written, so this is a data question
        # to answer in `data/` and then re-run, not a partial state to repair.
        sys.exit(f"the database rejected a row and the migration rolled back:\n  {exc}")

    print_report(report, dsn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
