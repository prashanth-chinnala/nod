"""
Proof that the Postgres store and the JSON-file store are the same store.

**What is under test is substitutability, not SQL.** `AVATAR_STORE=postgres` swaps the backing
of a surface that all eight routers call directly, and every one of them returns records
*exactly as the store wrote them* — no response model, deliberately, so that a field written by
a newer build cannot vanish from the console. That decision makes the store's return shape part
of the API's contract: a key the file store writes and Postgres does not (or the reverse —
`agent_name` is the tempting one) is a console field that appears or disappears when an operator
flips an environment variable, with no error anywhere. So most of these tests run the same call
against both backends and compare, rather than asserting against a shape written out here, which
would only prove that this file and one backend agree.

**Why the deliberate divergences are also tested.** Two behaviours are supposed to differ. A
dangling foreign key is accepted by the file store and refused by Postgres — that refusal is the
entire reason for the migration, so it gets a test that asserts the file store's acceptance too,
otherwise "both backends behave the same" would be an argument for undoing it. And a session
whose agent has been deleted keeps the dangling agent id in its doc while the typed column goes
null; a reader that helpfully overlaid the column onto the doc would rewrite history, which
`delete_agent` refuses to do.

**Why this skips instead of failing when there is no database.** CI has no Postgres and no
`psycopg` — `avatar` declares `dependencies = []`, and that emptiness is what lets the suite run
on a machine with no GPU and no services. A test that cannot run must therefore skip rather than
fail, and the cost of that is real and worth naming: on CI this file proves nothing, so a
Postgres regression is invisible until someone runs the suite on a machine with a database. The
one test here that needs no database (the read-modify-write demonstration) is written so it runs
everywhere, because it is the motivation for all the rest.

`pytest.importorskip("avatar.store_postgres")` also skips when the module has not been written
yet, which is the second cost: an ImportError inside a store that *does* exist reads here as
"no Postgres backend installed". Preferring that to a red CI is a judgment call; if this file
ever skips on a machine that has both psycopg and a database, the reason to check first is that
`avatar.store_postgres` failed to import.

State: one throwaway database per session, dropped in teardown; every table truncated between
tests. Per-test `createdb` was the other option and buys nothing over a truncate — it costs a
database creation per test and still leaks the database if the process is killed, which is the
only case a truncate does not cover either.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from avatar.store import NotFound, Store, now_iso

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def _migrations() -> list[Path]:
    """
    Every migration, in filename order.

    Globbed rather than naming `001_initial.sql`, which is what this was. A hardcoded
    filename means the throwaway database is built from an out-of-date schema the moment a
    second migration exists -- and the failure is not obviously about the fixture: it shows as
    `UndefinedColumn: column "voice_ref_id" of relation "agents" does not exist` from whichever
    test happens to insert first, which reads like a bug in the store.

    Filename order is the migration order, which is the convention the `NNN_` prefix exists to
    encode.
    """
    return sorted(MIGRATIONS_DIR.glob("*.sql"))

MAINTENANCE_DSN = "postgresql:///postgres"
"""
Where the throwaway database gets created.

No host, port or user: libpq fills those from `PGHOST`/`PGPORT`/`PGUSER` or its defaults, so the
same string works against the local socket and against a CI service container with those set,
without this file growing a configuration layer nothing has asked for yet.
"""


# -- reaching the backend ---------------------------------------------------------------
#
# Three separate reasons to skip, kept separate on purpose: "no driver", "no database" and "no
# backend module" are three different things to go and fix, and one combined skip message would
# send the reader to the wrong one.


def _psycopg() -> Any:
    return pytest.importorskip(
        "psycopg", reason='no Postgres driver: pip install "psycopg[binary]"'
    )


def _backend() -> Any:
    return pytest.importorskip(
        "avatar.store_postgres", reason="no Postgres backend importable in this environment"
    )


def _store_class() -> type[Any]:
    """
    Find the store class in `avatar.store_postgres`.

    Named-first rather than "the only class in the module", because a module is allowed to
    define helpers, and a scan that happened to find one would bind these tests to whichever
    class was defined first. `__module__` is checked so a re-exported `avatar.store.Store` —
    which would make every test below pass by testing the file store twice — cannot be picked
    up.
    """
    module = _backend()
    for name in ("PostgresStore", "PgStore", "Store"):
        candidate = getattr(module, name, None)
        if isinstance(candidate, type) and candidate.__module__ == module.__name__:
            return candidate
    defined = sorted(
        name
        for name, value in vars(module).items()
        if isinstance(value, type)
        and name.endswith("Store")
        and value.__module__ == module.__name__
    )
    if len(defined) == 1:
        return getattr(module, defined[0])  # type: ignore[no-any-return]
    pytest.fail(
        f"{module.__name__} defines no obvious store class (found {defined or 'none'}); "
        "these tests need one to compare against the file store"
    )


def _make_store(dsn: str) -> Any:
    """
    Construct the backend against `dsn`, positionally or by keyword.

    Two attempts and no more. If the real constructor takes something else — a connection pool,
    a config object — this raises the `TypeError` rather than skipping, because a backend that
    cannot be pointed at a test database is a finding, not an absent dependency.
    """
    cls = _store_class()
    try:
        return cls(dsn)
    except TypeError:
        return cls(dsn=dsn)


def _sql(dsn: str, statement: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    """
    Run one statement on its own connection, outside whatever the store is doing.

    Used for the two assertions that cannot be made through the store surface: that a typed
    column mirrors its doc key, and that the reference the database is enforcing is the one the
    doc claims. Asking the store to prove that would be asking it to confirm its own
    bookkeeping.
    """
    psycopg = _psycopg()
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(statement, params)
        return list(cur.fetchall()) if cur.description else []


@pytest.fixture(scope="session")
def postgres() -> Iterator[str]:
    """
    A database that exists only for this test session, with the schema applied.

    Session-scoped because applying the migration is the slow part and it is identical every
    time; isolation between tests comes from truncating, which is cheap.

    `WITH (FORCE)` on the drop is not optional: a store that holds an open connection would
    otherwise make the drop fail and leave the database behind — exactly the state this fixture
    promises not to leave.
    """
    psycopg = _psycopg()
    if not _migrations():
        pytest.skip(f"no schema to apply: {MIGRATIONS_DIR} has no .sql files")

    try:
        admin = psycopg.connect(MAINTENANCE_DSN, autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"no Postgres reachable at {MAINTENANCE_DSN}: {exc}")

    # pid in the name so two suites running at once cannot pick the same database, and a leaked
    # one names the process that leaked it.
    database = f"nod_test_{os.getpid()}_{uuid4().hex[:8]}"
    with admin:
        admin.execute(f'create database "{database}"')
        dsn = f"postgresql:///{database}"
        try:
            with psycopg.connect(dsn, autocommit=True) as conn:
                for migration in _migrations():
                    conn.execute(migration.read_text(encoding="utf-8"))
            yield dsn
        finally:
            admin.execute(f'drop database if exists "{database}" with (force)')


@pytest.fixture
def pg_store(postgres: str) -> Iterator[Callable[[], Any]]:
    """
    A factory for stores against the session database, with the tables emptied first.

    A factory rather than one instance because the concurrency test needs two independent stores
    — that is the point of it: two `uvicorn` workers are two processes with two connections, and
    a single shared store object would serialise them and prove nothing.
    """
    _truncate(postgres)
    made: list[Any] = []

    def build() -> Any:
        store = _make_store(postgres)
        made.append(store)
        return store

    yield build

    for store in made:
        closer = getattr(store, "close", None)
        if callable(closer):
            closer()


def _truncate(dsn: str) -> None:
    """
    Empty every table, whatever the tables happen to be.

    Read from the catalogue rather than listed here so a collection added to the schema is
    cleaned automatically; a hard-coded list would leave the new table's rows behind and the
    failure would land in an unrelated test.

    `lock_timeout` is what turns a store that left a transaction open into a five-second error
    naming this line, instead of a suite that hangs with no output.
    """
    psycopg = _psycopg()
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("set lock_timeout = '5s'")
        cur.execute("select tablename from pg_tables where schemaname = 'public'")
        tables = [row[0] for row in cur.fetchall()]
        if tables:
            quoted = ", ".join(f'"{name}"' for name in tables)
            cur.execute(f"truncate table {quoted} cascade")


@pytest.fixture
def pg(pg_store: Callable[[], Any]) -> Any:
    return pg_store()


@pytest.fixture
def fs(tmp_path: Path) -> Store:
    """The file store, on a throwaway directory. The reference implementation, not a double."""
    return Store(tmp_path)


# -- realistic bodies ------------------------------------------------------------------
#
# Copied from the routers' own create paths and from records in `data/`, not invented. A body
# with one key would pass a shape comparison while proving nothing about nested structures, and
# the nested structures are where the two backends have room to disagree: `parameters_schema` is
# a user-supplied JSON Schema, `detail` varies by kind, `competencies` is order-sensitive.

BODIES: dict[str, tuple[str, dict[str, Any]]] = {
    "faces": (
        "face",
        {
            "name": "Reference clip",
            "reference_path": "media/reference.mp4",
            "status": "queued",
            "reason": None,
            "enrollment_ms": None,
            "frame_count": None,
        },
    ),
    "rubrics": (
        "rubric",
        {
            "name": "Data platform",
            "competencies": [
                {"id": "scale", "name": "Scale", "signals": ["sharding", "partitioning"]},
                {"id": "debugging", "name": "Debugging", "signals": ["bisect"]},
            ],
        },
    ),
    "guardrails": (
        "guard",
        {
            "name": "Default policy",
            "banned_topics": ["salary"],
            "max_answer_chars": 600,
            "on_violation": "refuse",
            "pii_redaction": False,
            "refusal_message": "Not that.",
        },
    ),
    "pronunciations": (
        "lex",
        {
            "name": "Engineering lexicon",
            "entries": [
                {"term": "PostgreSQL", "say": "post-gress-cue-ell"},
                {"term": "nginx", "say": "engine ex"},
            ],
        },
    ),
    "knowledge": (
        "kb",
        {
            "name": "Platform docs",
            "description": "",
            "documents": [],
            "chunks": [],
            "chunk_count": 0,
            "total_chars": 0,
        },
    ),
    "tools": (
        "tool",
        {
            "name": "lookup_order",
            "description": "Fixed value, for testing.",
            "kind": "builtin",
            "enabled": True,
            "timeout_ms": 1500,
            "url": None,
            "parameters_schema": {"type": "object", "properties": {}},
        },
    ),
    "agents": (
        "agent",
        {
            "name": "Data platform interviewer",
            "system_prompt": "",
            "llm_provider": "scripted",
            "llm_model": "",
            "voice_provider": "tone",
            "voice_id": "",
            "face_id": None,
            "knowledge_base_ids": [],
            "tool_ids": [],
            "guardrail_id": None,
            "rubric_id": None,
            "pronunciation_id": None,
            "turn_taking": {
                "onset_probability": 0.6,
                "release_probability": 0.35,
                "onset_frames": 3,
                "min_speech_ms": 200,
                "end_of_turn_silence_ms": 700,
            },
        },
    ),
    "assistant_actions": (
        "act",
        {
            "actor": "prashanth",
            "kind": "flagged_for_review",
            "status": "proposed",
            "summary": "Competency 'Debugging' was rated no_evidence.",
            "target": "sess_b391f845",
            "detail": {"session_id": "sess_b391f845"},
            "proposed_at": "2026-07-29T20:19:37+00:00",
            "decided_at": None,
            "decided_by": None,
            "via": "assistant",
        },
    ),
    "sessions": ("sess", {}),  # built by _session_body; it needs a fresh timestamp
}


def _session_body(agent_id: str | None = None) -> dict[str, Any]:
    """Exactly what `create_session` sends, `turns: []` included — see the turns test."""
    return {
        "agent_id": agent_id,
        "started_at": now_iso(),
        "ended_at": None,
        "turns": [],
        "stale_dropped": 0,
        "frames_repeated": 0,
    }


TURN: dict[str, Any] = {
    "epoch": 1,
    "heard": "We shipped a queue-backed ingest that assumed ordering we never had.",
    "said": "How did you discover the ordering assumption broke?",
    "transcribed": True,
    "llm_ttft_ms": 2942.0,
    "tts_first_audio_ms": 956.0,
    "first_frame_ms": 4136.0,
    "perceived_total_ms": 4161.0,
    "interrupted": False,
    "silent": False,
}

VOLATILE = ("id", "created_at", "updated_at")
"""The three keys the store generates, so they cannot be compared by value across backends."""


def assert_interchangeable(pg_record: dict[str, Any], fs_record: dict[str, Any]) -> None:
    """
    The same keys, the same body, and generated values in the same format.

    The format check is not pedantry. `created_at` is compared as a string by
    `Store.list`, rendered as-is by the console, and used by `list` to order rows —
    `2026-07-29 18:11:55+00` and `2026-07-29T18:11:55+00:00` are the same instant, sort
    differently, and only one of them is what the file store writes.
    """
    assert sorted(pg_record) == sorted(fs_record), "different keys between backends"

    body = {k: v for k, v in pg_record.items() if k not in VOLATILE}
    assert body == {k: v for k, v in fs_record.items() if k not in VOLATILE}

    assert pg_record["id"].split("_")[0] == fs_record["id"].split("_")[0], "different id prefix"
    for key in ("created_at", "updated_at"):
        stamp = pg_record[key]
        assert isinstance(stamp, str), f"{key} is {type(stamp).__name__}, not a string"
        parsed = datetime.fromisoformat(stamp)
        assert parsed.tzinfo is not None, f"{key} has no offset: {stamp!r}"
        assert parsed.isoformat() == stamp, f"{key} is not the file store's format: {stamp!r}"


def _body_for(collection: str) -> tuple[str, dict[str, Any]]:
    prefix, body = BODIES[collection]
    return prefix, _session_body() if collection == "sessions" else dict(body)


# -- shape parity, every collection ----------------------------------------------------


@pytest.mark.parametrize("collection", sorted(BODIES))
def test_create_returns_the_same_record_as_the_file_store(
    pg: Any, fs: Store, collection: str
) -> None:
    """
    Prevents a console field that appears or disappears when `AVATAR_STORE` is flipped.

    Every router returns the store's record verbatim, so an extra key (the denormalised
    `agent_name` is the one that wants to leak out) or a missing one is an API change made by an
    environment variable. Runs over all nine collections because a collection whose table was
    never created fails here rather than in whichever test happened to touch it first.
    """
    prefix, body = _body_for(collection)

    assert_interchangeable(
        pg.create(collection, prefix, body), fs.create(collection, prefix, body)
    )


@pytest.mark.parametrize("collection", sorted(BODIES))
def test_get_returns_exactly_what_create_returned(pg: Any, collection: str) -> None:
    """
    Prevents a write path and a read path that disagree.

    The routers use both interchangeably — `POST` returns the created record and the console
    immediately re-reads it — so a field that survives the insert but is dropped, reordered into
    a different type, or reformatted on the way back out shows up as a form that loses a value
    the moment it is saved.
    """
    prefix, body = _body_for(collection)
    created = pg.create(collection, prefix, body)

    assert pg.get(collection, created["id"]) == created


def test_list_matches_the_file_store(pg: Any, fs: Store) -> None:
    """
    Prevents a list view that renders different fields from the detail view.

    `list` is a separate query from `get` in SQL and the same file read in the file store, which
    is exactly why it can drift: a `select id, name` for the list page would return rows the
    console renders with everything else blank.
    """
    for name in ("First", "Second"):
        prefix, body = _body_for("faces")
        pg.create("faces", prefix, {**body, "name": name})
        fs.create("faces", prefix, {**body, "name": name})

    listed = pg.list("faces")
    reference = fs.list("faces")

    assert len(listed) == len(reference) == 2
    by_name = {row["name"]: row for row in listed}
    for row in reference:
        assert_interchangeable(by_name[row["name"]], row)


def test_iter_all_yields_what_list_returns(pg: Any) -> None:
    """
    Prevents the two read-everything paths diverging.

    `iter_all` exists so a caller can stream instead of materialising, and the file store
    implements it by delegating to `list`. A SQL backend has a real reason to implement it
    separately — a server-side cursor — and that is the version that can drift.
    """
    prefix, body = _body_for("tools")
    pg.create("tools", prefix, body)

    assert list(pg.iter_all("tools")) == pg.list("tools")


def test_list_is_newest_first(pg: Any, fs: Store) -> None:
    """
    Prevents a just-created resource being buried at the bottom of the console's list.

    The real 1.05-second sleeps are the cost of not being able to fake the clock here.
    `test_api_faces.py` monkeypatches `avatar.store.now_iso` for the same assertion, which only
    works because the file store calls it through the module global — a SQL backend may take its
    ordering key from `now()`, which no monkeypatch reaches. Sleeping past a second boundary is
    the one approach that holds whichever source the timestamp comes from.
    """
    created: list[dict[str, Any]] = []
    for index, name in enumerate(("Oldest", "Middle", "Newest")):
        prefix, body = _body_for("rubrics")
        created.append(pg.create("rubrics", prefix, {**body, "name": name}))
        fs.create("rubrics", prefix, {**body, "name": name})
        if index < 2:
            time.sleep(1.05)  # `created_at` has second precision; a tie has no defined order

    listed = pg.list("rubrics")

    assert [row["name"] for row in listed] == ["Newest", "Middle", "Oldest"]
    assert [row["name"] for row in listed] == [row["name"] for row in fs.list("rubrics")]
    assert [row["id"] for row in listed] == [row["id"] for row in reversed(created)]


# -- missing ids -----------------------------------------------------------------------


def test_a_missing_id_raises_the_same_notfound_class(pg: Any, fs: Store) -> None:
    """
    Prevents every 404 in the console becoming a 500.

    Each router catches `avatar.store.NotFound` by name and maps it to a 404. A backend that
    raised its own not-found type — or let psycopg's "no rows" surface — would turn "you
    followed a stale link" into "the server is broken", and the operator's next step would be to
    read a traceback instead of the page.
    """
    for missing in ("face_nope", "../../etc/passwd"):
        with pytest.raises(NotFound):
            pg.get("faces", missing)
        with pytest.raises(NotFound):
            fs.get("faces", missing)

        with pytest.raises(NotFound):
            pg.update("faces", missing, {"name": "x"})
        with pytest.raises(NotFound):
            fs.update("faces", missing, {"name": "x"})

        with pytest.raises(NotFound):
            pg.delete("faces", missing)
        with pytest.raises(NotFound):
            fs.delete("faces", missing)


def test_delete_removes_it_from_get_and_from_list(pg: Any) -> None:
    """
    Prevents a delete that only unlinks one of the two read paths.

    A row soft-deleted for the detail view but still returned by `list` is a resource the
    console shows and cannot open, which reads as data corruption rather than as a deletion.
    """
    prefix, body = _body_for("guardrails")
    created = pg.create("guardrails", prefix, body)

    pg.delete("guardrails", created["id"])

    with pytest.raises(NotFound):
        pg.get("guardrails", created["id"])
    assert pg.list("guardrails") == []


# -- the merge semantics ---------------------------------------------------------------


def test_a_patch_setting_a_key_to_null_clears_it(pg: Any, fs: Store) -> None:
    """
    Prevents the regression the file store was fixed for: a picker that cannot be cleared.

    Dropping `None` from a patch makes "not sent" and "explicitly cleared" the same request, so
    detaching a rubric from an agent silently does nothing while the console shows a control
    that appears to work. `doc || '{"rubric_id": null}'` preserves the fix because jsonb
    concatenation sets a key to null rather than removing it — which is the specific reason the
    design says to merge with `||` and not to strip nulls first.

    The key must still be *present* and null, not absent: the console distinguishes "no rubric"
    from a field it was never told about.
    """
    rubric_prefix, rubric_body = _body_for("rubrics")
    rubric = pg.create("rubrics", rubric_prefix, rubric_body)
    prefix, body = _body_for("agents")
    pg_agent = pg.create("agents", prefix, {**body, "rubric_id": rubric["id"]})
    fs_agent = fs.create("agents", prefix, {**body, "rubric_id": rubric["id"]})

    patched = pg.update("agents", pg_agent["id"], {"rubric_id": None})
    reference = fs.update("agents", fs_agent["id"], {"rubric_id": None})

    assert "rubric_id" in patched, "the key was dropped, not cleared"
    assert patched["rubric_id"] is None
    assert_interchangeable(patched, reference)
    assert pg.get("agents", pg_agent["id"])["rubric_id"] is None


def test_a_patch_leaves_keys_it_does_not_mention_alone(pg: Any, fs: Store) -> None:
    """
    Prevents a PATCH that behaves like a PUT and blanks the rest of the form.

    Every router dumps its body with `exclude_unset=True`, so the store receives only the keys
    the client actually sent. A backend that replaced `doc` instead of merging it would erase
    every field the operator did not happen to edit — and it would look like it worked.
    """
    prefix, body = _body_for("tools")
    pg_tool = pg.create("tools", prefix, body)
    fs_tool = fs.create("tools", prefix, body)

    patched = pg.update("tools", pg_tool["id"], {"enabled": False})
    reference = fs.update("tools", fs_tool["id"], {"enabled": False})

    assert patched["enabled"] is False
    assert patched["parameters_schema"] == body["parameters_schema"]
    assert patched["timeout_ms"] == body["timeout_ms"]
    assert patched["url"] is None
    assert_interchangeable(patched, reference)


def test_a_patch_replaces_a_nested_value_whole(pg: Any, fs: Store) -> None:
    """
    Prevents a deep merge, which would make an invariant uncheckable.

    `AgentUpdate` sends `turn_taking` whole precisely because the hysteresis rule spans two of
    its fields: a backend that merged one level deeper would let `{"release_probability": 0.9}`
    land against a stored `onset_probability` of 0.6 and store an inverted pair that
    `TurnTaking` had never seen. jsonb `||` is a shallow merge, which is the behaviour required
    here — and the schema's CHECK is the backstop for exactly this.
    """
    prefix, body = _body_for("agents")
    replacement = {
        "onset_probability": 0.7,
        "release_probability": 0.4,
        "onset_frames": 2,
        "min_speech_ms": 150,
        "end_of_turn_silence_ms": 500,
    }
    pg_agent = pg.create("agents", prefix, body)
    fs_agent = fs.create("agents", prefix, body)

    patched = pg.update("agents", pg_agent["id"], {"turn_taking": replacement})
    reference = fs.update("agents", fs_agent["id"], {"turn_taking": replacement})

    assert patched["turn_taking"] == replacement
    assert_interchangeable(patched, reference)


def test_a_patch_cannot_change_an_id_on_either_backend(pg: Any, fs: Store) -> None:
    """
    Prevents one resource silently becoming another.

    No router's patch model has an `id` field, so this cannot arrive over HTTP — the store is
    the layer that has to hold the rule, because the assistant and the scripts call it directly.
    On Postgres it is enforced twice: the store drops the key, and the schema's
    `doc->>'id' = id` CHECK means a merged-in id would be an error rather than a record that
    serves itself under one id while claiming another.
    """
    prefix, body = _body_for("faces")
    pg_face = pg.create("faces", prefix, body)
    fs_face = fs.create("faces", prefix, body)
    hostile = {"id": "face_hijacked", "name": "Kept"}

    patched = pg.update("faces", pg_face["id"], dict(hostile))
    reference = fs.update("faces", fs_face["id"], dict(hostile))

    assert patched["id"] == pg_face["id"]
    assert patched["name"] == "Kept", "the legitimate part of the patch was dropped too"
    assert patched["updated_at"] >= patched["created_at"]
    assert_interchangeable(patched, reference)
    assert pg.get("faces", pg_face["id"])["id"] == pg_face["id"]
    with pytest.raises(NotFound):
        pg.get("faces", "face_hijacked")


def test_created_at_is_the_one_divergence_in_the_merge(
    pg: Any, fs: Store, postgres: str
) -> None:
    """
    Pins the single place the two backends do not behave the same, so it stays a decision.

    `Store.update`'s docstring says "`id` and `created_at` are immutable" and its code enforces
    only the first: `record.update(patch)` then re-pins the id, so a patch carrying `created_at`
    rewrites it on the file store. `PostgresStore.update` drops both, and argues for the
    stricter half — `created_at` is what `list` orders on and what the typed column records as
    the true insert time, so honouring such a patch would reorder the console and leave the two
    copies permanently out of step.

    Recorded as an assertion rather than left implicit because a divergence nobody wrote down is
    how the two backends stop being interchangeable one key at a time. This one is out of reach
    of the API: every patch model is `extra="forbid"` and none declares `created_at`, so nothing
    an operator can do reaches it. The callers that bypass the routers — the assistant, the
    scripts — are what make it worth asserting at all.

    Either half changing breaks this test on purpose. If the file store starts protecting the
    key, the backends agree and the docstring becomes true; if Postgres stops, list ordering
    becomes patchable. Both are decisions for the human, and both should fail a test first.
    """
    prefix, body = _body_for("faces")
    pg_face = pg.create("faces", prefix, body)
    fs_face = fs.create("faces", prefix, body)
    backdated = {"created_at": "1999-01-01T00:00:00+00:00", "name": "Kept"}

    patched = pg.update("faces", pg_face["id"], dict(backdated))
    reference = fs.update("faces", fs_face["id"], dict(backdated))

    assert patched["created_at"] == pg_face["created_at"], "a patch moved created_at"
    assert reference["created_at"] == "1999-01-01T00:00:00+00:00", (
        "the file store now protects created_at too — the divergence is gone and its docstring "
        "is finally true; delete this test and fold the key back into the id test above"
    )
    assert patched["name"] == reference["name"] == "Kept", "the rest of the patch was dropped"
    assert sorted(patched) == sorted(reference), "the divergence must be one key's value"

    column = _sql(postgres, "select created_at from faces where id = %s", (pg_face["id"],))
    assert column[0][0].year != 1999, "the typed column no longer holds the true insert time"


def test_a_patched_reference_updates_the_typed_column_too(pg: Any, postgres: str) -> None:
    """
    Prevents a foreign key that is enforced against a value nothing reads.

    The doc is what the API returns; the typed column is what the database constrains. If a
    patch updates only the doc, the column keeps pointing at the old rubric — so `ON DELETE
    RESTRICT` protects a reference the agent no longer has, and lets the operator delete the
    rubric it actually uses. Nothing would fail until a candidate connected. Checked in SQL
    because this is the one property the store cannot be asked to confirm about itself.
    """
    rubric_prefix, rubric_body = _body_for("rubrics")
    first = pg.create("rubrics", rubric_prefix, {**rubric_body, "name": "First"})
    second = pg.create("rubrics", rubric_prefix, {**rubric_body, "name": "Second"})
    prefix, body = _body_for("agents")
    agent = pg.create("agents", prefix, {**body, "rubric_id": first["id"]})

    patched = pg.update("agents", agent["id"], {"rubric_id": second["id"]})

    assert patched["rubric_id"] == second["id"]
    columns = _sql(
        postgres,
        "select rubric_id, doc->>'rubric_id' from agents where id = %s",
        (agent["id"],),
    )
    assert columns == [(second["id"], second["id"])], "the column and the doc disagree"


# -- the constraints the migration exists for ------------------------------------------


def test_a_reference_to_a_missing_record_is_refused(pg: Any, fs: Store) -> None:
    """
    The reason for the migration: prevents a dangling rubric id surfacing as a session that
    will not start when a candidate joins.

    This is a deliberate divergence, so the file store's acceptance is asserted alongside it —
    if both backends refused, this test would be an argument that the foreign keys changed
    nothing, and if both accepted, the schema is not doing the one job it was added for.

    The exception type is deliberately not pinned. Whether the store translates psycopg's
    `ForeignKeyViolation` into something of its own is its decision; what must hold is that the
    write fails and nothing is stored — a rejected create that left a row behind would be worse
    than either outcome.
    """
    prefix, body = _body_for("agents")
    dangling = {**body, "rubric_id": "rubric_does_not_exist"}

    assert fs.create("agents", prefix, dict(dangling))["rubric_id"] == "rubric_does_not_exist"

    try:
        pg.create("agents", prefix, dict(dangling))
    except Exception:
        pass  # the type is the store's business; the effect below is the contract
    else:
        pytest.fail("Postgres accepted an agent pointing at a rubric that does not exist")

    assert pg.list("agents") == [], "the refused agent was stored anyway"

    # Control: a store that refused every agent would pass everything above.
    rubric_prefix, rubric_body = _body_for("rubrics")
    rubric = pg.create("rubrics", rubric_prefix, rubric_body)
    accepted = pg.create("agents", prefix, {**body, "rubric_id": rubric["id"]})
    assert accepted["rubric_id"] == rubric["id"]


def test_deleting_an_agent_keeps_the_transcript_and_its_dangling_id(
    pg: Any, postgres: str
) -> None:
    """
    Prevents a tidy-up rewriting history.

    `delete_agent` refuses to cascade into sessions on the grounds that a transcript which no
    longer says which agent produced it is worth less than a dangling id. `ON DELETE SET NULL`
    keeps the delete possible; the doc keeps the id. The column and the doc are supposed to
    disagree here, and this test exists so that a future reader who notices the disagreement
    finds it asserted on purpose rather than "fixes" it by overlaying the column onto the doc —
    which would silently blank the agent id on every report of a deleted agent's interviews.
    """
    prefix, body = _body_for("agents")
    agent = pg.create("agents", prefix, body)
    session = pg.create("sessions", "sess", _session_body(agent_id=agent["id"]))

    pg.delete("agents", agent["id"])

    kept = pg.get("sessions", session["id"])
    assert kept["agent_id"] == agent["id"], "the transcript lost its record of what ran it"
    assert _sql(postgres, "select agent_id from sessions where id = %s", (session["id"],)) == [
        (None,)
    ], "the foreign key column should be null once the agent is gone"


# -- sessions: turns, and the write pattern that motivated all of this -----------------


def test_a_session_round_trips_the_routers_own_create_body(pg: Any, fs: Store) -> None:
    """
    Prevents `POST /sessions` 500ing on the backend it was not developed against.

    `create_session` sends `turns: []` as part of the body, and the schema forbids a `turns` key
    in the session's doc — so the store has to route that array to the child table rather than
    merge it. A backend that passes every other test can still fail here, on the very first call
    the runtime makes when a candidate connects.
    """
    body = _session_body()

    created = pg.create("sessions", "sess", dict(body))
    reference = fs.create("sessions", "sess", dict(body))

    assert created["turns"] == []
    assert_interchangeable(created, reference)
    assert pg.get("sessions", created["id"]) == created


def test_turns_survive_the_update_call_the_router_actually_makes(pg: Any, fs: Store) -> None:
    """
    Prevents an appended turn being dropped, reordered, or losing its timings.

    `append_turn` reads the array, appends and calls `update(..., {"turns": turns})`, so this is
    the exact call the child table has to serve. Both turns here carry `epoch: 1`, which is not
    a mistake: a barge-in during `THINKING` leaves the epoch where it was, so two turns in one
    session legitimately share one — and a child table keyed on `(session_id, epoch)` would
    reject the second, losing precisely the interrupted turns that the latency figures are drawn
    from.

    The interrupted turn also carries three null timings, which is a real state and not missing
    data: the turn was cut off after the LLM and before anything else.
    """
    body = _session_body()
    pg_session = pg.create("sessions", "sess", dict(body))
    fs_session = fs.create("sessions", "sess", dict(body))
    interrupted = {
        **TURN,
        "heard": "Actually, let me start again.",
        "said": "",
        "interrupted": True,
        "tts_first_audio_ms": None,
        "first_frame_ms": None,
        "perceived_total_ms": None,
    }

    for turns in ([TURN], [TURN, interrupted]):
        patched = pg.update("sessions", pg_session["id"], {"turns": list(turns)})
        reference = fs.update("sessions", fs_session["id"], {"turns": list(turns)})
        assert patched["turns"] == list(turns)
        assert_interchangeable(patched, reference)

    stored = pg.get("sessions", pg_session["id"])["turns"]
    assert [turn["epoch"] for turn in stored] == [1, 1], "a same-epoch turn was lost"
    assert stored[1]["first_frame_ms"] is None
    assert stored[0]["perceived_total_ms"] == 4161.0


def test_two_concurrent_appends_both_land_and_neither_wins(
    pg_store: Callable[[], Any], postgres: str
) -> None:
    """
    Prevents the lost turn, on the one method that can actually prevent it.

    `append_turn` is Postgres-only — the file store has no such method, so this is the one test
    here that is not a parity test, and it is included because the turn append is the race the
    schema was designed around. Two writers appending at once through `update(turns=[...])`
    still lose one, because that call sends a whole array assembled in Python; only an insert
    that allocates `seq` inside the statement can be safe, and that is what this exercises.

    Both turns carry `epoch: 1` — see the test above for why that is legitimate — so this also
    demonstrates why the key is `(session_id, seq)`: two rows with one epoch, both kept, in
    arrival order.

    The ended-session half is the same race in the guard rather than in the insert: a turn from
    a socket that closed a moment ago must not land after `ended_at`, or the record claims a
    conversation continued past its own end.
    """
    writer_one, writer_two = pg_store(), pg_store()
    session = writer_one.create("sessions", "sess", _session_body())
    barrier = threading.Barrier(2)

    def append(store: Any, heard: str) -> Any:
        barrier.wait(timeout=10)
        return store.append_turn(session["id"], {**TURN, "heard": heard})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(append, writer_one, "first writer"),
            pool.submit(append, writer_two, "second writer"),
        ]
        sequences = sorted(future.result(timeout=30) for future in futures)

    assert sequences == [0, 1], f"two appends produced {sequences}, so one overwrote the other"
    stored = writer_one.get("sessions", session["id"])["turns"]
    assert sorted(turn["heard"] for turn in stored) == ["first writer", "second writer"]
    assert [turn["epoch"] for turn in stored] == [1, 1]
    counted = _sql(
        postgres, "select count(*) from turns where session_id = %s", (session["id"],)
    )
    assert counted == [(2,)]

    writer_one.update("sessions", session["id"], {"ended_at": now_iso()})
    try:
        writer_two.append_turn(session["id"], {**TURN, "heard": "too late"})
    except Exception:
        pass  # the type is the store's business; what matters is that nothing was written
    else:
        pytest.fail("a turn was appended to a session that had already ended")

    assert len(writer_one.get("sessions", session["id"])["turns"]) == 2


def test_two_overlapping_patches_to_one_session_both_survive(
    pg_store: Callable[[], Any], postgres: str
) -> None:
    """
    The concurrency case the whole design is for: prevents one writer's field vanishing because
    another writer patched a different field at the same moment.

    Three writers touch a live session — the recording hook from `/rtc`, the coverage snapshot,
    and the turn appender — and today they are safe only by accident of being one process.
    Nobody would see the first `uvicorn --workers 2` break it: a lost `recording` block looks
    like a session that was never recorded.

    Two stores, not one shared store, because two workers are two connections; a shared store
    object could serialise the writes internally and the test would pass without proving
    anything. Ten rounds, asserted after each, because a single overlap can interleave
    harmlessly by luck — a read-modify-write backend fails a round quickly, and a merge in one
    statement cannot fail any of them.
    """
    left, right = pg_store(), pg_store()
    session = pg_store().create("sessions", "sess", _session_body())
    barrier = threading.Barrier(2)

    def patch(store: Any, key: str, value: Any) -> None:
        barrier.wait(timeout=10)
        store.update("sessions", session["id"], {key: value})

    with ThreadPoolExecutor(max_workers=2) as pool:
        for round_number in range(10):
            coverage = {"round": round_number, "competencies": [{"id": "scale", "rated": "ok"}]}
            recording = {"round": round_number, "status": "active"}
            futures = [
                pool.submit(patch, left, "coverage", coverage),
                pool.submit(patch, right, "recording", recording),
            ]
            for future in futures:
                future.result(timeout=30)

            stored = left.get("sessions", session["id"])
            assert stored["coverage"] == coverage, f"round {round_number} lost coverage"
            assert stored["recording"] == recording, f"round {round_number} lost recording"

    # The fields the create wrote are still there: a merge that survived the races must not have
    # been a whole-doc replace that happened to win.
    final = left.get("sessions", session["id"])
    assert final["stale_dropped"] == 0
    assert final["started_at"], "the create's own fields were overwritten"
    assert _sql(postgres, "select count(*) from sessions") == [(1,)]


def test_the_file_store_loses_a_write_when_two_appends_overlap(fs: Store) -> None:
    """
    Why the above matters, demonstrated deterministically on the backend in production today.

    This asserts current behaviour rather than desired behaviour — it is the motivation for the
    turns table, written down as an executable claim so that "read-modify-write races" is not
    something a reader has to take on trust. `append_turn` reads the whole array, appends one
    turn and writes the array back; two of those overlapping means the second read missed the
    first write, and the second write silently drops a turn and returns 201.

    No database and no threads: the interleaving is written out, so this runs in CI while every
    other test in this file skips, and it cannot be flaky.

    What it does not claim: the Postgres schema does not fix *this* on its own. Turns in a child
    table remove the O(n²) rewrite, and the doc merge fixes the field-level case above, but as
    long as the router sends a whole array the append itself stays a read-modify-write. Closing
    that needs an append that inserts one row — a change to `sessions.py`, not to the schema.
    """
    session = fs.create("sessions", "sess", _session_body())
    second_turn = {**TURN, "epoch": 2, "heard": "And then we added a sequence number."}

    # Both writers read the same empty array, as two concurrent `append_turn` calls would.
    seen_by_first = list(fs.get("sessions", session["id"])["turns"])
    seen_by_second = list(fs.get("sessions", session["id"])["turns"])

    fs.update("sessions", session["id"], {"turns": [*seen_by_first, TURN]})
    fs.update("sessions", session["id"], {"turns": [*seen_by_second, second_turn]})

    stored = fs.get("sessions", session["id"])["turns"]
    assert len(stored) == 1, "one of the two appended turns is gone, with no error anywhere"
    assert stored == [second_turn], "the first turn no longer loses — update this test"
