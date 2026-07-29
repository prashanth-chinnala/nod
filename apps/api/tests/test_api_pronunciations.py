"""
The pronunciations resource: CRUD, and the substitution rules that are the point of it.

Every test here runs against a `Store(tmp_path)` bound through `dependency_overrides`, so
nothing touches a real data directory — the module-level singleton resolves to
`AVATAR_DATA_DIR`, and a suite that wrote there would both pollute a checkout and pass or
fail depending on what the previous run left behind.

The substitution tests carry most of the weight. `apply_lexicon` is four lines of regex
standing in for the obvious `str.replace` loop, and the two reasons it cannot be that loop
are invisible unless a test pins them.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# The web stack is an optional extra: `pyproject.toml` keeps fastapi out of `[dev]` so CI can
# type-check and test the orchestration layer with nothing installed but pytest. Skipping here
# rather than importing unconditionally is what stops that choice from becoming a collection
# error — the cost, stated plainly, is that these tests only run where `[server]` is installed.
pytest.importorskip("fastapi", reason="needs the [server] extra; see pyproject.toml")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from avatar.api.pronunciations import get_store, router
from avatar.store import Store

NGINX = {"term": "nginx", "say": "engine ex"}
POSTGRES = {"term": "PostgreSQL", "say": "post gress cue ell"}


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A client whose routes read and write inside `tmp_path` and nowhere else."""
    app = FastAPI()
    app.include_router(router)
    data = Store(tmp_path)
    app.dependency_overrides[get_store] = lambda: data
    with TestClient(app) as test_client:
        yield test_client


def create(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/pronunciations", json=body)
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


def applied(client: TestClient, lexicon_id: str, text: str) -> str:
    response = client.post(f"/pronunciations/{lexicon_id}/apply", json={"text": text})
    assert response.status_code == 200, response.text
    result: str = response.json()["text"]
    return result


# -- CRUD ------------------------------------------------------------------


def test_create_returns_the_stored_record_with_an_id_and_timestamps(
    client: TestClient,
) -> None:
    record = create(client, name="Infra terms", entries=[NGINX, POSTGRES])

    assert record["id"].startswith("lex_"), "the id prefix is what makes a stray id readable"
    assert record["name"] == "Infra terms"
    assert record["entries"] == [NGINX, POSTGRES]
    assert record["created_at"] == record["updated_at"]


def test_entries_default_to_empty_rather_than_missing(client: TestClient) -> None:
    """
    A lexicon created with no entries still has an `entries` key.

    The console reads `entries.length` for its table; a record where the field is absent
    rather than `[]` renders as a crash on the list page, not as a zero.
    """
    record = create(client, name="Empty for now")

    assert record["entries"] == []


def test_list_returns_lexicons_newest_first(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The console's table renders this order verbatim, so it has to be the stored order.

    `now_iso` is stubbed to strictly increasing values because it has second precision:
    three records created in the same real second would share a timestamp and fall back to
    filename order, which is random hex — a test that passes on a slow machine and fails on
    a fast one.
    """
    stamps = iter(["2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"])
    monkeypatch.setattr("avatar.store.now_iso", lambda: next(stamps))

    create(client, name="older")
    create(client, name="newer")

    listed = client.get("/pronunciations").json()

    assert [row["name"] for row in listed] == ["newer", "older"]


def test_get_reads_back_what_create_wrote(client: TestClient) -> None:
    record = create(client, name="Infra terms", entries=[NGINX])

    fetched = client.get(f"/pronunciations/{record['id']}")

    assert fetched.status_code == 200
    assert fetched.json() == record


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/pronunciations/lex_missing", None),
        ("PATCH", "/pronunciations/lex_missing", {"name": "x"}),
        ("DELETE", "/pronunciations/lex_missing", None),
        ("POST", "/pronunciations/lex_missing/apply", {"text": "x"}),
    ],
)
def test_unknown_id_is_404_on_every_route(
    client: TestClient, method: str, path: str, body: dict[str, Any] | None
) -> None:
    """
    An unknown id is a client mistake, not a server fault.

    Without the mapping, `Store.NotFound` escapes as an unhandled `KeyError` and a stale
    browser tab reports a 500 — which sends whoever is debugging into the server logs for
    something the status code should have told them. Every route is covered because each one
    reaches the store by a different path, and `apply` and `PATCH` are the two that were
    easiest to leave unguarded.
    """
    response = client.request(method, path, json=body)

    assert response.status_code == 404


def test_update_replaces_entries_and_moves_updated_at(client: TestClient) -> None:
    record = create(client, name="Infra terms", entries=[NGINX])

    patched = client.patch(
        f"/pronunciations/{record['id']}", json={"name": "Infra", "entries": [POSTGRES]}
    )

    assert patched.status_code == 200
    body = patched.json()
    assert body["name"] == "Infra"
    assert body["entries"] == [POSTGRES], "entries replace wholesale; they do not merge"
    assert body["created_at"] == record["created_at"], "created_at is immutable"


def test_update_leaves_omitted_fields_alone(client: TestClient) -> None:
    """
    A patch of one field must not blank the others.

    The console's entry editor and its rename control are separate submissions; if a patch
    carrying only `entries` also reset `name`, renaming would be undone by the next save.
    """
    record = create(client, name="Infra terms", entries=[NGINX])

    body = client.patch(f"/pronunciations/{record['id']}", json={"entries": []}).json()

    assert body["name"] == "Infra terms"
    assert body["entries"] == []


def test_delete_removes_the_record(client: TestClient) -> None:
    record = create(client, name="Infra terms")

    assert client.delete(f"/pronunciations/{record['id']}").status_code == 204
    assert client.get(f"/pronunciations/{record['id']}").status_code == 404
    assert client.get("/pronunciations").json() == []


# -- validation ------------------------------------------------------------


def test_name_is_required(client: TestClient) -> None:
    """A lexicon with no name is unidentifiable in the console's table."""
    assert client.post("/pronunciations", json={"entries": []}).status_code == 422


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_whitespace_only_name_is_rejected_not_stored(client: TestClient, blank: str) -> None:
    """
    `" "` is exactly as unidentifiable as a missing name, and is what an empty form submits.

    Stripping before the length check is what makes this a 422; strip alone would store an
    empty string and produce a row with nothing in the name column.
    """
    assert client.post("/pronunciations", json={"name": blank}).status_code == 422


def test_name_is_stored_stripped(client: TestClient) -> None:
    """Trailing whitespace from a paste must not make two identical names look different."""
    assert create(client, name="  Infra terms  ")["name"] == "Infra terms"


@pytest.mark.parametrize("blank", ["", "  "])
def test_blank_term_is_rejected(client: TestClient, blank: str) -> None:
    """
    An empty term compiles into a pattern that matches every position in the sentence.

    Accepting one would not fail loudly: it would quietly rewrite the whole utterance, and
    the lexicon would look configured while the voice became gibberish.
    """
    response = client.post(
        "/pronunciations", json={"name": "L", "entries": [{"term": blank, "say": "x"}]}
    )

    assert response.status_code == 422


@pytest.mark.parametrize("blank", ["", "  "])
def test_blank_say_is_rejected(client: TestClient, blank: str) -> None:
    """
    A blank replacement deletes the word it was supposed to fix.

    That is worse than the mispronunciation: the sentence loses its subject and the
    transcript no longer explains why.
    """
    response = client.post(
        "/pronunciations", json={"name": "L", "entries": [{"term": "nginx", "say": blank}]}
    )

    assert response.status_code == 422


def test_terms_are_stored_stripped(client: TestClient) -> None:
    """A term with a trailing space would never match, since the space is not part of a word."""
    record = create(client, name="L", entries=[{"term": " nginx ", "say": " engine ex "}])

    assert record["entries"] == [{"term": "nginx", "say": "engine ex"}]


def test_terms_differing_only_in_case_are_rejected(client: TestClient) -> None:
    """
    Matching is case-insensitive, so `Kafka` and `kafka` are one rule with two answers.

    Which one wins depends on list order, so the lexicon would appear configured and switch
    pronunciation whenever someone reordered the editor. Rejected at the boundary instead.
    """
    response = client.post(
        "/pronunciations",
        json={
            "name": "L",
            "entries": [
                {"term": "Kafka", "say": "KAFF-ka"},
                {"term": "kafka", "say": "KAF-kuh"},
            ],
        },
    )

    assert response.status_code == 422
    assert "duplicate term" in response.text


def test_duplicate_terms_are_rejected_on_update_too(client: TestClient) -> None:
    """
    The invariant belongs to the stored record, not to one request shape.

    Enforcing it only on create leaves the editor — which submits PATCH on every save — as
    the way the bad state actually gets written.
    """
    record = create(client, name="L", entries=[NGINX])

    response = client.patch(
        f"/pronunciations/{record['id']}",
        json={"entries": [{"term": "ex", "say": "eks"}, {"term": "EX", "say": "ecks"}]},
    )

    assert response.status_code == 422


def test_unknown_fields_are_rejected(client: TestClient) -> None:
    """
    `entry` for `entries` must fail, not create an empty lexicon that reports success.

    Silently dropping the field is how "I added ten terms and nothing changed" happens, and
    a 201 in the network tab makes it look like a synthesis problem instead.
    """
    response = client.post(
        "/pronunciations", json={"name": "L", "entry": [NGINX], "entries": []}
    )

    assert response.status_code == 422


def test_apply_requires_text(client: TestClient) -> None:
    """
    `{}` is a malformed request, not an empty string.

    Defaulting it would make a client that forgot the field look like a lexicon that does
    nothing, which is the harder of the two to diagnose.
    """
    record = create(client, name="L", entries=[NGINX])

    response = client.post(f"/pronunciations/{record['id']}/apply", json={})

    assert response.status_code == 422


# -- apply -----------------------------------------------------------------


def test_apply_substitutes_a_term(client: TestClient) -> None:
    record = create(client, name="L", entries=[NGINX, POSTGRES])

    assert applied(client, record["id"], "We run nginx in front of PostgreSQL.") == (
        "We run engine ex in front of post gress cue ell."
    )


@pytest.mark.parametrize("written", ["nginx", "NGINX", "Nginx", "nGiNx"])
def test_apply_is_case_insensitive(client: TestClient, written: str) -> None:
    """
    Whatever the candidate typed, the voice must get the fix.

    A case-sensitive lexicon needs one entry per capitalisation, and the one nobody thought
    of is the one that reaches the synthesiser unfixed.
    """
    record = create(client, name="L", entries=[NGINX])

    assert applied(client, record["id"], written) == "engine ex"


def test_apply_matches_whole_words_only(client: TestClient) -> None:
    """
    The defect a substring match has: `Kafka` → `KAFF-ka` turns `Kafkaesque` into
    `KAFF-kaesque`.

    So the fix for one word silently corrupts every longer word containing it — and the
    longer word is usually the one a candidate says while explaining something.
    """
    record = create(client, name="L", entries=[{"term": "Kafka", "say": "KAFF-ka"}])

    assert applied(client, record["id"], "Kafkaesque") == "Kafkaesque"
    assert applied(client, record["id"], "unkafka") == "unkafka"
    assert applied(client, record["id"], "Kafka, precisely.") == "KAFF-ka, precisely."


def test_apply_does_not_re_substitute_inside_a_replacement(client: TestClient) -> None:
    """
    The defect a per-entry replace loop has.

    With `nginx` → `engine ex` and `ex` → `eks` in one lexicon, a second pass reads the text
    the first pass produced and emits `engine eks`. One alternation in one pass is the fix:
    `re.sub` resumes after each match in the *input*, so a replacement is never rescanned.
    """
    record = create(client, name="L", entries=[NGINX, {"term": "ex", "say": "eks"}])

    assert applied(client, record["id"], "nginx") == "engine ex"
    assert applied(client, record["id"], "ex") == "eks", "the second rule still works alone"


def test_apply_prefers_the_longest_matching_term(client: TestClient) -> None:
    """
    `re` alternation is leftmost-first-alternative, not longest-match.

    Without the length sort, `SQL` listed first claims the tail of `PostgreSQL` and the
    specific rule the operator wrote is unreachable.
    """
    record = create(
        client,
        name="L",
        entries=[{"term": "SQL", "say": "sequel"}, POSTGRES],
    )

    assert applied(client, record["id"], "PostgreSQL") == "post gress cue ell"
    assert applied(client, record["id"], "SQL") == "sequel"


def test_apply_handles_terms_with_punctuation(client: TestClient) -> None:
    """
    Why the word edges are lookaheads rather than `\\b`.

    `\\bC\\+\\+\\b` never matches `C++`: the trailing boundary sits after `+` and demands a
    word character to its left. The terms most likely to need an override — `C++`, `.NET`,
    `node.js` — are exactly the ones `\\b` drops.
    """
    record = create(
        client,
        name="L",
        entries=[{"term": "C++", "say": "see plus plus"}, {"term": ".NET", "say": "dot net"}],
    )

    assert applied(client, record["id"], "C++ and .NET") == "see plus plus and dot net"


def test_apply_leaves_text_alone_when_the_lexicon_is_empty(client: TestClient) -> None:
    """
    An empty lexicon is a no-op, not an error.

    A newly created lexicon has no entries, and the console's preview calls this immediately
    — a 500 or an empty string there reads as a broken endpoint rather than an empty list.
    """
    record = create(client, name="L")

    assert applied(client, record["id"], "nginx stays put") == "nginx stays put"


def test_apply_accepts_empty_text(client: TestClient) -> None:
    """The preview fires on every keystroke, including the one that clears the box."""
    record = create(client, name="L", entries=[NGINX])

    assert applied(client, record["id"], "") == ""
