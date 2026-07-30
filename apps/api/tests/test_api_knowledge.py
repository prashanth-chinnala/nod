"""
Behaviour of the knowledge resource: CRUD, chunking, and the retrieval scorer.

The app under test is assembled here rather than imported from `avatar.server`, and the
store is bound to a `tmp_path` through `dependency_overrides`. Both are deliberate: the
router must be provably independent of the session runtime, and a suite that reached the
process-wide `store` would write into whatever `AVATAR_DATA_DIR` points at — someone's real
console data — which is the kind of test failure you only notice afterwards.

The scoring tests assert *ordering and exclusion*, never a specific score. Pinning a float
would turn any future tuning of `k1`/`b` into a red suite that says nothing about whether
retrieval got better or worse.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="console routers need the [server] extra")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from avatar import store as store_module
from avatar.api.knowledge import TOP_K_MAX, get_store, router
from avatar.store import Store

# Four paragraphs, chosen so the scorer's behaviour is observable rather than incidental:
# "the" appears in every one (so its IDF is near zero), "gpu" in exactly one, and the
# first paragraph repeats "the" six times.
CORPUS = "the the the the the the\n\ngpu the\n\nthe latency budget\n\nthe render loop\n"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_store] = lambda: Store(tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def create(client: TestClient, name: str = "Role brief", **extra: Any) -> dict[str, Any]:
    response = client.post("/knowledge", json={"name": name, **extra})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def upload(client: TestClient, kb_id: str, filename: str, text: str) -> dict[str, Any]:
    response = client.post(
        f"/knowledge/{kb_id}/documents", json={"filename": filename, "text": text}
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def search(client: TestClient, kb_id: str, query: str, **extra: Any) -> list[dict[str, Any]]:
    response = client.post(f"/knowledge/{kb_id}/query", json={"query": query, **extra})
    assert response.status_code == 200, response.text
    hits: list[dict[str, Any]] = response.json()
    return hits


# -- CRUD -------------------------------------------------------------------


def test_create_returns_a_prefixed_id_and_zeroed_derived_fields(client: TestClient) -> None:
    """
    A fresh base must already have `documents`, `chunk_count` and `total_chars`.

    Prevents the console crashing on `documents.length` for a base nobody has uploaded to,
    and prevents each consumer inventing its own "absent means empty" fallback.
    """
    record = create(client, "Role brief", description="What the role needs")

    assert record["id"].startswith("kb_")
    assert record["name"] == "Role brief"
    assert record["description"] == "What the role needs"
    assert record["documents"] == []
    assert record["chunk_count"] == 0
    assert record["total_chars"] == 0


def test_list_returns_newest_first(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The table's order is the response's order, so the response must be newest-first.

    Timestamps are pinned because `Store` records `created_at` to the second: three bases
    created in the same second would tie, and the assertion would pass or fail on filename
    order — a test that is green by luck is worse than no test.
    """
    stamps = iter(["2026-01-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00"])
    monkeypatch.setattr(store_module, "now_iso", lambda: next(stamps))

    older = create(client, "Older")
    newer = create(client, "Newer")

    listed = client.get("/knowledge").json()
    assert [row["id"] for row in listed] == [newer["id"], older["id"]]


def test_get_returns_the_stored_record(client: TestClient) -> None:
    """A create that does not survive a round trip to disk is not a create."""
    created = create(client, "Company FAQ")

    fetched = client.get(f"/knowledge/{created['id']}")

    assert fetched.status_code == 200
    assert fetched.json() == created


def test_update_patches_only_the_supplied_field(client: TestClient) -> None:
    """
    Prevents a rename blanking the description.

    A PATCH built from the full model rather than the supplied keys would write
    `description=""` for every request that omitted it — silent data loss on a rename.
    """
    created = create(client, "Role brief", description="keep me")

    patched = client.patch(f"/knowledge/{created['id']}", json={"name": "Role brief v2"})

    assert patched.status_code == 200
    assert patched.json()["name"] == "Role brief v2"
    assert patched.json()["description"] == "keep me"


def test_update_does_not_disturb_the_chunk_corpus(client: TestClient) -> None:
    """
    Renaming a base must not drop its documents.

    The derived fields are not in the patch model, so a merge that rebuilt the record from
    the request body would erase the corpus that answers queries while leaving the base
    looking healthy in the list.
    """
    kb_id = create(client)["id"]
    upload(client, kb_id, "brief.md", CORPUS)

    patched = client.patch(f"/knowledge/{kb_id}", json={"description": "renamed"}).json()

    assert patched["chunk_count"] == 4
    assert len(patched["documents"]) == 1


def test_delete_removes_it_from_the_collection(client: TestClient) -> None:
    """A delete that leaves the record readable is a delete that did nothing."""
    kb_id = create(client)["id"]

    assert client.delete(f"/knowledge/{kb_id}").status_code == 204
    assert client.get(f"/knowledge/{kb_id}").status_code == 404
    assert client.get("/knowledge").json() == []


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/knowledge/kb_missing", None),
        ("patch", "/knowledge/kb_missing", {"name": "x"}),
        ("delete", "/knowledge/kb_missing", None),
        ("post", "/knowledge/kb_missing/documents", {"filename": "a.md", "text": "hello"}),
        ("post", "/knowledge/kb_missing/query", {"query": "hello"}),
    ],
)
def test_unknown_id_is_404_on_every_route(
    client: TestClient, method: str, path: str, body: dict[str, Any] | None
) -> None:
    """
    Every route that takes an id maps `NotFound` to 404.

    Prevents a stale browser tab — a base someone deleted in another window — producing a
    500 and an unhandled `KeyError` traceback in the log instead of a routine 404.
    """
    response = client.request(method, path, json=body)

    assert response.status_code == 404


# -- validation -------------------------------------------------------------


def test_create_requires_a_name(client: TestClient) -> None:
    """An unnamed base is unidentifiable in the picker the retrieval tester uses."""
    assert client.post("/knowledge", json={"description": "no name"}).status_code == 422


@pytest.mark.parametrize("name", ["", "   ", "\n\t"])
def test_create_rejects_a_blank_name(client: TestClient, name: str) -> None:
    """
    Whitespace-only names are rejected, which `min_length` alone would accept.

    `" "` is what a half-filled form submits, and a base named with spaces renders as an
    empty row: present in the list, impossible to point at.
    """
    assert client.post("/knowledge", json={"name": name}).status_code == 422


def test_create_rejects_an_overlong_name(client: TestClient) -> None:
    """Prevents an unbounded string breaking the table layout it has to render in."""
    assert client.post("/knowledge", json={"name": "x" * 121}).status_code == 422


@pytest.mark.parametrize("field", ["chunk_count", "total_chars", "documents", "chunks"])
def test_create_refuses_client_supplied_derived_fields(client: TestClient, field: str) -> None:
    """
    The counters are server-owned; supplying them is a loud 422, not a silent drop.

    Accepting `chunk_count: 99` would put the console's totals permanently at odds with the
    corpus that actually answers queries, with nothing in the record to reveal which is
    wrong.
    """
    response = client.post("/knowledge", json={"name": "Brief", field: 99})

    assert response.status_code == 422


def test_update_rejects_a_blank_name(client: TestClient) -> None:
    """
    The name rule holds on PATCH too.

    Validating only on create is the standard hole: the same unusable record arrives one
    request later, through the edit form instead of the create form.
    """
    kb_id = create(client)["id"]

    assert client.patch(f"/knowledge/{kb_id}", json={"name": "  "}).status_code == 422


def test_update_refuses_to_patch_derived_fields(client: TestClient) -> None:
    """Closes the back door left by validating only the create body."""
    kb_id = create(client)["id"]

    assert client.patch(f"/knowledge/{kb_id}", json={"chunk_count": 99}).status_code == 422


@pytest.mark.parametrize("filename", ["", "   "])
def test_document_requires_a_filename(client: TestClient, filename: str) -> None:
    """
    A chunk's only provenance is its document's filename.

    Without one, a retrieval hit cannot be traced back to a source, which is exactly what
    the console's tester exists to show.
    """
    kb_id = create(client)["id"]

    response = client.post(
        f"/knowledge/{kb_id}/documents", json={"filename": filename, "text": "hello"}
    )

    assert response.status_code == 422


@pytest.mark.parametrize("text", ["", "   ", "\n\n\n", " \n \n "])
def test_document_that_would_chunk_to_nothing_is_rejected(
    client: TestClient, text: str
) -> None:
    """
    Prevents a stored document that is permanently unretrievable.

    Blank text yields no paragraph, so the document would appear in the list, contribute
    nothing to the corpus, and make retrieval look broken when it was the upload that was
    empty.
    """
    kb_id = create(client)["id"]

    response = client.post(
        f"/knowledge/{kb_id}/documents", json={"filename": "empty.md", "text": text}
    )

    assert response.status_code == 422


@pytest.mark.parametrize("query", ["", "   "])
def test_query_must_not_be_blank(client: TestClient, query: str) -> None:
    """
    A blank query tokenises to nothing and can only ever return nothing.

    Rejecting it keeps "no results" meaning "nothing matched" rather than "you asked
    nothing", which is the distinction the tester is there to make.
    """
    kb_id = create(client)["id"]

    assert client.post(f"/knowledge/{kb_id}/query", json={"query": query}).status_code == 422


@pytest.mark.parametrize("top_k", [0, -1])
def test_query_rejects_a_non_positive_top_k(client: TestClient, top_k: int) -> None:
    """
    `top_k=0` is rejected rather than clamped.

    A query that can never return a chunk looks configured rather than broken, and would be
    debugged as a scorer bug.
    """
    kb_id = create(client)["id"]

    response = client.post(f"/knowledge/{kb_id}/query", json={"query": "gpu", "top_k": top_k})

    assert response.status_code == 422


def test_query_rejects_a_top_k_above_the_ceiling(client: TestClient) -> None:
    """
    Every returned chunk becomes prompt text, so `top_k` is bounded.

    Prevents one request pulling the entire corpus into a turn that already has no latency
    budget to spare.
    """
    kb_id = create(client)["id"]

    response = client.post(
        f"/knowledge/{kb_id}/query", json={"query": "gpu", "top_k": TOP_K_MAX + 1}
    )

    assert response.status_code == 422


# -- chunking ---------------------------------------------------------------


def test_documents_are_chunked_on_blank_lines(client: TestClient) -> None:
    """
    Paragraphs, not lines: a single newline must not start a new chunk.

    Splitting on every newline would shatter a wrapped paragraph into fragments that each
    score badly and read as nonsense when pasted into a prompt.
    """
    kb_id = create(client)["id"]

    record = upload(
        client, kb_id, "brief.md", "first line\nstill the first chunk\n\nsecond chunk"
    )

    assert record["chunk_count"] == 2
    assert [chunk["text"] for chunk in record["chunks"]] == [
        "first line\nstill the first chunk",
        "second chunk",
    ]


def test_windows_line_endings_still_chunk(client: TestClient) -> None:
    """
    Prevents a pasted Windows file arriving as one giant unretrievable chunk.

    `\\r\\n\\r\\n` does not match a `\\n\\n` split, so without normalisation the whole
    document scores as a single blob and every query returns all of it.
    """
    kb_id = create(client)["id"]

    record = upload(client, kb_id, "windows.md", "alpha\r\n\r\nbeta")

    assert [chunk["text"] for chunk in record["chunks"]] == ["alpha", "beta"]


def test_counters_are_derived_from_the_whole_corpus(client: TestClient) -> None:
    """
    A second upload accumulates rather than replacing, and the totals match the chunks.

    Recomputed from the full list precisely so `chunk_count` cannot drift from the corpus:
    an incremental `+=` is wrong the first time a write is retried, and the console then
    reports a number nothing can be checked against.
    """
    kb_id = create(client)["id"]

    upload(client, kb_id, "one.md", "alpha\n\nbeta")
    record = upload(client, kb_id, "two.md", "gamma")

    assert len(record["documents"]) == 2
    assert record["chunk_count"] == 3
    assert record["chunk_count"] == len(record["chunks"])
    assert record["total_chars"] == sum(len(c["text"]) for c in record["chunks"])


def test_each_chunk_records_the_document_it_came_from(client: TestClient) -> None:
    """
    Provenance is what makes a retrieval hit actionable.

    Without `document_id` on the chunk, the tester can show that something matched but not
    which file to go and fix.
    """
    kb_id = create(client)["id"]

    first = upload(client, kb_id, "one.md", "alpha\n\nbeta")
    second = upload(client, kb_id, "two.md", "gamma")

    documents = {doc["filename"]: doc["id"] for doc in second["documents"]}
    owners = {chunk["text"]: chunk["document_id"] for chunk in second["chunks"]}

    assert first["documents"][0]["chunk_count"] == 2
    assert owners["alpha"] == documents["one.md"]
    assert owners["beta"] == documents["one.md"]
    assert owners["gamma"] == documents["two.md"]


# -- retrieval --------------------------------------------------------------


def test_query_returns_the_chunk_containing_the_term(client: TestClient) -> None:
    """The floor of the whole feature: a term in a chunk retrieves that chunk."""
    kb_id = create(client)["id"]
    upload(client, kb_id, "brief.md", CORPUS)

    hits = search(client, kb_id, "latency")

    assert [hit["text"] for hit in hits] == ["the latency budget"]


def test_a_rare_term_outranks_a_repeated_common_one(client: TestClient) -> None:
    """
    Prevents a regression to counting matches.

    "the gpu" hits the first paragraph six times ("the" x6) and the second twice, so raw
    overlap ranks the stopword paragraph first. IDF is what puts the chunk that actually
    mentions "gpu" on top — the difference between a useful tester and a misleading one.
    """
    kb_id = create(client)["id"]
    upload(client, kb_id, "brief.md", CORPUS)

    hits = search(client, kb_id, "the gpu", top_k=4)

    assert hits[0]["text"] == "gpu the"


def test_hits_are_ordered_by_descending_score(client: TestClient) -> None:
    """
    The console renders the response order as the ranking.

    An unsorted response would show a lower-scoring chunk above a higher one, and the
    scores in the table would visibly contradict their own order.
    """
    kb_id = create(client)["id"]
    upload(client, kb_id, "brief.md", CORPUS)

    scores = [hit["score"] for hit in search(client, kb_id, "the gpu latency", top_k=TOP_K_MAX)]

    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_top_k_bounds_the_number_of_hits(client: TestClient) -> None:
    """`top_k` is a limit, not a hint: an over-long list becomes over-long prompt text."""
    kb_id = create(client)["id"]
    upload(client, kb_id, "brief.md", CORPUS)

    assert len(search(client, kb_id, "the", top_k=2)) == 2


def test_a_query_sharing_no_term_returns_nothing(client: TestClient) -> None:
    """
    A miss must be visibly empty, not padded to `top_k` with irrelevant chunks.

    This is the single most important property for debugging: it is what separates "the
    knowledge base has no answer" from "the model ignored the answer it was given".
    """
    kb_id = create(client)["id"]
    upload(client, kb_id, "brief.md", CORPUS)

    assert search(client, kb_id, "kubernetes helm chart") == []


def test_query_against_an_empty_base_returns_nothing(client: TestClient) -> None:
    """
    Prevents a base with no documents 500-ing on the scorer's length normalisation.

    An empty corpus divides by an average chunk length of zero unless it is guarded, and
    the first thing an operator does with a new base is type into the tester.
    """
    kb_id = create(client)["id"]

    assert search(client, kb_id, "gpu") == []


def test_matching_is_case_and_punctuation_insensitive(client: TestClient) -> None:
    """
    Prevents a query missing a term only because it ended a sentence.

    Tokenising on word characters means `GPU?` and `gpu.` are the same term; comparing raw
    strings would make retrieval depend on typography.
    """
    kb_id = create(client)["id"]
    upload(client, kb_id, "brief.md", "The GPU budget is fixed.")

    assert len(search(client, kb_id, "gpu?")) == 1


def test_hits_carry_the_document_id_for_provenance(client: TestClient) -> None:
    """A hit the operator cannot trace to a file tells them nothing they can act on."""
    kb_id = create(client)["id"]
    record = upload(client, kb_id, "brief.md", CORPUS)

    hit = search(client, kb_id, "latency")[0]

    assert hit["document_id"] == record["documents"][0]["id"]
