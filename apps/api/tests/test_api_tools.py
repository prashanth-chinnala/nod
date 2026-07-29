"""
Behaviour of the /tools router.

Two things every test here is built around:

  1. **No real data directory.** The store comes in through `get_store`, overridden per
     test with a `Store(tmp_path)`. A suite that wrote through the process-wide default
     would create and delete records in whatever `AVATAR_DATA_DIR` points at, which on a
     developer machine is their actual console content.

  2. **The validation rules are the product.** A tool that cannot be called does not raise
     anywhere — it is a function the model was offered and that silently never fires, or a
     round trip with no deadline inside a turn that already measures 2.7-5.8s. Those are
     invisible at runtime, so each rule gets a test that names what it prevents.

Time is controlled where ordering is asserted: `Store` stamps `created_at` at second
precision, so two records created in the same second sort by an arbitrary tiebreak. A test
that depended on wall-clock luck would pass on a slow machine and fail on a fast one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "fastapi",
    reason="the [server] extra; CI installs only [dev] so the state machine stays web-free",
)
pytest.importorskip("httpx", reason="TestClient's transport")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from avatar.api.tools import (
    DEFAULT_TIMEOUT_MS,
    TIMEOUT_MAX_MS,
    get_store,
    router,
)
from avatar.store import Store

BUILTIN = {"name": "score_answer", "kind": "builtin"}
HTTP = {
    "name": "lookup_candidate_history",
    "kind": "http",
    "url": "http://127.0.0.1:9001/lookup",
}


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def client(data_dir: Path) -> Iterator[TestClient]:
    """A bare app carrying only this router, so nothing else in the API is under test."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_store] = lambda: Store(data_dir)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[str]]:
    """
    Replaces the store's timestamp source with a per-call counter.

    Patched on `avatar.store`, where `create`/`update` look the name up, rather than on the
    router — the router never stamps a time itself.
    """
    stamps: list[str] = []

    def fake_now() -> str:
        stamps.append(f"2026-01-01T00:00:{len(stamps):02d}+00:00")
        return stamps[-1]

    monkeypatch.setattr("avatar.store.now_iso", fake_now)
    return lambda: stamps


def create(client: TestClient, **overrides: Any) -> dict[str, Any]:
    body = {**BUILTIN, **overrides}
    response = client.post("/tools", json=body)
    assert response.status_code == 201, response.text
    return dict(response.json())


# -- the happy path ---------------------------------------------------------


def test_create_returns_the_stored_record_with_defaults_applied(client: TestClient) -> None:
    """
    The response is what was persisted, defaults included.

    Returning only the submitted fields would leave the console guessing at the timeout it
    did not set — and the timeout is the field that decides whether a turn stalls.
    """
    body = client.post("/tools", json=BUILTIN)

    assert body.status_code == 201
    record = body.json()
    assert record["id"].startswith("tool_")
    assert record["name"] == "score_answer"
    assert record["kind"] == "builtin"
    assert record["timeout_ms"] == DEFAULT_TIMEOUT_MS
    assert record["enabled"] is True
    assert record["description"] == ""
    assert record["parameters_schema"] == {}
    assert record["created_at"] == record["updated_at"]


def test_create_writes_into_the_injected_store_and_nowhere_else(
    client: TestClient, data_dir: Path
) -> None:
    """
    Guards the dependency override itself.

    If `get_store` were bypassed, every test in this file would still pass while writing
    into the real data directory. Asserting the file landed under tmp_path is the only way
    that failure is visible.
    """
    record = create(client)

    written = list((data_dir / "tools").glob("*.json"))
    assert [p.stem for p in written] == [record["id"]]


def test_parameters_schema_survives_a_round_trip_unchanged(client: TestClient) -> None:
    """
    The JSON Schema is handed to the model verbatim.

    Nested objects and required-lists must come back byte-identical; a store that flattened
    or reordered them would change the contract the model is given without any error.
    """
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
        "required": ["score"],
    }
    record = create(client, parameters_schema=schema)

    assert client.get(f"/tools/{record['id']}").json()["parameters_schema"] == schema


def test_list_is_newest_first(
    client: TestClient, frozen_clock: Callable[[], list[str]]
) -> None:
    """
    The console renders this order verbatim, so the API owns it.

    Oldest-first would put the tool an operator just added at the bottom of a growing
    table, which reads as "the create silently failed".
    """
    first = create(client, name="first_tool")
    second = create(client, name="second_tool")
    third = create(client, name="third_tool")

    listed = client.get("/tools").json()

    assert [r["id"] for r in listed] == [third["id"], second["id"], first["id"]]


def test_list_is_empty_before_anything_exists(client: TestClient) -> None:
    """An empty collection is `[]`, not a 404 — the page's empty state needs a 200."""
    response = client.get("/tools")

    assert response.status_code == 200
    assert response.json() == []


def test_get_returns_the_record(client: TestClient) -> None:
    record = create(client, **HTTP)

    fetched = client.get(f"/tools/{record['id']}")

    assert fetched.status_code == 200
    assert fetched.json() == record


def test_get_unknown_id_is_404(client: TestClient) -> None:
    """
    `NotFound` must not escape as a 500.

    A stale browser tab pointing at a deleted tool is routine; a 500 sends whoever sees it
    looking for a server fault that does not exist.
    """
    response = client.get("/tools/tool_deadbeef")

    assert response.status_code == 404
    assert "tool_deadbeef" in response.json()["detail"]


def test_update_merges_and_bumps_updated_at(
    client: TestClient, frozen_clock: Callable[[], list[str]]
) -> None:
    """
    A patch touches only what it names, and `id`/`created_at` are immutable.

    An id that moved under a patch would make one tool silently become another; a
    `created_at` that moved would reshuffle the list view.
    """
    record = create(client, description="original", timeout_ms=900)

    patched = client.patch(f"/tools/{record['id']}", json={"description": "revised"})

    assert patched.status_code == 200
    body = patched.json()
    assert body["description"] == "revised"
    assert body["timeout_ms"] == 900, "an unmentioned field must not be reset to its default"
    assert body["id"] == record["id"]
    assert body["created_at"] == record["created_at"]
    assert body["updated_at"] > record["updated_at"]


def test_update_can_disable_a_tool(client: TestClient) -> None:
    """
    `enabled: false` must persist.

    Falsy-but-present values are the classic casualty of a patch that filters on
    truthiness: the one operation that takes a misbehaving tool out of the interview
    silently does nothing.
    """
    record = create(client, enabled=True)

    patched = client.patch(f"/tools/{record['id']}", json={"enabled": False})

    assert patched.json()["enabled"] is False
    assert client.get(f"/tools/{record['id']}").json()["enabled"] is False


def test_update_unknown_id_is_404(client: TestClient) -> None:
    response = client.patch("/tools/tool_missing", json={"enabled": False})

    assert response.status_code == 404


def test_delete_removes_the_record(client: TestClient, data_dir: Path) -> None:
    record = create(client)

    assert client.delete(f"/tools/{record['id']}").status_code == 204
    assert client.get(f"/tools/{record['id']}").status_code == 404
    assert list((data_dir / "tools").glob("*.json")) == []


def test_delete_unknown_id_is_404(client: TestClient) -> None:
    """
    Not a silent 204.

    A delete that reports success for an id that never existed hides the case that
    matters: the operator is looking at the wrong console, or the record went already.
    """
    assert client.delete("/tools/tool_missing").status_code == 404


# -- name: it becomes a function name the model emits -----------------------


@pytest.mark.parametrize(
    "name",
    [
        "Score_Answer",  # uppercase
        "score answer",  # space
        "score-answer",  # hyphen
        "2score",  # leading digit
        "_score",  # leading underscore
        "score.answer",  # dot
        "",  # empty
        "score_answer ",  # trailing space, what a copy-paste leaves behind
    ],
)
def test_illegal_names_are_rejected(client: TestClient, name: str) -> None:
    """
    Prevents a tool the model can never call.

    `name` is emitted verbatim into the function schema sent to the provider. Anything
    outside `^[a-z][a-z0-9_]*$` is either rejected upstream or paraphrased by the model
    instead of emitted exactly — and the only symptom is an interviewer that appears to
    ignore a tool it was given. There is no runtime error to find.
    """
    response = client.post("/tools", json={**BUILTIN, "name": name})

    assert response.status_code == 422


@pytest.mark.parametrize("name", ["end_interview", "flag_for_review", "t", "score2", "a_1_b"])
def test_legal_names_are_accepted(client: TestClient, name: str) -> None:
    """The pattern must not be so tight it rejects the names the roadmap already names."""
    assert client.post("/tools", json={**BUILTIN, "name": name}).status_code == 201


def test_name_longer_than_the_provider_limit_is_rejected(client: TestClient) -> None:
    """
    64 characters is where both major providers reject a function name.

    Accepting a longer one moves the failure from this form, where it can be fixed, to the
    first LLM call of a live interview, where it cannot.
    """
    response = client.post("/tools", json={**BUILTIN, "name": "a" * 65})

    assert response.status_code == 422
    assert client.post("/tools", json={**BUILTIN, "name": "a" * 64}).status_code == 201


# -- kind / url: a tool that cannot be reached -----------------------------


def test_http_kind_without_a_url_is_rejected(client: TestClient) -> None:
    """
    Prevents a silent no-op mid-conversation.

    An `http` tool with no endpoint is registered with the model, offered in the interview,
    and has nothing to call. The turn stalls or the call is dropped, with no configuration
    error anywhere to explain it.
    """
    response = client.post("/tools", json={"name": "lookup", "kind": "http"})

    assert response.status_code == 422


@pytest.mark.parametrize("url", ["", "   "])
def test_http_kind_with_a_blank_url_is_rejected(client: TestClient, url: str) -> None:
    """
    A whitespace url is exactly as uncallable as a missing one.

    This is what a half-filled create form submits, so treating `""` as present would let
    the empty-endpoint case straight through the check above.
    """
    response = client.post("/tools", json={"name": "lookup", "kind": "http", "url": url})

    assert response.status_code == 422


def test_builtin_kind_needs_no_url(client: TestClient) -> None:
    """A builtin is dispatched in-process; requiring a url would make it unconfigurable."""
    record = create(client, name="end_interview", kind="builtin")

    assert record["url"] is None


def test_unknown_kind_is_rejected(client: TestClient) -> None:
    """
    Only `http` and `builtin` have a dispatch path.

    A third value would be stored happily and then match no branch at call time — a tool
    that exists in the console and does nothing in the interview.
    """
    response = client.post("/tools", json={"name": "lookup", "kind": "grpc"})

    assert response.status_code == 422


def test_update_cannot_turn_a_builtin_into_an_unreachable_http_tool(
    client: TestClient,
) -> None:
    """
    The reachability rule is checked against the merged record, not the patch.

    Validating the patch alone accepts `{"kind": "http"}` on a record with no url, which
    reintroduces the exact unreachable tool the create rule rejects — just one PATCH later.
    """
    record = create(client, name="end_interview", kind="builtin")

    response = client.patch(f"/tools/{record['id']}", json={"kind": "http"})

    assert response.status_code == 422
    assert client.get(f"/tools/{record['id']}").json()["kind"] == "builtin", (
        "a rejected patch must not have been half-applied"
    )


def test_update_to_http_is_allowed_when_a_url_comes_with_it(client: TestClient) -> None:
    record = create(client, name="end_interview", kind="builtin")

    response = client.patch(
        f"/tools/{record['id']}",
        json={"kind": "http", "url": "http://127.0.0.1:9001/end"},
    )

    assert response.status_code == 200
    assert response.json()["kind"] == "http"


def test_update_to_http_is_allowed_when_the_record_already_has_a_url(
    client: TestClient,
) -> None:
    """The merged view is what matters: a url already on the record satisfies the rule."""
    record = create(client, name="lookup", kind="builtin", url="http://127.0.0.1:9001/lookup")

    assert client.patch(f"/tools/{record['id']}", json={"kind": "http"}).status_code == 200


# -- timeout_ms: a round trip inside the turn budget -----------------------


@pytest.mark.parametrize("timeout_ms", [0, -1, -1500, TIMEOUT_MAX_MS + 1, 30_000])
def test_timeouts_outside_the_turn_budget_are_rejected(
    client: TestClient, timeout_ms: int
) -> None:
    """
    Prevents a hung interview, and prevents a guaranteed no-op.

    A tool call is a round trip *inside* a conversational turn that already measures
    2.7-5.8s, so an unbounded deadline leaves the candidate in front of a frozen avatar
    with no signal about whether it is thinking or dead. `0` and negatives are the other
    end of the same mistake: a deadline that can never be met looks configured and is a
    tool that never succeeds.
    """
    response = client.post("/tools", json={**BUILTIN, "timeout_ms": timeout_ms})

    assert response.status_code == 422


@pytest.mark.parametrize("timeout_ms", [1, 900, DEFAULT_TIMEOUT_MS, TIMEOUT_MAX_MS])
def test_timeouts_inside_the_bound_are_accepted(client: TestClient, timeout_ms: int) -> None:
    """The wall is at 5000ms inclusive; rejecting the boundary would be a silent off-by-one."""
    assert client.post("/tools", json={**BUILTIN, "timeout_ms": timeout_ms}).status_code == 201


@pytest.mark.parametrize("timeout_ms", [0, -5, TIMEOUT_MAX_MS + 1])
def test_update_enforces_the_same_timeout_bound(client: TestClient, timeout_ms: int) -> None:
    """
    A bound enforced only on create is not a bound.

    Every record here is edited far more often than it is created, so a PATCH is the likely
    route to an out-of-budget timeout.
    """
    record = create(client)

    response = client.patch(f"/tools/{record['id']}", json={"timeout_ms": timeout_ms})

    assert response.status_code == 422
    assert client.get(f"/tools/{record['id']}").json()["timeout_ms"] == DEFAULT_TIMEOUT_MS


def test_update_enforces_the_name_pattern(client: TestClient) -> None:
    """Same reasoning: the model-facing name can be broken by an edit as easily as a create."""
    record = create(client)

    assert client.patch(f"/tools/{record['id']}", json={"name": "Bad Name"}).status_code == 422


# -- unknown fields --------------------------------------------------------


@pytest.mark.parametrize("route", ["create", "update"])
def test_unknown_fields_are_rejected(client: TestClient, route: str) -> None:
    """
    A misspelled field must not be dropped in silence.

    `timeout` for `timeout_ms` would otherwise return 200 with the default still in place,
    and the operator would believe a deadline they can see in their own request body was
    applied.
    """
    if route == "create":
        response = client.post("/tools", json={**BUILTIN, "timeout": 800})
    else:
        record = create(client)
        response = client.patch(f"/tools/{record['id']}", json={"timeout": 800})

    assert response.status_code == 422
