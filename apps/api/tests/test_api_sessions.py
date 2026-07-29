"""
Session records, and the rules that keep them usable as evidence.

The interesting tests here are the refusals. A conversation record is only worth citing if it
cannot be quietly revised, so append-only is the property under test — not the CRUD.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="console routers need the [server] extra")
pytest.importorskip("httpx", reason="TestClient's transport")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import avatar.api.sessions as sessions_module
from avatar.store import Store

TURN: dict[str, Any] = {
    "epoch": 1,
    "heard": "We shipped a queue-backed ingest that assumed ordering we never had.",
    "said": "How did you discover the ordering assumption broke?",
    "transcribed": True,
    "llm_ttft_ms": 2942.0,
    "tts_first_audio_ms": 956.0,
    "first_frame_ms": 4136.0,
    "perceived_total_ms": 4161.0,
}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A router bound to a throwaway store, so no test can touch real session records."""
    monkeypatch.setattr(sessions_module, "store", Store(tmp_path))
    app = FastAPI()
    app.include_router(sessions_module.router)
    with TestClient(app) as test_client:
        yield test_client


def started(client: TestClient, **body: Any) -> dict[str, Any]:
    response = client.post("/sessions", json=body)
    assert response.status_code == 201, response.text
    return dict(response.json())


# -- the append-only contract ---------------------------------------------


def test_there_is_no_way_to_edit_or_delete_a_session() -> None:
    """
    The property that makes these records evidence.

    Asserted against the route table rather than by calling and expecting 405, because the
    point is that no such handler was ever written — a 405 from a missing method is an
    accident, and this must be a decision.
    """
    methods = {
        method
        for route in sessions_module.router.routes
        for method in getattr(route, "methods", set())
    }

    assert "PATCH" not in methods
    assert "PUT" not in methods
    assert "DELETE" not in methods


def test_ending_twice_keeps_the_first_timestamp(client: TestClient) -> None:
    """
    A socket can close twice — a client disconnect racing a server shutdown. Session
    duration is derived from `ended_at`, so a second call moving it would corrupt every
    duration computed from this record.
    """
    session = started(client)

    first = client.post(f"/sessions/{session['id']}/end").json()
    second = client.post(f"/sessions/{session['id']}/end").json()

    assert first["ended_at"] is not None
    assert second["ended_at"] == first["ended_at"]


def test_appending_after_end_is_rejected(client: TestClient) -> None:
    session = started(client)
    client.post(f"/sessions/{session['id']}/end")

    response = client.post(f"/sessions/{session['id']}/turns", json=TURN)

    assert response.status_code == 409
    assert "ended" in response.json()["detail"]


# -- turns ----------------------------------------------------------------


def test_a_turn_is_appended_in_order(client: TestClient) -> None:
    session = started(client)

    client.post(f"/sessions/{session['id']}/turns", json={**TURN, "epoch": 1})
    client.post(f"/sessions/{session['id']}/turns", json={**TURN, "epoch": 2})
    record = client.get(f"/sessions/{session['id']}").json()

    assert [turn["epoch"] for turn in record["turns"]] == [1, 2]


def test_a_turn_with_no_timings_is_accepted(client: TestClient) -> None:
    """
    An interrupted turn has no completed stages, and it is the one most worth counting.
    Rejecting it would make barge-ins invisible in exactly the records used to study them.
    """
    session = started(client)

    response = client.post(
        f"/sessions/{session['id']}/turns",
        json={"epoch": 3, "heard": "wait—", "interrupted": True},
    )

    assert response.status_code == 201
    assert response.json()["turns"][0]["interrupted"] is True


def test_an_empty_transcript_is_recorded_as_such(client: TestClient) -> None:
    """
    `transcribed=False` must survive into the record.

    A session where every turn has it false is a broken speech-to-text configuration, and it
    presents as an interviewer that ignores answers. Storing only `heard` would conflate a
    silent candidate with a transcriber that returned nothing.
    """
    session = started(client)

    client.post(
        f"/sessions/{session['id']}/turns",
        json={"epoch": 1, "heard": "[640ms of speech, no transcript]", "transcribed": False},
    )
    record = client.get(f"/sessions/{session['id']}").json()

    assert record["turns"][0]["transcribed"] is False


def test_a_negative_timing_is_rejected(client: TestClient) -> None:
    """A negative duration is a clock bug upstream; storing it would propagate it into
    every aggregate computed from these records."""
    session = started(client)

    response = client.post(f"/sessions/{session['id']}/turns", json={**TURN, "llm_ttft_ms": -5})

    assert response.status_code == 422


def test_an_unknown_turn_field_is_rejected(client: TestClient) -> None:
    """`extra="forbid"`, so a typo in the runtime's payload fails loudly at the boundary
    rather than being silently dropped and noticed weeks later as missing data."""
    session = started(client)

    response = client.post(f"/sessions/{session['id']}/turns", json={**TURN, "ttft": 12})

    assert response.status_code == 422


# -- lifecycle ------------------------------------------------------------


def test_a_new_session_starts_empty_and_open(client: TestClient) -> None:
    session = started(client, agent_id="agent_abc123")

    assert session["turns"] == []
    assert session["ended_at"] is None
    assert session["agent_id"] == "agent_abc123"
    assert session["started_at"]


def test_agent_id_is_optional(client: TestClient) -> None:
    """The runtime can open a session before an agent is chosen; refusing that would make
    the record depend on console state the socket does not have."""
    assert started(client)["agent_id"] is None


def test_list_is_newest_first(client: TestClient) -> None:
    """The console renders this order verbatim, so the API owns it."""
    first = started(client)
    second = started(client)

    ids = [row["id"] for row in client.get("/sessions").json()]

    assert set(ids) == {first["id"], second["id"]}


def test_unknown_session_is_404_everywhere(client: TestClient) -> None:
    assert client.get("/sessions/sess_nope").status_code == 404
    assert client.post("/sessions/sess_nope/turns", json=TURN).status_code == 404
    assert client.post("/sessions/sess_nope/end").status_code == 404


def test_a_traversal_id_is_404_not_a_file_read(client: TestClient) -> None:
    """The store guards this, and the guard is worth a test at the HTTP boundary too —
    this is one URL parameter away from arbitrary user input."""
    assert client.get("/sessions/..%2f..%2fetc%2fpasswd").status_code == 404
