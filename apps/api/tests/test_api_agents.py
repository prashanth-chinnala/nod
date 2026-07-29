"""
Behaviour of the agents router.

Two setup decisions worth stating, because both are about not lying to the suite:

  1. `pytest.importorskip` rather than a plain import. CI installs only `[dev]`, which has
     no web stack — that is the whole reason `test_boundaries.py` can assert a GPU-free,
     framework-free orchestration layer. A hard import here would turn "the console is not
     installed" into a red build on a change that never touched the console.

  2. Every test gets its own `Store(tmp_path)`, patched over the module global the router
     reads. Nothing here can see, or write to, a real `data/` directory — a suite that
     mutates the operator's agents while checking a 404 is a suite nobody dares run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="console routers need the [server] extra")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import avatar.store as store_module
from avatar.api import agents
from avatar.audio.turn_detection import (
    END_OF_TURN_SILENCE_MS,
    MIN_SPEECH_MS,
    ONSET_FRAMES,
    ONSET_PROBABILITY,
    RELEASE_PROBABILITY,
)
from avatar.store import Store

MINIMAL: dict[str, Any] = {"name": "Screening interviewer"}


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the router's store at a throwaway directory for the duration of one test."""
    monkeypatch.setattr(agents, "store", Store(tmp_path))
    return tmp_path


@pytest.fixture
def client(data_dir: Path) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(agents.router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def ticking_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Distinct, increasing `created_at` for every write.

    Real timestamps are second-precision, so two records created inside the same test share
    one and the store's sort has nothing to order them by. Ordering asserted against a real
    clock is therefore a coin flip; this makes it a fact.
    """
    counter = iter(range(1, 100))

    def stamp() -> str:
        return f"2026-07-27T12:00:{next(counter):02d}+00:00"

    monkeypatch.setattr(store_module, "now_iso", stamp)


def create(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post("/agents", json={**MINIMAL, **overrides})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


# -- the happy path --------------------------------------------------------


def test_create_returns_201_with_an_identifier_and_timestamps(client: TestClient) -> None:
    """
    A create that returns no id forces the caller to re-list to find what it just made,
    and the console's create-then-edit flow has nothing to navigate to.
    """
    agent = create(client, name="Screening interviewer")

    assert agent["id"].startswith("agent_")
    assert agent["name"] == "Screening interviewer"
    assert agent["created_at"] == agent["updated_at"]


def test_create_writes_only_inside_the_injected_store(
    client: TestClient, data_dir: Path
) -> None:
    """
    The store is really the patched one. Without this, every other test in the file could
    be passing against the developer's real `data/agents` and nobody would know.
    """
    agent = create(client)

    assert (data_dir / "agents" / f"{agent['id']}.json").is_file()


def test_defaults_match_the_running_detectors_own_constants(client: TestClient) -> None:
    """
    The console must not describe a policy the server does not run.

    Pinned against `audio.turn_detection` rather than against literals so that tuning a
    threshold there cannot leave this API quietly serving the old value as its default.
    """
    turn_taking = create(client)["turn_taking"]

    assert turn_taking == {
        "onset_probability": ONSET_PROBABILITY,
        "release_probability": RELEASE_PROBABILITY,
        "onset_frames": ONSET_FRAMES,
        "min_speech_ms": MIN_SPEECH_MS,
        "end_of_turn_silence_ms": END_OF_TURN_SILENCE_MS,
    }


def test_defaults_need_no_credentials(client: TestClient) -> None:
    """
    A fresh agent has to be runnable on a clean clone. Defaulting to a vendor provider
    would make the first session fail on a missing key rather than talk.
    """
    agent = create(client)

    assert agent["llm_provider"] == "scripted"
    assert agent["voice_provider"] == "tone"


def test_list_returns_newest_first(client: TestClient, ticking_clock: None) -> None:
    """
    Newest-first is the order the list view is built around. A router that re-sorted, or
    passed the store's order through reversed, would bury a just-created agent at the
    bottom of a long table — the one place the operator will not look for it.
    """
    first = create(client, name="Older")
    second = create(client, name="Newer")

    listed = client.get("/agents").json()

    assert [a["id"] for a in listed] == [second["id"], first["id"]]


def test_list_is_empty_before_anything_is_created(client: TestClient) -> None:
    """An empty collection is a 200 and an empty list, never a 404 or a 500."""
    response = client.get("/agents")

    assert response.status_code == 200
    assert response.json() == []


def test_get_returns_the_stored_record(client: TestClient) -> None:
    agent = create(client, system_prompt="Ask about failure modes.")

    fetched = client.get(f"/agents/{agent['id']}").json()

    assert fetched == agent


def test_update_changes_the_sent_field_and_leaves_the_rest_alone(client: TestClient) -> None:
    """
    A patch that silently resets omitted fields to their defaults would wipe a tuned
    turn-taking policy every time someone renamed an agent.
    """
    agent = create(client, system_prompt="Ask about failure modes.", llm_model="claude-x")

    patched = client.patch(f"/agents/{agent['id']}", json={"name": "Renamed"}).json()

    assert patched["name"] == "Renamed"
    assert patched["system_prompt"] == "Ask about failure modes."
    assert patched["llm_model"] == "claude-x"
    assert patched["turn_taking"] == agent["turn_taking"]
    assert patched["id"] == agent["id"]
    assert patched["created_at"] == agent["created_at"]


def test_update_replaces_the_whole_turn_taking_object(client: TestClient) -> None:
    """
    Documents the merge semantics the hysteresis rule depends on: `turn_taking` arrives
    complete, so `TurnTaking` alone decides whether a pair is legal. If this ever became a
    field-level merge, a legal-looking partial patch could compose an illegal stored pair.
    """
    agent = create(client)
    replacement = {
        "onset_probability": 0.7,
        "release_probability": 0.2,
        "onset_frames": 4,
        "min_speech_ms": 250,
        "end_of_turn_silence_ms": 500,
    }

    patched = client.patch(f"/agents/{agent['id']}", json={"turn_taking": replacement}).json()

    assert patched["turn_taking"] == replacement


def test_delete_removes_the_record(client: TestClient) -> None:
    agent = create(client)

    assert client.delete(f"/agents/{agent['id']}").status_code == 204
    assert client.get(f"/agents/{agent['id']}").status_code == 404
    assert client.get("/agents").json() == []


# -- unknown ids -----------------------------------------------------------


@pytest.mark.parametrize("method", ["get", "patch", "delete"])
def test_unknown_id_is_a_404_on_every_verb(client: TestClient, method: str) -> None:
    """
    `NotFound` is a `KeyError`. Unmapped it becomes a 500, which tells the console
    "the server is broken" for what is only a stale link.
    """
    kwargs: dict[str, Any] = {"json": {"name": "x"}} if method == "patch" else {}
    response = getattr(client, method)("/agents/agent_deadbeef", **kwargs)

    assert response.status_code == 404
    assert "agent_deadbeef" in response.json()["detail"]


def test_a_traversal_attempt_in_the_id_is_a_404_not_a_file_read(client: TestClient) -> None:
    """
    The id goes into a filename. `..%2F..%2Fetc%2Fpasswd` must not resolve to a read
    outside the collection — the store guards it, and this is the check that says so from
    the outside, where the untrusted value actually enters.
    """
    response = client.get("/agents/..%2F..%2Fsecrets")

    assert response.status_code == 404


# -- the validation that matters -------------------------------------------


@pytest.mark.parametrize(
    ("onset", "release"),
    [(0.6, 0.7), (0.4, 0.9)],
    ids=["inverted", "far-inverted"],
)
def test_release_above_onset_is_rejected(
    client: TestClient, onset: float, release: float
) -> None:
    """
    An inverted hysteresis pair makes speech harder to sustain than to enter, so the
    detector ends the turn inside almost every word. `TurnDetector` raises on it at
    construction — i.e. when a candidate connects — so storing it here would convert an
    operator's typo into a session that dies at start, minutes later and somewhere else.
    """
    response = client.post(
        "/agents",
        json={
            **MINIMAL,
            "turn_taking": {"onset_probability": onset, "release_probability": release},
        },
    )

    assert response.status_code == 422
    detail = response.text
    assert str(release) in detail and str(onset) in detail, "the message must name both values"


def test_release_equal_to_onset_is_rejected(client: TestClient) -> None:
    """
    A zero gap is no hysteresis at all: the same probability both fails to sustain speech
    and ends the turn, so a plosive mid-word ends it. Deliberately stricter than the
    detector, which only rejects strictly-inverted pairs.
    """
    response = client.post(
        "/agents",
        json={
            **MINIMAL,
            "turn_taking": {"onset_probability": 0.5, "release_probability": 0.5},
        },
    )

    assert response.status_code == 422


def test_the_hysteresis_rule_also_guards_the_update_path(client: TestClient) -> None:
    """
    Create-time validation alone leaves the back door open: a patch is how an operator
    tunes thresholds, so it is the *likelier* route to an inverted pair, not the rarer one.
    """
    agent = create(client)

    response = client.patch(
        f"/agents/{agent['id']}",
        json={
            "turn_taking": {
                "onset_probability": 0.3,
                "release_probability": 0.8,
                "onset_frames": ONSET_FRAMES,
                "min_speech_ms": MIN_SPEECH_MS,
                "end_of_turn_silence_ms": END_OF_TURN_SILENCE_MS,
            }
        },
    )

    assert response.status_code == 422
    assert client.get(f"/agents/{agent['id']}").json()["turn_taking"] == agent["turn_taking"]


@pytest.mark.parametrize(
    "field",
    ["onset_probability", "release_probability"],
)
@pytest.mark.parametrize("value", [-0.1, 1.4], ids=["below-zero", "above-one"])
def test_probabilities_outside_zero_to_one_are_rejected(
    client: TestClient, field: str, value: float
) -> None:
    """
    A threshold above 1 can never be met, so speech is never confirmed and the avatar
    never answers; below 0 it is always met, so the room's noise floor is a turn. Both
    fail silently at runtime — there is nothing to raise, the policy just stops working.
    """
    response = client.post("/agents", json={**MINIMAL, "turn_taking": {field: value}})

    assert response.status_code == 422


def test_onset_frames_below_one_is_rejected(client: TestClient) -> None:
    """
    Mirrors `TurnDetector`, which raises on `onset_frames < 1`. Zero consecutive frames
    means the requirement for sustained speech is gone and a door slam is an interruption.
    """
    response = client.post("/agents", json={**MINIMAL, "turn_taking": {"onset_frames": 0}})

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["min_speech_ms", "end_of_turn_silence_ms"])
def test_negative_millisecond_thresholds_are_rejected(client: TestClient, field: str) -> None:
    """
    A negative duration is not a lenient setting, it is a threshold that is already
    satisfied before any audio arrives: end-of-turn would fire on the first quiet frame.
    """
    response = client.post("/agents", json={**MINIMAL, "turn_taking": {field: -1}})

    assert response.status_code == 422


@pytest.mark.parametrize("name", ["", "   "])
def test_a_blank_name_is_rejected(client: TestClient, name: str) -> None:
    """
    Name is the only handle the console has on an agent. A blank or whitespace-only one
    renders as an empty row that cannot be recognised, searched for, or safely deleted.
    """
    response = client.post("/agents", json={"name": name})

    assert response.status_code == 422


def test_a_name_is_stored_trimmed(client: TestClient) -> None:
    """Surrounding whitespace is invisible in the table but not in a lookup or a sort."""
    assert create(client, name="  Screening  ")["name"] == "Screening"


@pytest.mark.parametrize(
    ("field", "value"),
    [("llm_provider", "gemini"), ("voice_provider", "elevenlabs")],
)
def test_an_unsupported_provider_is_rejected(
    client: TestClient, field: str, value: str
) -> None:
    """
    The runtime builds these by name and raises on an unknown one. Accepting a provider
    with no adapter stores a session that cannot start; the enum is the whole guard.
    """
    response = client.post("/agents", json={**MINIMAL, field: value})

    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"name": "x", "system_promt": "typo"},
        {"name": "x", "turn_taking": {"end_of_turn_silence": 700}},
    ],
    ids=["top-level", "nested"],
)
def test_an_unknown_field_is_rejected_rather_than_stored(
    client: TestClient, body: dict[str, Any]
) -> None:
    """
    Silently accepting a misspelled key is the cruellest failure available here: the write
    succeeds, the operator believes the threshold moved, and the session behaves as if
    they never touched it. A 422 costs one retry and no confusion.
    """
    response = client.post("/agents", json=body)

    assert response.status_code == 422


def test_name_is_required(client: TestClient) -> None:
    """An unnamed agent is unusable in every list view the console has."""
    assert client.post("/agents", json={}).status_code == 422
