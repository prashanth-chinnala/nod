"""
Tests for the rubric API and the agent field that points at it.

The validators here exist to stop data that would look fine on disk and misbehave weeks
later, so the tests are written against those specific outcomes rather than against status
codes.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.importorskip("fastapi", reason="console routers need the [server] extra")

from fastapi.testclient import TestClient

from avatar.agent_config import AgentNotConfigured, resolve_agent
from avatar.server import app
from avatar.store import Store


@pytest.fixture
def client(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A client over a throwaway store, so a test cannot see or damage real records."""
    import avatar.api.rubrics as rubrics_api
    import avatar.server as server_module

    fresh = Store(tmp_path)  # type: ignore[arg-type]
    monkeypatch.setattr(rubrics_api, "store", fresh)
    monkeypatch.setattr(server_module, "store", fresh)
    for module in ("avatar.api.agents", "avatar.agent_config"):
        monkeypatch.setattr(__import__(module, fromlist=["store"]), "store", fresh)
    with TestClient(app) as instance:
        yield instance


BACKEND = {
    "name": "Backend engineer",
    "description": "Senior backend, data-heavy",
    "competencies": [
        {
            "name": "Debugging under pressure",
            "probe": "How they find a fault in a system they did not build.",
            "signals": ["profiler", "flame graph", "bisect"],
        },
        {"name": "Data at scale", "signals": ["sharding", "partition"], "max_turns": 2},
    ],
}


def test_create_stamps_derived_ids(client: TestClient) -> None:
    """
    Ids are in the stored data, not computed on read.

    Computing on read works until a rename, at which point past session coverage keys off an id
    no longer derivable from the current name — and a report silently loses an area.
    """
    created = client.post("/rubrics", json=BACKEND).json()
    assert [c["id"] for c in created["competencies"]] == [
        "debugging-under-pressure",
        "data-at-scale",
    ]


def test_colliding_names_are_refused(client: TestClient) -> None:
    """
    Two competencies that slug to one id would share a coverage record.

    Refused here because downstream the failure is invisible: the second appears in the console,
    never accumulates its own evidence, and inherits the first's status.
    """
    response = client.post(
        "/rubrics",
        json={
            "name": "Colliding",
            "competencies": [{"name": "Data at scale"}, {"name": "data at scale!"}],
        },
    )
    assert response.status_code == 422
    assert "share one coverage record" in response.text


def test_unreachable_min_signals_is_refused(client: TestClient) -> None:
    """
    `min_signals` above the number of signals is a competency that can never be evidenced.

    Not a strict rubric — a broken one, which would be probed to exhaustion every single time
    and reported as a dead end.
    """
    response = client.post(
        "/rubrics",
        json={
            "name": "Impossible",
            "competencies": [{"name": "Scale", "signals": ["sharding"], "min_signals": 3}],
        },
    )
    assert response.status_code == 422
    assert "could never be evidenced" in response.text


def test_signal_less_competency_saves_but_warns(client: TestClient) -> None:
    """
    A draft is allowed; a draft that cannot work says so.

    Refusing would push an operator to invent signals to satisfy a validator. Saying nothing
    would let it be discovered from a confusing report weeks later.
    """
    created = client.post(
        "/rubrics", json={"name": "Draft", "competencies": [{"name": "Communication"}]}
    ).json()
    assert created["id"]
    assert any("no signals for Communication" in note for note in created["warnings"])


def test_empty_rubric_warns_that_it_will_not_steer(client: TestClient) -> None:
    created = client.post("/rubrics", json={"name": "Empty"}).json()
    assert any("will not steer" in note for note in created["warnings"])


def test_patch_replaces_competencies_whole(client: TestClient) -> None:
    """
    Whole rather than merged, so the uniqueness rule stays in one place.

    Merging one competency would let a rename collide with a sibling that was not sent, and the
    check would have to move somewhere it could be forgotten.
    """
    created = client.post("/rubrics", json=BACKEND).json()
    updated = client.patch(
        f"/rubrics/{created['id']}",
        json={"competencies": [{"name": "Only this", "signals": ["x"]}]},
    ).json()
    assert [c["id"] for c in updated["competencies"]] == ["only-this"]
    assert updated["name"] == "Backend engineer"  # untouched keys survive


def test_patch_rejects_a_collision_too(client: TestClient) -> None:
    """The update path needs its own check; a validator on create only is half a rule."""
    created = client.post("/rubrics", json=BACKEND).json()
    response = client.patch(
        f"/rubrics/{created['id']}",
        json={"competencies": [{"name": "Scale"}, {"name": "scale"}]},
    )
    assert response.status_code == 422


def test_missing_rubric_is_404_not_500(client: TestClient) -> None:
    assert client.get("/rubrics/rubric_nope").status_code == 404
    assert client.patch("/rubrics/rubric_nope", json={"name": "x"}).status_code == 404
    assert client.delete("/rubrics/rubric_nope").status_code == 404


# -- the agent link -------------------------------------------------------


def test_agent_resolves_its_rubric_into_a_plan(client: TestClient, tmp_path: object) -> None:
    """
    The console field has to reach the runtime, which is the whole point of the resource.

    Asserted through `resolve_agent` rather than through a live session because this is the seam
    that was missing for every other resource before it was wired: a stored setting the
    conversation ignores is a demo of a console.
    """
    rubric = client.post("/rubrics", json=BACKEND).json()
    agent = client.post(
        "/agents", json={"name": "Interviewer", "rubric_id": rubric["id"]}
    ).json()

    resolved = resolve_agent(agent["id"], data=Store(tmp_path))  # type: ignore[arg-type]
    assert resolved.plan.name == "Backend engineer"
    assert [c.id for c in resolved.plan.competencies] == [
        "debugging-under-pressure",
        "data-at-scale",
    ]
    assert resolved.plan.competencies[1].max_turns == 2


def test_agent_pointing_at_a_deleted_rubric_refuses_to_start(
    client: TestClient, tmp_path: object
) -> None:
    """
    Loud rather than degrading, matching every other resource.

    An interview that quietly runs with no plan looks completely normal and covers nothing in
    particular, which is the failure class that is hardest to notice.
    """
    rubric = client.post("/rubrics", json=BACKEND).json()
    agent = client.post(
        "/agents", json={"name": "Interviewer", "rubric_id": rubric["id"]}
    ).json()
    client.delete(f"/rubrics/{rubric['id']}")

    with pytest.raises(AgentNotConfigured, match="does not exist"):
        resolve_agent(agent["id"], data=Store(tmp_path))  # type: ignore[arg-type]


def test_agent_without_a_rubric_gets_an_inactive_plan(
    client: TestClient, tmp_path: object
) -> None:
    """A clean clone with no rubric must still hold a conversation."""
    agent = client.post("/agents", json={"name": "Plain"}).json()
    resolved = resolve_agent(agent["id"], data=Store(tmp_path))  # type: ignore[arg-type]
    assert resolved.plan.active is False


def test_a_rubric_can_be_detached_from_an_agent(client: TestClient) -> None:
    """
    Sending `rubric_id: null` clears it; omitting the key leaves it alone.

    Both halves matter and they used to be the same request. The store dropped nulls, so
    "detach" silently did nothing — which would have made the console's picker offer a "none"
    option that appeared to work and changed nothing. `exclude_unset` is what distinguishes
    them, so this asserts the distinction rather than just the clear.
    """
    rubric = client.post("/rubrics", json=BACKEND).json()
    agent = client.post(
        "/agents", json={"name": "Interviewer", "rubric_id": rubric["id"]}
    ).json()

    # Omitting the key must not disturb it.
    untouched = client.patch(f"/agents/{agent['id']}", json={"name": "Renamed"}).json()
    assert untouched["rubric_id"] == rubric["id"]

    detached = client.patch(f"/agents/{agent['id']}", json={"rubric_id": None}).json()
    assert detached["rubric_id"] is None
