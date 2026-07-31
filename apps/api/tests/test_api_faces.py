"""
Tests for the faces router.

Every test runs against a `Store` rooted at `tmp_path`, injected by rebinding the router's
module-level `store`. Nothing here can reach a real data directory, which matters more for
this resource than most: a stray write would leave a face on disk claiming an enrollment
measurement that no run produced.

The renderer is injected the same way, by rebinding `faces.build`. That is not laziness about
using the real stub — most tests do use it — it is the only way to exercise the failure path
at all, because `StubRenderer.prepare_identity` cannot fail. A fake that raises is what proves
a bad reference clip becomes a stored `failed` row rather than a 500.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# The web stack is an optional extra: `pyproject.toml` keeps fastapi out of `[dev]` so CI can
# type-check and test the orchestration layer with nothing installed but pytest. Skipping here
# rather than importing unconditionally is what stops that choice from turning into a
# collection error — the trade-off, stated plainly, is that these tests only run where the
# server extra is installed.
pytest.importorskip("fastapi", reason="needs the [server] extra; see pyproject.toml")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from avatar.api import faces
from avatar.store import Store

REFERENCE = "media/ada.mp4"


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path)


@pytest.fixture
def client(store: Store, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """
    A client for an app containing only this router.

    Built here rather than imported from `avatar.server`: importing the server would load
    every adapter and read the environment's `.env` files, which is a lot of unrelated
    machinery for a CRUD test and one more thing that can fail for reasons that are not the
    router's fault.
    """
    monkeypatch.setattr(faces, "store", store)
    app = FastAPI()
    app.include_router(faces.router)
    with TestClient(app) as test_client:
        yield test_client


def make_face(
    client: TestClient, *, name: str = "Ada", path: str = REFERENCE
) -> dict[str, Any]:
    response = client.post("/faces", json={"name": name, "reference_path": path})
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


class RecordingRenderer:
    """Records what it was asked to enroll. Only `prepare_identity` is ever called."""

    def __init__(self, identity: object = object()) -> None:
        self.identity = identity
        self.paths: list[str] = []

    def prepare_identity(self, reference_path: str) -> object:
        self.paths.append(reference_path)
        return self.identity


def use_renderer(
    monkeypatch: pytest.MonkeyPatch,
    renderer: object,
    configs: list[Any] | None = None,
) -> None:
    """Rebind `faces.build` so prepare runs against `renderer`."""

    def fake_build(config: Any = None) -> object:
        if configs is not None:
            configs.append(config)
        return renderer

    monkeypatch.setattr(faces, "build", fake_build)


# -- create -----------------------------------------------------------------------------


def test_create_starts_queued_with_every_measurement_null(client: TestClient) -> None:
    """
    A new face must be unenrolled and carry no numbers.

    Prevents the defect that matters most here: a face that looks ready, or that shows an
    enrollment cost, before any renderer has run. The four derived keys are asserted present
    rather than merely falsy so the console never has to tell "not measured" from "key
    missing".
    """
    face = make_face(client)

    assert face["status"] == "queued"
    assert face["enrollment_ms"] is None
    assert face["frame_count"] is None
    assert face["failure_reason"] is None
    assert face["id"].startswith("face_")
    assert face["created_at"] == face["updated_at"]


def test_create_strips_surrounding_whitespace(client: TestClient) -> None:
    """
    Prevents a pasted name or path keeping its trailing newline — which makes a face look
    duplicated in the list and, for the path, sends the renderer a filename that does not
    exist for a reason nobody can see.
    """
    face = make_face(client, name="  Ada  ", path="  media/ada.mp4\n")

    assert face["name"] == "Ada"
    assert face["reference_path"] == "media/ada.mp4"


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "ready"),
        ("enrollment_ms", 42),
        ("frame_count", 900),
        ("failure_reason", "none"),
    ],
)
def test_create_refuses_client_supplied_findings(
    client: TestClient, field: str, value: object
) -> None:
    """
    Prevents a client declaring a face enrolled without an enrollment having happened.

    Each of these four fields is produced by a prepare run. Accepting any of them would put an
    unmeasured latency figure into the record — the single worst outcome this project defines
    — so the request is rejected rather than silently ignored.
    """
    response = client.post(
        "/faces", json={"name": "Ada", "reference_path": REFERENCE, field: value}
    )

    assert response.status_code == 422, response.text


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_create_refuses_a_blank_name(client: TestClient, blank: str) -> None:
    """
    Prevents a whitespace-only name, which is worse than an empty one: `min_length=1` accepts
    a single space, and the row then renders blank and cannot be found by anyone scanning the
    list for it.
    """
    response = client.post("/faces", json={"name": blank, "reference_path": REFERENCE})

    assert response.status_code == 422, response.text


@pytest.mark.parametrize("body", [{"name": "Ada"}, {"name": "Ada", "reference_path": "  "}])
def test_create_requires_a_usable_reference_path(
    client: TestClient, body: dict[str, Any]
) -> None:
    """
    Prevents a face with nothing to enroll from reaching disk, whether the path was omitted or
    was whitespace. Such a record can only ever produce a failed prepare, and it would produce
    it much later, in front of a different person.
    """
    response = client.post("/faces", json=body)

    assert response.status_code == 422, response.text


# -- read -------------------------------------------------------------------------------


def test_list_is_newest_first(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Prevents the list view burying a just-created face.

    `created_at` has second precision, so two faces created in the same second tie and the
    order falls back to filename. The clock is stepped here to make the ordering rule itself
    testable rather than accidentally passing on timing.
    """
    stamps = iter(["2026-07-01T00:00:00+00:00", "2026-07-02T00:00:00+00:00"])
    monkeypatch.setattr("avatar.store.now_iso", lambda: next(stamps))

    first = make_face(client, name="Older")
    second = make_face(client, name="Newer")

    listed = client.get("/faces").json()
    assert [face["id"] for face in listed] == [second["id"], first["id"]]


def test_get_returns_the_stored_record(client: TestClient) -> None:
    """Prevents a read path that reformats or drops fields the write path stored."""
    face = make_face(client)

    response = client.get(f"/faces/{face['id']}")

    assert response.status_code == 200
    assert response.json() == face


def test_get_unknown_id_is_404(client: TestClient) -> None:
    """Prevents `NotFound` escaping as a 500, which would report a client mistake as a bug."""
    response = client.get("/faces/face_missing")

    assert response.status_code == 404
    assert "face_missing" in response.json()["detail"]


# -- update -----------------------------------------------------------------------------


def test_update_renames_without_disturbing_enrollment(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Prevents a rename resetting enrollment. Relabelling is the one edit that cannot invalidate
    a measurement, so a ready face must stay ready — otherwise every typo fix costs a re-run.
    """
    use_renderer(monkeypatch, RecordingRenderer())
    face = make_face(client)
    enroll(client, face["id"])

    response = client.patch(f"/faces/{face['id']}", json={"name": "Ada Lovelace"})

    assert response.status_code == 200
    assert response.json()["name"] == "Ada Lovelace"
    assert response.json()["status"] == "ready"
    assert response.json()["enrollment_ms"] is not None


def test_update_refuses_to_repoint_the_reference(client: TestClient) -> None:
    """
    Prevents a measurement outliving the media it was measured from.

    `status`, `enrollment_ms` and `frame_count` describe one specific clip, and the store's
    merge cannot clear them in the same write. Rejecting the patch is what stops a ready face
    reporting an enrollment cost for a file it no longer points at.
    """
    face = make_face(client)

    response = client.patch(f"/faces/{face['id']}", json={"reference_path": "media/other.mp4"})

    assert response.status_code == 422, response.text
    assert client.get(f"/faces/{face['id']}").json()["reference_path"] == REFERENCE


def test_update_refuses_an_empty_patch(client: TestClient) -> None:
    """
    Prevents `updated_at` moving for a request that changed nothing. That column is the only
    evidence an operator has about when a face last changed; a no-op bump makes it lie.
    """
    face = make_face(client)

    response = client.patch(f"/faces/{face['id']}", json={})

    assert response.status_code == 422
    assert client.get(f"/faces/{face['id']}").json()["updated_at"] == face["updated_at"]


def test_update_refuses_a_blank_name(client: TestClient) -> None:
    """Prevents create's name rule and update's name rule drifting apart."""
    face = make_face(client)

    response = client.patch(f"/faces/{face['id']}", json={"name": "   "})

    assert response.status_code == 422, response.text


def test_update_unknown_id_is_404(client: TestClient) -> None:
    """Prevents a patch against a deleted face creating one, or 500ing."""
    response = client.patch("/faces/face_missing", json={"name": "Ada"})

    assert response.status_code == 404


# -- delete -----------------------------------------------------------------------------


def test_delete_removes_the_face(client: TestClient) -> None:
    """Prevents a delete that reports success while leaving the record readable."""
    face = make_face(client)

    assert client.delete(f"/faces/{face['id']}").status_code == 204
    assert client.get(f"/faces/{face['id']}").status_code == 404
    assert client.get("/faces").json() == []


def test_delete_unknown_id_is_404(client: TestClient) -> None:
    """Prevents a double delete looking like a server fault."""
    assert client.delete("/faces/face_missing").status_code == 404


def enroll(client: TestClient, face_id: str) -> dict[str, Any]:
    """
    Start enrollment, wait for the worker, and return the finished record.

    `POST /prepare` answers 202 and does the work on a thread -- it takes minutes with a real
    renderer, so holding the connection open timed out proxies and left rows stuck in
    `preparing`
    when a process died. These tests want the outcome, so they wait for it explicitly rather
    than
    making the endpoint synchronous again: the asynchrony is the behaviour under test everywhere
    else, and a fixture that quietly removed it would test something the product does not do.
    """
    from avatar import jobs

    accepted = client.post(f"/faces/{face_id}/prepare")
    assert accepted.status_code == 202, accepted.text
    assert jobs.wait_for_idle(30.0), "the enrollment job did not finish"
    return client.get(f"/faces/{face_id}").json()


# -- prepare ----------------------------------------------------------------------------


def test_prepare_enrolls_with_the_stub_renderer_and_records_a_measurement(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Prevents the two ways prepare can be wrong about its own inputs and outputs: enrolling
    with something other than the GPU-free stub (which is all this milestone has), and
    reporting a status of ready without an `enrollment_ms` beside it.

    The bound is `>= 0`, not a specific figure. The stub's prepare is sub-millisecond, so 0 is
    the truthful value here; asserting anything larger would be asserting a fabrication.
    """
    configs: list[Any] = []
    renderer = RecordingRenderer()
    use_renderer(monkeypatch, renderer, configs)
    face = make_face(client)

    body = enroll(client, face["id"])

    assert body["status"] == "ready"
    assert isinstance(body["enrollment_ms"], int)
    assert body["enrollment_ms"] >= 0
    assert [config.name for config in configs] == ["stub"]
    assert renderer.paths == [REFERENCE]


def test_prepare_with_the_real_stub_renderer_reaches_ready(client: TestClient) -> None:
    """
    Prevents the whole endpoint being proved only against a test double.

    No `use_renderer` here: this is `avatar.renderers.build` and `StubRenderer` for real, which
    is what shows the router calls the renderer contract correctly rather than calling a fake
    that was shaped to fit it.
    """
    face = make_face(client)

    body = enroll(client, face["id"])

    assert body["status"] == "ready"
    assert body["enrollment_ms"] is not None


def test_prepare_writes_preparing_before_touching_the_renderer(
    client: TestClient, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Prevents a process killed mid-enrollment leaving a record that still claims to be queued.

    The renderer reads the record back while it is being called, which is the only moment the
    intermediate state is observable through the store.
    """
    observed: list[str] = []
    face = make_face(client)

    class ObservingRenderer:
        def prepare_identity(self, reference_path: str) -> object:
            observed.append(str(store.get("faces", face["id"])["status"]))
            return object()

    use_renderer(monkeypatch, ObservingRenderer())

    enroll(client, face["id"])

    assert observed == ["preparing"]


def test_prepare_records_a_frame_count_the_artifact_reports(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Prevents the frame count being dropped once a renderer does report one — the field exists
    for the real renderer, and a silently ignored value would look like the model failing to
    read the clip.
    """

    class CountedIdentity:
        frame_count = 900

    use_renderer(monkeypatch, RecordingRenderer(CountedIdentity()))
    face = make_face(client)

    assert enroll(client, face["id"])["frame_count"] == 900


def test_prepare_leaves_frame_count_null_when_the_artifact_has_none(client: TestClient) -> None:
    """
    Prevents a plausible-looking frame count being invented for an artifact that does not
    report one. `StubIdentity` carries only a path, so the honest answer is an empty cell.
    """
    face = make_face(client)

    assert enroll(client, face["id"])["frame_count"] is None


def test_prepare_records_a_renderer_failure_instead_of_raising(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Prevents a bad reference clip becoming a 500.

    Pointing a model at operator-supplied media fails routinely; a 500 would file that next to
    real bugs and leave the operator with an unchanged row and no explanation. The reason must
    reach the record, and the status must not stay `preparing`.
    """

    class FailingRenderer:
        def prepare_identity(self, reference_path: str) -> object:
            raise RuntimeError("no face detected in reference clip")

    use_renderer(monkeypatch, FailingRenderer())
    face = make_face(client)

    body = enroll(client, face["id"])

    assert body["status"] == "failed"
    assert "no face detected in reference clip" in body["failure_reason"]
    assert "RuntimeError" in body["failure_reason"]
    assert body["enrollment_ms"] is None


def test_prepare_records_a_renderer_that_cannot_be_built(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Prevents a face stranded in `preparing` when construction is what fails.

    `build` raises for an unknown renderer, and a real one raises when its weights are absent.
    Because `preparing` is written first, a construction failure outside the try block would
    leave a record that nothing can retry and nothing explains.
    """

    def failing_build(config: Any = None) -> object:
        raise ValueError("unknown renderer 'musetalk'; weights not on disk")

    monkeypatch.setattr(faces, "build", failing_build)
    face = make_face(client)

    body = enroll(client, face["id"])

    assert body["status"] == "failed"
    assert "weights not on disk" in body["failure_reason"]


def test_a_failed_face_can_be_prepared_again(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Prevents one bad run being terminal. The console offers Prepare on failed rows, so a retry
    after fixing the clip must reach ready — a 409 here would force a delete and lose the
    record's history.

    Also asserts that the previous `failure_reason` is gone. It used to survive a successful
    retry, because the store dropped nulls and so the field could not be unset -- leaving a
    record that read `ready` while still carrying the reason it failed two attempts ago. That
    was safe only for as long as every reader checked `status` first, which is a rule a record
    should not need. The store now writes nulls, so the record simply stops contradicting
    itself.
    """

    class FlakyRenderer:
        def __init__(self) -> None:
            self.calls = 0

        def prepare_identity(self, reference_path: str) -> object:
            self.calls += 1
            if self.calls == 1:
                raise OSError("could not open reference clip")
            return object()

    use_renderer(monkeypatch, FlakyRenderer())
    face = make_face(client)

    assert enroll(client, face["id"])["status"] == "failed"
    retried = enroll(client, face["id"])

    assert retried["status"] == "ready"
    assert retried["enrollment_ms"] is not None
    assert retried["failure_reason"] is None


def test_prepare_refuses_a_face_already_preparing(client: TestClient, store: Store) -> None:
    """
    Prevents a double-clicked button running two enrollments against one record, where the
    loser's result silently overwrites the winner's.
    """
    face = make_face(client)
    store.update("faces", face["id"], {"status": "preparing"})

    response = client.post(f"/faces/{face['id']}/prepare")

    assert response.status_code == 409
    assert "preparing" in response.json()["detail"]


def test_prepare_refuses_a_ready_face(client: TestClient) -> None:
    """
    Prevents a recorded measurement being overwritten in place. Refusing means the
    `enrollment_ms` on a ready face is always the number produced by the run that made it
    ready, rather than whichever run happened last.
    """
    face = make_face(client)
    first = enroll(client, face["id"])

    response = client.post(f"/faces/{face['id']}/prepare")

    assert response.status_code == 409
    assert client.get(f"/faces/{face['id']}").json()["enrollment_ms"] == first["enrollment_ms"]


def test_prepare_unknown_id_is_404(client: TestClient) -> None:
    """
    Prevents a missing face being treated as a preparable one — the only outcome of this
    endpoint that is a genuine client error rather than a stored failure.
    """
    assert client.post("/faces/face_missing/prepare").status_code == 404


def test_prepare_is_dispatched_off_the_event_loop() -> None:
    """
    Prevents `prepare_face` being changed to `async def`.

    Enrollment is synchronous and allowed to be slow. On the event loop it would block every
    live WebSocket session for its duration; declared as a plain `def`, FastAPI runs it in a
    threadpool instead. That is invisible in a passing CRUD test, so it is asserted directly.
    """
    import inspect

    assert not inspect.iscoroutinefunction(faces.prepare_face)


def test_router_is_mounted_under_faces() -> None:
    """Prevents a prefix or tag change silently moving every URL the console fetches."""
    assert faces.router.prefix == "/faces"
    assert faces.router.tags == ["faces"]
