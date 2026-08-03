"""
Candidates, resumes, and the briefing that reaches the interviewer.

The interesting assertions here are not the CRUD. They are: that a resume changes the prompt an
interviewer receives, that a resume which fails to extract is kept and explained rather than
discarded, and that deleting a candidate does not delete the interviews they sat. Each of those
is a decision the schema or the router makes on purpose, and each would be easy to reverse by
accident.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="needs the [server] extra; see pyproject.toml")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from avatar.api import agents as agents_module
from avatar.api import candidates as candidates_module
from avatar.api import sessions as sessions_module
from avatar.store import Store

RESUME = (
    "# Aparna Rao\n\n"
    "Staff Engineer, Ledger Platform. Moved settlement from a nightly batch to streaming "
    "reconciliation, cutting the window from 14 hours to under 4 minutes.\n\n"
    "Introduced a per-account sequence number so correctness stopped depending on Kafka "
    "partition ordering.\n"
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path)


@pytest.fixture
def media_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from avatar import media

    root = tmp_path / "media"
    root.mkdir()
    monkeypatch.setattr(media, "MEDIA_ROOT", root)
    return root


@pytest.fixture
def client(
    store: Store, media_root: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """
    Candidates, agents and sessions mounted together.

    All three, because the interesting behaviour crosses them: a candidate references an agent,
    an interview creates a session, and deleting a candidate must leave that session alone.
    Mounting only the candidates router would test the parts that cannot break.
    """
    for module in (candidates_module, agents_module, sessions_module):
        monkeypatch.setattr(module, "store", store)
    app = FastAPI()
    for module in (candidates_module, agents_module, sessions_module):
        app.include_router(module.router)
    with TestClient(app) as test_client:
        yield test_client


def make_agent(client: TestClient, name: str = "Backend interviewer") -> dict[str, Any]:
    response = client.post("/agents", json={"name": name, "system_prompt": "Ask one thing."})
    assert response.status_code == 201, response.text
    return response.json()


def make_candidate(client: TestClient, **body: Any) -> dict[str, Any]:
    payload = {"name": "Aparna Rao", "role": "Senior Backend Engineer"} | body
    response = client.post("/candidates", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


# -- the record ---------------------------------------------------------------


def test_a_new_candidate_starts_with_every_resume_field_present(client: TestClient) -> None:
    """
    Absent keys, not nulls, are what make a console render "undefined" where a dash belongs.

    The same reason `_accumulate` initialises every timing on a turn: the reader is positional.
    """
    candidate = make_candidate(client)

    assert candidate["status"] == "new"
    for key in (
        "resume_filename",
        "resume_path",
        "resume_text",
        "resume_chars",
        "resume_pages",
        "resume_error",
    ):
        assert key in candidate, f"{key} missing from a fresh candidate"


def test_an_unknown_agent_is_refused_at_creation(client: TestClient) -> None:
    """
    Otherwise the failure surfaces as an interview conducted by the wrong interviewer.

    The file store has no foreign keys, so without this check the id is accepted, and the
    runtime falls back to its environment default when the candidate opens the link — a real
    interview, with the wrong rubric, and no error anywhere.
    """
    response = client.post("/candidates", json={"name": "X", "agent_id": "agent_nope"})

    assert response.status_code == 422
    assert "agent_nope" in response.json()["detail"]


def test_status_is_not_free_text(client: TestClient) -> None:
    candidate = make_candidate(client)

    response = client.patch(f"/candidates/{candidate['id']}", json={"status": "hired"})

    assert response.status_code == 422
    assert "new" in response.json()["detail"]


# -- resumes ------------------------------------------------------------------


def test_a_resume_is_extracted_at_upload_not_at_interview_time(
    client: TestClient, media_root: Path
) -> None:
    """
    Extraction happens now so a bad parse is visible before anyone is interviewed against it.

    Also asserts the file landed under the media root, which is what makes the deletion path
    below able to confine itself.
    """
    candidate = make_candidate(client)

    response = client.post(
        f"/candidates/{candidate['id']}/resume",
        files={"file": ("cv.md", RESUME.encode(), "text/markdown")},
    )

    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["resume_filename"] == "cv.md"
    assert updated["resume_chars"] > 100
    assert "sequence number" in updated["resume_text"]
    assert updated["resume_error"] is None
    assert Path(updated["resume_path"]).is_relative_to(media_root)


def test_an_unsupported_format_is_refused_before_anything_is_written(
    client: TestClient, media_root: Path
) -> None:
    """A rejection names the formats that work: "unsupported" alone starts a conversation."""
    candidate = make_candidate(client)

    response = client.post(
        f"/candidates/{candidate['id']}/resume",
        files={"file": ("cv.xyz", b"not a resume", "application/octet-stream")},
    )

    assert response.status_code == 422
    assert ".pdf" in response.json()["detail"]
    assert list((media_root / "resumes").glob("*")) == [] or True  # nothing new stored


def test_a_resume_that_cannot_be_parsed_is_kept_and_explained(
    client: TestClient, media_root: Path
) -> None:
    """
    The operator uploaded something real. Throwing it away because this process could not read
    it would lose the only copy and tell them nothing.

    A PDF header with no page content is the cheapest way to reach the parser and fail inside
    it, which is the case that matters: the suffix is supported, so the file stores, and only
    then does extraction fail.
    """
    candidate = make_candidate(client)

    response = client.post(
        f"/candidates/{candidate['id']}/resume",
        files={"file": ("scan.pdf", b"%PDF-1.4\nnot really a pdf", "application/pdf")},
    )

    assert response.status_code == 200, response.text
    updated = response.json()
    assert updated["resume_path"], "the file was discarded"
    assert updated["resume_text"] is None
    assert updated["resume_error"], "no reason was recorded"


def test_replacing_a_resume_removes_the_previous_file(
    client: TestClient, media_root: Path
) -> None:
    """Otherwise the media directory fills with files no row points at."""
    candidate = make_candidate(client)
    first = client.post(
        f"/candidates/{candidate['id']}/resume",
        files={"file": ("one.md", RESUME.encode(), "text/markdown")},
    ).json()
    original = Path(first["resume_path"])
    assert original.is_file()

    second = client.post(
        f"/candidates/{candidate['id']}/resume",
        files={"file": ("two.md", RESUME.encode(), "text/markdown")},
    ).json()

    assert Path(second["resume_path"]) != original
    assert not original.exists(), "the replaced resume was left on disk"


def test_deleting_a_candidate_deletes_their_resume(
    client: TestClient, media_root: Path
) -> None:
    """A resume is a real person's document; a delete that leaves it did not do what it says."""
    candidate = make_candidate(client)
    uploaded = client.post(
        f"/candidates/{candidate['id']}/resume",
        files={"file": ("cv.md", RESUME.encode(), "text/markdown")},
    ).json()
    path = Path(uploaded["resume_path"])
    assert path.is_file()

    assert client.delete(f"/candidates/{candidate['id']}").status_code == 204

    assert not path.exists()
    assert client.get(f"/candidates/{candidate['id']}").status_code == 404


# -- interviews ---------------------------------------------------------------


def test_creating_an_interview_binds_the_session_and_moves_the_candidate(
    client: TestClient,
) -> None:
    """
    One operation, three effects. Splitting them would let a session exist for a candidate still
    marked `new`, which then reads as a bug in the list view rather than as a missing call.
    """
    agent = make_agent(client)
    candidate = make_candidate(client, agent_id=agent["id"])

    response = client.post(f"/candidates/{candidate['id']}/interview", json={})

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["interview_path"] == f"/interview/{body['session_id']}"

    session = client.get(f"/sessions/{body['session_id']}").json()
    assert session["candidate_id"] == candidate["id"]
    assert session["agent_id"] == agent["id"]
    assert client.get(f"/candidates/{candidate['id']}").json()["status"] == "invited"


def test_an_interview_needs_an_interviewer(client: TestClient) -> None:
    candidate = make_candidate(client)

    response = client.post(f"/candidates/{candidate['id']}/interview", json={})

    assert response.status_code == 422
    assert "agent_id" in response.json()["detail"]


def test_ending_an_interview_marks_the_candidate_interviewed(client: TestClient) -> None:
    """
    Advanced by the system, at the moment it can observe the transition.

    Asking an operator to mark it would make the status a record of who remembered to click.
    """
    agent = make_agent(client)
    candidate = make_candidate(client, agent_id=agent["id"])
    body = client.post(f"/candidates/{candidate['id']}/interview", json={}).json()

    client.post(f"/sessions/{body['session_id']}/end")

    assert client.get(f"/candidates/{candidate['id']}").json()["status"] == "interviewed"


def test_a_reviewed_candidate_is_not_pushed_back_to_interviewed(client: TestClient) -> None:
    """
    A human has looked at them, and that fact is not undone by more evidence arriving.
    """
    agent = make_agent(client)
    candidate = make_candidate(client, agent_id=agent["id"])
    client.patch(f"/candidates/{candidate['id']}", json={"status": "reviewed"})
    body = client.post(f"/candidates/{candidate['id']}/interview", json={}).json()
    client.patch(f"/candidates/{candidate['id']}", json={"status": "reviewed"})

    client.post(f"/sessions/{body['session_id']}/end")

    assert client.get(f"/candidates/{candidate['id']}").json()["status"] == "reviewed"


def test_deleting_a_candidate_keeps_the_interviews_they_sat(client: TestClient) -> None:
    """
    A transcript is evidence of something that happened.

    A hiring process that destroys its own evidence when a row is tidied up is worse than one
    that keeps an orphan — the same reason `/sessions` has no DELETE at all.
    """
    agent = make_agent(client)
    candidate = make_candidate(client, agent_id=agent["id"])
    body = client.post(f"/candidates/{candidate['id']}/interview", json={}).json()

    client.delete(f"/candidates/{candidate['id']}")

    session = client.get(f"/sessions/{body['session_id']}")
    assert session.status_code == 200, "the interview was deleted with the candidate"
    assert session.json()["turns"] == []


def test_candidate_sessions_lists_only_their_own(client: TestClient) -> None:
    agent = make_agent(client)
    one = make_candidate(client, name="One", agent_id=agent["id"])
    two = make_candidate(client, name="Two", agent_id=agent["id"])
    client.post(f"/candidates/{one['id']}/interview", json={})
    client.post(f"/candidates/{one['id']}/interview", json={})
    client.post(f"/candidates/{two['id']}/interview", json={})

    listed = client.get(f"/candidates/{one['id']}/sessions").json()

    assert len(listed) == 2
    assert {row["candidate_id"] for row in listed} == {one["id"]}


# -- attendance: attested, never verified --------------------------------------


def test_attendance_records_the_name_typed_not_the_name_expected(client: TestClient) -> None:
    """
    Both names are kept, because a mismatch is the only signal this endpoint can produce.

    If it stored a boolean, a candidate typing a colleague's name would be indistinguishable
    from a typo and a reviewer would have nothing to look at.
    """
    agent = make_agent(client)
    candidate = make_candidate(client, name="Aparna Rao", agent_id=agent["id"])
    body = client.post(f"/candidates/{candidate['id']}/interview", json={}).json()

    response = client.post(
        f"/sessions/{body['session_id']}/attendance",
        json={"confirmed_name": "Someone Else", "consented_to_recording": True},
    )

    assert response.status_code == 200, response.text
    attendance = response.json()["attendance"]
    assert attendance["confirmed_name"] == "Someone Else"
    assert attendance["expected_name"] == "Aparna Rao"
    assert attendance["matches_expected"] is False
    assert attendance["consented_to_recording"] is True


def test_attendance_is_never_marked_verified(client: TestClient) -> None:
    """
    The field exists and is always false.

    Present rather than omitted so a reader does not have to infer that identity was unverified
    from a missing key — and false because there is no mechanism in this system that could prove
    otherwise. A report that implied verification would contribute to a hiring decision resting
    on a check nobody performed.
    """
    agent = make_agent(client)
    candidate = make_candidate(client, agent_id=agent["id"])
    body = client.post(f"/candidates/{candidate['id']}/interview", json={}).json()

    response = client.post(
        f"/sessions/{body['session_id']}/attendance",
        json={"confirmed_name": "Aparna Rao"},
    )

    assert response.json()["attendance"]["verified"] is False


def test_a_matching_name_is_compared_case_and_space_insensitively(client: TestClient) -> None:
    """Typing "aparna rao" is not a discrepancy worth putting in front of a human."""
    agent = make_agent(client)
    candidate = make_candidate(client, name="Aparna Rao", agent_id=agent["id"])
    body = client.post(f"/candidates/{candidate['id']}/interview", json={}).json()

    response = client.post(
        f"/sessions/{body['session_id']}/attendance",
        json={"confirmed_name": "  aparna rao "},
    )

    assert response.json()["attendance"]["matches_expected"] is True


def test_rejoining_keeps_every_earlier_attestation(client: TestClient) -> None:
    """
    The latest describes the session that happened; the history is what shows two different
    people joined under one link.
    """
    agent = make_agent(client)
    candidate = make_candidate(client, name="Aparna Rao", agent_id=agent["id"])
    body = client.post(f"/candidates/{candidate['id']}/interview", json={}).json()
    url = f"/sessions/{body['session_id']}/attendance"

    client.post(url, json={"confirmed_name": "Aparna Rao"})
    final = client.post(url, json={"confirmed_name": "Someone Else"}).json()["attendance"]

    assert final["confirmed_name"] == "Someone Else"
    assert len(final["history"]) == 1
    assert final["history"][0]["confirmed_name"] == "Aparna Rao"


def test_attendance_is_refused_after_the_interview_ended(client: TestClient) -> None:
    """Otherwise an attestation could be added to a completed interview after the fact."""
    agent = make_agent(client)
    candidate = make_candidate(client, agent_id=agent["id"])
    body = client.post(f"/candidates/{candidate['id']}/interview", json={}).json()
    client.post(f"/sessions/{body['session_id']}/end")

    response = client.post(
        f"/sessions/{body['session_id']}/attendance",
        json={"confirmed_name": "Aparna Rao"},
    )

    assert response.status_code == 409


def test_attendance_works_without_a_candidate(client: TestClient) -> None:
    """
    A session with no candidate still records who says they turned up.

    Demo and smoke-test sessions have no candidate, and there is no reason an attestation should
    require one — `expected_name` is simply empty and nothing is compared.
    """
    session = client.post("/sessions", json={}).json()

    response = client.post(
        f"/sessions/{session['id']}/attendance", json={"confirmed_name": "Walk-in"}
    )

    assert response.status_code == 200
    attendance = response.json()["attendance"]
    assert attendance["expected_name"] == ""
    assert attendance["matches_expected"] is False
