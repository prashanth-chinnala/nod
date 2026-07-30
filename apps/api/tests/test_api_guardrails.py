"""
Behaviour of the guardrails router and the local policy check.

Two things this file is careful about:

  1. **No real data directory.** Every test binds a `Store(tmp_path)` through
     `dependency_overrides`, so a suite run can never read or overwrite a real console
     record. The process-wide `store` is never touched.
  2. **Deterministic list order.** `Store.list` sorts on a second-precision timestamp, so
     two records created in the same second tie and fall back to filename order. The
     `clock` fixture makes each write's timestamp strictly later, which is what lets the
     ordering assertion mean "newest first" rather than "happened to pass".
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="console routers need the [server] extra")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from avatar.api.guardrails import (
    DEFAULT_MAX_ANSWER_CHARS,
    EMAIL_PLACEHOLDER,
    MAX_ANSWER_CHARS_CEILING,
    NUMBER_PLACEHOLDER,
    Policy,
    evaluate,
    get_store,
    router,
)
from avatar.store import Store


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Strictly increasing creation timestamps.

    Without this the newest-first assertion is a coin flip: `now_iso` has one-second
    resolution and a test creates several records inside one second, so the sort key ties
    and `Store.list` falls back to whatever `glob` returned.
    """
    ticks = iter(f"2026-07-29T12:00:{second:02d}+00:00" for second in range(60))
    monkeypatch.setattr("avatar.store.now_iso", lambda: next(ticks))


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_store] = lambda: Store(tmp_path)
    with TestClient(app) as test_client:
        yield test_client


def create(client: TestClient, **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"name": "House policy"}
    body.update(overrides)
    response = client.post("/guardrails", json=body)
    assert response.status_code == 201, response.text
    created: dict[str, Any] = response.json()
    return created


# -- CRUD ---------------------------------------------------------------------


def test_create_applies_documented_defaults(client: TestClient) -> None:
    """
    Prevents a create that silently stores a policy enforcing nothing.

    Every default in the spec is load-bearing: a record written without them evaluates as
    "no cap, no redaction, no refusal text" while looking configured in the table.
    """
    record = create(client, name="  House policy  ")

    assert record["id"].startswith("guard_")
    assert record["name"] == "House policy", "name must be stripped before storage"
    assert record["banned_topics"] == []
    assert record["pii_redaction"] is False
    assert record["max_answer_chars"] == DEFAULT_MAX_ANSWER_CHARS
    assert record["on_violation"] == "refuse"
    assert record["refusal_message"].strip() != ""
    assert record["created_at"] == record["updated_at"]


def test_list_is_newest_first(client: TestClient, clock: None) -> None:
    """
    Prevents a table that buries the guardrail just created at the bottom of the page.

    The console renders this order verbatim, so the ordering contract lives here rather
    than in the page.
    """
    first = create(client, name="First")
    second = create(client, name="Second")
    third = create(client, name="Third")

    listed = client.get("/guardrails").json()

    assert [row["id"] for row in listed] == [third["id"], second["id"], first["id"]]


def test_list_is_empty_before_anything_is_created(client: TestClient) -> None:
    """Prevents a 500 on a cold install, where the collection directory does not exist."""
    response = client.get("/guardrails")

    assert response.status_code == 200
    assert response.json() == []


def test_get_returns_the_stored_record(client: TestClient) -> None:
    """Prevents a read path that reconstructs defaults instead of returning what was saved."""
    record = create(client, banned_topics=["salary"], pii_redaction=True)

    fetched = client.get(f"/guardrails/{record['id']}").json()

    assert fetched == record


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("get", {}),
        ("patch", {"json": {"name": "Renamed"}}),
        ("delete", {}),
        ("post", {"json": {"text": "hello", "direction": "input"}}),
    ],
)
def test_unknown_id_is_404_not_500(
    client: TestClient, method: str, kwargs: dict[str, Any]
) -> None:
    """
    Prevents `NotFound` escaping as a 500.

    A stale browser tab pointing at a deleted guardrail is routine; paging someone because
    it looks like a server fault is not.
    """
    path = "/guardrails/guard_missing"
    if method == "post":
        path += "/check"

    response = client.request(method, path, **kwargs)

    assert response.status_code == 404


def test_update_merges_and_preserves_identity(client: TestClient, clock: None) -> None:
    """
    Prevents a patch that replaces the record wholesale, or that lets `id`/`created_at` move.

    Both would break the list ordering and, in the id case, silently turn one guardrail into
    another while sessions still reference the old one.
    """
    record = create(client, banned_topics=["salary"], max_answer_chars=400)

    updated = client.patch(
        f"/guardrails/{record['id']}",
        json={"max_answer_chars": 900, "on_violation": "end_session"},
    ).json()

    assert updated["id"] == record["id"]
    assert updated["created_at"] == record["created_at"]
    assert updated["max_answer_chars"] == 900
    assert updated["on_violation"] == "end_session"
    assert updated["banned_topics"] == ["salary"], "untouched fields must survive a patch"
    assert updated["updated_at"] != record["created_at"]


def test_update_can_clear_banned_topics_with_an_empty_list(client: TestClient) -> None:
    """
    Prevents an unremovable ban.

    `Store.update` drops `None` from a patch, so `[]` is the only way to clear this field.
    If the router filtered empty lists too, a topic added by mistake could never be removed
    except by deleting the guardrail.
    """
    record = create(client, banned_topics=["salary", "visa"])

    updated = client.patch(f"/guardrails/{record['id']}", json={"banned_topics": []}).json()

    assert updated["banned_topics"] == []


def test_delete_removes_the_record(client: TestClient) -> None:
    """Prevents a delete that reports success while leaving the policy on disk and in use."""
    record = create(client)

    assert client.delete(f"/guardrails/{record['id']}").status_code == 204
    assert client.get(f"/guardrails/{record['id']}").status_code == 404
    assert client.get("/guardrails").json() == []


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize("name", ["", "   ", "\n\t"])
def test_blank_name_is_rejected(client: TestClient, name: str) -> None:
    """
    Prevents an unnameable row.

    Guardrails are attached by picking one from a list; two blank names are impossible to
    tell apart, so the wrong policy gets attached to a live session.
    """
    assert client.post("/guardrails", json={"name": name}).status_code == 422


def test_unknown_field_is_rejected(client: TestClient) -> None:
    """
    Prevents a misspelled field being accepted and ignored.

    `{"max_answer_char": 20}` stored silently produces "I set the cap and it ignored me",
    with nothing in any log to explain it.
    """
    response = client.post("/guardrails", json={"name": "Typo", "max_answer_char": 20})

    assert response.status_code == 422


@pytest.mark.parametrize("topics", [[""], ["  "], ["salary", ""]])
def test_blank_banned_topic_is_rejected(client: TestClient, topics: list[str]) -> None:
    """
    Prevents a policy that refuses every turn.

    An empty term compiles to a pattern matching any string, so one stray comma in the
    topics field would block the whole interview — a total outage that otherwise passes
    validation and looks like a working config.
    """
    assert (
        client.post("/guardrails", json={"name": "P", "banned_topics": topics}).status_code
        == 422
    )


def test_banned_topics_are_normalised_and_deduplicated(client: TestClient) -> None:
    """
    Prevents the same violation being reported twice under different spellings.

    Matching is case-insensitive, so "Salary" and "salary" are one rule; storing both makes
    the violations list — and any counter built on it — double-count.
    """
    record = create(client, banned_topics=["Salary", "salary", " visa  status ", "SALARY"])

    assert record["banned_topics"] == ["salary", "visa status"]


@pytest.mark.parametrize("value", [0, -1, MAX_ANSWER_CHARS_CEILING + 1])
def test_max_answer_chars_outside_bounds_is_rejected(client: TestClient, value: int) -> None:
    """
    Prevents a cap that is a mute button or a no-op.

    Zero or negative rejects every possible answer — the avatar goes silent on every turn
    while the config reads as a length limit. Above the ceiling, truncation can never fire,
    so the field looks set and enforces nothing.
    """
    response = client.post("/guardrails", json={"name": "P", "max_answer_chars": value})

    assert response.status_code == 422


def test_unknown_on_violation_is_rejected(client: TestClient) -> None:
    """
    Prevents an unhandled action being stored.

    The caller switches on this value; an unrecognised one falls through every branch, which
    means a detected violation is detected and then ignored.
    """
    response = client.post("/guardrails", json={"name": "P", "on_violation": "shutdown"})

    assert response.status_code == 422


@pytest.mark.parametrize("message", ["", "   "])
def test_blank_refusal_message_is_rejected(client: TestClient, message: str) -> None:
    """
    Prevents a refusal that is indistinguishable from a crash.

    Every `on_violation` mode speaks this string. An empty one means the candidate asks a
    question and hears nothing — a working guardrail that reads as a dead avatar.
    """
    response = client.post("/guardrails", json={"name": "P", "refusal_message": message})

    assert response.status_code == 422


def test_patch_cannot_leave_the_stored_policy_invalid(client: TestClient) -> None:
    """
    Prevents a field-by-field-valid patch writing a record `evaluate()` would misread.

    Validating a patch in isolation is not enough — the invariants belong to the merged
    record. The disk assertion is the point: a rejected patch must not have partially
    landed.
    """
    record = create(client, refusal_message="Let's move on.")

    response = client.patch(f"/guardrails/{record['id']}", json={"refusal_message": "   "})

    assert response.status_code == 422
    assert client.get(f"/guardrails/{record['id']}").json() == record


# -- the check endpoint -------------------------------------------------------


def check(client: TestClient, guardrail_id: str, text: str, direction: str) -> dict[str, Any]:
    response = client.post(
        f"/guardrails/{guardrail_id}/check", json={"text": text, "direction": direction}
    )
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def test_clean_text_passes_through_untouched(client: TestClient) -> None:
    """
    Prevents a checker that reports violations on ordinary answers.

    A guardrail that fires on innocuous text is one an operator turns off wholesale, which
    removes the protection entirely.
    """
    record = create(client, banned_topics=["salary"], pii_redaction=True)

    result = check(client, record["id"], "I led the migration to a new scheduler.", "output")

    assert result == {
        "allowed": True,
        "violations": [],
        "redacted_text": "I led the migration to a new scheduler.",
    }


def test_banned_topic_blocks_and_names_the_term(client: TestClient) -> None:
    """
    Prevents a block with no explanation.

    The violation code is what the console shows and telemetry counts; "not allowed" with no
    term is untriageable when a policy has a dozen topics.
    """
    record = create(client, banned_topics=["salary"])

    result = check(client, record["id"], "My salary expectation is high.", "output")

    assert result["allowed"] is False
    assert result["violations"] == ["banned_topic:salary"]


def test_banned_topic_matching_is_case_insensitive(client: TestClient) -> None:
    """Prevents a ban bypassed by capitalisation — LLM output capitalises sentence-initially."""
    record = create(client, banned_topics=["visa status"])

    result = check(client, record["id"], "Visa Status is not something I discuss.", "input")

    assert result["violations"] == ["banned_topic:visa status"]


@pytest.mark.parametrize("text", ["I said yes.", "We maintain the index.", "Sustainability."])
def test_banned_topic_does_not_match_inside_a_word(client: TestClient, text: str) -> None:
    """
    Prevents the substring false positive that discredits the whole checker.

    A ban on "ai" implemented with `in` fires on "said", "maintain", and "sustainability" —
    three refusals in an ordinary sentence, and an operator who never trusts it again.
    """
    record = create(client, banned_topics=["ai"])

    result = check(client, record["id"], text, "output")

    assert result["allowed"] is True
    assert result["violations"] == []


def test_email_is_redacted_without_blocking_the_turn(client: TestClient) -> None:
    """
    Prevents redaction being escalated into a refusal.

    A candidate who mentions their own email must not lose their answer to a policy meant to
    protect them; the repaired text ships and the violation is still reported.
    """
    record = create(client, pii_redaction=True)

    result = check(client, record["id"], "Mail me at ada.l+cv@example.co.uk please.", "output")

    assert result["allowed"] is True
    assert result["violations"] == ["pii:email"]
    assert result["redacted_text"] == f"Mail me at {EMAIL_PLACEHOLDER} please."


@pytest.mark.parametrize(
    "text",
    [
        "Call 5558675309 now.",
        "Call 555 867 5309 now.",
        "Call (555) 867-5309 now.",
        "Card 4111-1111-1111-1111 was declined.",
    ],
)
def test_long_digit_runs_are_redacted_in_every_spacing(client: TestClient, text: str) -> None:
    """
    Prevents a redactor that only catches unspaced digits.

    Humans say phone and card numbers in groups and STT writes them that way, so a bare
    `\\d{7,}` pattern misses the majority of real occurrences — the ones that matter.
    """
    record = create(client, pii_redaction=True)

    result = check(client, record["id"], text, "input")

    assert "pii:number" in result["violations"]
    assert NUMBER_PLACEHOLDER in result["redacted_text"]
    assert not any(char.isdigit() for char in result["redacted_text"])


@pytest.mark.parametrize(
    "text",
    ["In 2026 I shipped it.", "It cost 45000 dollars.", "Version 3.11.4 of the runtime."],
)
def test_short_digit_runs_survive_redaction(client: TestClient, text: str) -> None:
    """
    Prevents a redactor that mangles ordinary answers.

    Years, prices, and version numbers are shorter than any phone number. Blanking them
    turns every technical answer into placeholders, and the operator's fix is to disable PII
    redaction entirely — so over-redacting costs more privacy than it buys.
    """
    record = create(client, pii_redaction=True)

    result = check(client, record["id"], text, "output")

    assert result["redacted_text"] == text
    assert result["violations"] == []


def test_pii_redaction_off_leaves_text_alone(client: TestClient) -> None:
    """Prevents redaction running unconditionally, ignoring the flag the operator set."""
    record = create(client, pii_redaction=False)
    text = "Mail me at ada@example.com or call 555 867 5309."

    result = check(client, record["id"], text, "output")

    assert result == {"allowed": True, "violations": [], "redacted_text": text}


def test_over_length_output_is_blocked_and_truncated_on_a_word_boundary(
    client: TestClient,
) -> None:
    """
    Prevents both halves of the length rule failing quietly.

    An unbounded answer is a candidate waiting through a minute of synthesis; a mid-word cut
    is a nonsense syllable that reads as a broken renderer rather than as policy.
    """
    record = create(client, max_answer_chars=40)
    text = "This answer runs considerably longer than the configured ceiling allows."

    result = check(client, record["id"], text, "output")

    assert result["allowed"] is False
    assert result["violations"] == [f"max_answer_chars:{len(text)}>40"]
    assert len(result["redacted_text"]) <= 40
    assert text.startswith(result["redacted_text"])
    assert result["redacted_text"] == "This answer runs considerably longer"


def test_length_rule_does_not_apply_to_candidate_input(client: TestClient) -> None:
    """
    Prevents the cap being applied in the wrong direction.

    `max_answer_chars` bounds what the avatar says. Enforcing it on the input side would
    truncate the candidate's answer — the one artefact this product exists to collect — and
    the interviewer would respond to half a sentence.
    """
    record = create(client, max_answer_chars=20)

    result = check(
        client, record["id"], "A long answer from the candidate, well past twenty.", "input"
    )

    assert result["allowed"] is True
    assert result["violations"] == []
    assert result["redacted_text"].endswith("twenty.")


def test_redaction_cannot_hide_a_banned_topic(client: TestClient) -> None:
    """
    Prevents a placeholder swallowing a banned term and reporting the turn as clean.

    Topic matching runs on the original text for exactly this reason: if it ran on the
    redacted string, PII redaction would become a bypass for the topic ban.
    """
    record = create(client, banned_topics=["4111"], pii_redaction=True)

    result = check(client, record["id"], "The card 4111-1111-1111-1111 failed.", "output")

    assert result["allowed"] is False
    assert "banned_topic:4111" in result["violations"]
    assert "pii:number" in result["violations"]


def test_length_is_measured_on_the_redacted_text(client: TestClient) -> None:
    """
    Prevents the budget being computed against a string that never ships.

    TTS receives `redacted_text`, so that is the only length worth bounding. Measuring the
    original would block a turn whose redacted form fits comfortably.
    """
    record = create(client, max_answer_chars=30, pii_redaction=True)

    result = check(client, record["id"], "Reach me: someone@somewhere.example", "output")

    assert result["allowed"] is True
    assert result["violations"] == ["pii:email"]


@pytest.mark.parametrize("direction", ["inputs", "", "OUTPUT", "both"])
def test_unknown_direction_is_rejected(client: TestClient, direction: str) -> None:
    """
    Prevents a typo'd direction silently taking the wrong branch.

    The two directions have different rules; defaulting an unrecognised value to either one
    means the caller believes it checked something it did not.
    """
    record = create(client)

    response = client.post(
        f"/guardrails/{record['id']}/check", json={"text": "hi", "direction": direction}
    )

    assert response.status_code == 422


def test_evaluate_performs_no_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Prevents the check being "improved" into a model call.

    The output-side check runs between the LLM and TTS, inside a latency budget with nothing
    to spare, so a round trip here is self-defeating — and one that times out either stalls
    the turn or fails open on a guardrail that is trusted. Socket construction is made fatal
    for the duration so any future I/O in this path fails in CI instead of in production.
    """

    def no_sockets(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("evaluate() must not open a socket")

    monkeypatch.setattr(socket, "socket", no_sockets)
    monkeypatch.setattr(socket, "create_connection", no_sockets)

    policy = Policy(name="P", banned_topics=["salary"], pii_redaction=True)
    result = evaluate(policy, "My salary is ada@example.com 555 867 5309", "output")

    assert result.allowed is False
    assert result.violations == ["pii:email", "pii:number", "banned_topic:salary"]
