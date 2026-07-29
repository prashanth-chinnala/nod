"""
Tests for egress configuration.

Bounded honestly: there is no egress service in this environment, so nothing here proves a file
gets written. What these do cover is every decision that can be checked without one -- the
request shape, the opt-in, and the three outcomes the session record has to be able to tell
apart. The gap is stated in `avatar.transport.recording` rather than papered over with a mock
that would "pass" while proving nothing about a real recording.
"""

from __future__ import annotations

import pytest

from avatar.transport import recording


@pytest.fixture(autouse=True)
def _no_ambient_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Clear the recording and SFU variables for every test.

    A developer with `AVATAR_RECORD=1` in their environment would otherwise flip the opt-in
    tests into the opposite assertion, and the suite would pass or fail depending on whose shell
    ran it.
    """
    for name in ("AVATAR_RECORD", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_URL"):
        monkeypatch.delenv(name, raising=False)


# -- the opt-in ------------------------------------------------------------


def test_recording_is_off_unless_asked_for() -> None:
    """
    Recording must not switch itself on because an SFU happens to be configured.

    It is a decision with consent and retention consequences, and the candidate's pre-join
    screen promises a recording -- that promise should follow the setting rather than the
    reverse.
    """
    assert recording.recording_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " on "])
def test_truthy_spellings_all_enable(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """
    Several spellings, because this is set by hand in a `.env` file.

    A variable that reads `AVATAR_RECORD=true` and silently means false is the kind of thing
    discovered when a reviewer asks for a video that does not exist.
    """
    monkeypatch.setenv("AVATAR_RECORD", value)
    assert recording.recording_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_falsy_spellings_stay_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AVATAR_RECORD", value)
    assert recording.recording_enabled() is False


# -- the request -----------------------------------------------------------


def test_egress_asks_for_a_room_composite_not_a_single_track() -> None:
    """
    A reviewer wants the interview, not four files to line up by hand.

    Composite also captures both sides: a recording with only the avatar in it is not a
    reviewable artefact of a conversation.
    """
    pytest.importorskip("livekit.api")
    config = recording.egress_config("session-abc")
    assert config.room.room_name == "session-abc"
    assert len(config.room.file_outputs) == 1
    assert config.room.audio_only is False


def test_filepath_carries_the_room_and_a_timestamp() -> None:
    """
    The room derives from the session id, so a file can be traced back with no lookup table.

    `{time}` is expanded by the egress service and is there so re-running a session cannot
    silently overwrite the first recording -- two attempts at one interview are two artefacts.
    """
    pytest.importorskip("livekit.api")
    output = recording.egress_config("session-abc").room.file_outputs[0]
    assert "session-abc" in output.filepath
    assert "{time}" in output.filepath
    assert output.filepath.endswith(".mp4")


def test_output_directory_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local files keep the credential-free path credential-free; S3 is a field on the same
    output object, not a change in this module."""
    pytest.importorskip("livekit.api")
    monkeypatch.setattr(recording, "RECORDINGS_DIR", "/mnt/interviews")
    output = recording.egress_config("session-abc").room.file_outputs[0]
    assert output.filepath.startswith("/mnt/interviews/")


# -- the three outcomes ----------------------------------------------------


@pytest.mark.asyncio
async def test_not_requested_is_reported_as_off_with_a_reason() -> None:
    """
    `off` has to be stored, not omitted.

    A record that says nothing about recording is indistinguishable from one where recording
    silently failed, and that difference only matters at the moment someone asks for the video
    -- far too late to act on.
    """
    result = await recording.ensure_room("session-abc")
    assert result["status"] == "off"
    assert "AVATAR_RECORD" in result["reason"]


@pytest.mark.asyncio
async def test_missing_credentials_are_unavailable_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A candidate should be interviewed even when the recording cannot be set up.

    So this degrades and stores why, rather than raising into session start. The reason names
    the variables, because that is the actionable part.
    """
    monkeypatch.setenv("AVATAR_RECORD", "1")
    result = await recording.ensure_room("session-abc")
    assert result["status"] == "unavailable"
    assert "LIVEKIT_API_KEY" in result["reason"]


@pytest.mark.asyncio
async def test_a_failing_sfu_is_unavailable_with_the_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The expected failure when no egress service is registered must not end the interview."""
    pytest.importorskip("livekit.api")
    monkeypatch.setenv("AVATAR_RECORD", "1")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")
    monkeypatch.setenv("LIVEKIT_URL", "ws://127.0.0.1:1")  # nothing listens here

    result = await recording.ensure_room("session-abc")
    assert result["status"] == "unavailable"
    assert result["reason"]  # names the exception type and message


@pytest.mark.asyncio
async def test_success_is_called_requested_and_never_recording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The naming is the assertion.

    Nothing in this module observes a file being written, so a successful room creation may only
    claim that a recording was *requested*. Calling it `recording` would be exactly the kind of
    unverified assertion the project's first rule prohibits -- and the SFU accepts the config
    whether or not an egress worker exists to act on it, so the optimistic reading would be
    wrong silently.
    """
    pytest.importorskip("livekit.api")
    monkeypatch.setenv("AVATAR_RECORD", "1")
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret")

    class FakeRoom:
        async def create_room(self, request: object) -> object:
            self.request = request
            return type("Created", (), {"sid": "RM_test"})()

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.room = FakeRoom()

        async def aclose(self) -> None:
            return None

    from livekit import api

    monkeypatch.setattr(api, "LiveKitAPI", FakeClient)

    result = await recording.ensure_room("session-abc")
    assert result["status"] == "requested"
    assert result["room_sid"] == "RM_test"
    assert "cannot confirm" in result["reason"]
