"""
Worker delivery: the runtime streams speech to a renderer in another process.

These tests cover the parts that are GPU-free and that break silently. Every one of them exists
because the failure it catches presents as "the interview works but the face never appears" -- a
symptom with no stack trace, which is the expensive kind.

The live proof that the two processes actually exchange media is `scripts/avatar_worker.py`
against a real SFU; that cannot run here and is not simulated, because a mock of an SFU would
only prove the mock agrees with itself.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from avatar.contracts import AudioChunk, Frame, FrameCodec
from avatar.transport.worker_audio import DEFAULT_WORKER_IDENTITY, WorkerAudioTransport
from tests.conftest import ScriptedLLM, settle

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def chunk(ms: int = 20) -> AudioChunk:
    return AudioChunk(pcm=b"\x00\x00" * (16 * ms), epoch=1, duration_ms=float(ms))


def test_the_runtime_does_not_claim_the_identity_the_browser_subscribes_to() -> None:
    """
    The collision that would evict a participant mid-interview.

    `sessions.py` tells the browser to subscribe to `avatar-agent`, so the worker has to be that
    participant. If the runtime's own transport also joined under it, the SFU would resolve the
    duplicate by dropping one of them -- and which one is not deterministic. Asserted as a
    difference rather than as two literals so that renaming the constant cannot quietly make
    both sides agree on the same string again.
    """
    transport = WorkerAudioTransport("session-abc")
    assert transport.worker_identity == DEFAULT_WORKER_IDENTITY
    assert transport.identity != transport.worker_identity


def test_the_browser_is_told_to_subscribe_to_whoever_the_worker_claims() -> None:
    """
    The other half of the same fact, from the session layer's side.

    The value the join response hands the client and the value the runtime addresses its audio
    to have to be the same, and they are produced by two modules that do not otherwise know
    about each other. A worker publishing as one identity while the client waits for another is
    a black video element with no error anywhere -- so this asserts the session layer reads the
    shared constant rather than carrying its own copy of the string.
    """
    from avatar.api import sessions

    source = Path(sessions.__file__).read_text()
    assert '"agent_identity": AGENT_IDENTITY' in source
    assert '"agent_identity": "avatar-agent"' not in source
    assert DEFAULT_WORKER_IDENTITY == "avatar-agent"


def test_frames_from_a_local_renderer_are_dropped_and_counted() -> None:
    """
    Counted, because a non-zero count means a GPU is rendering frames nobody will ever see.

    Dropping them is correct -- the worker renders the published video -- but doing it silently
    would hide a misconfigured deployment paying twice for one face, which shows up on a bill
    rather than in a log.
    """
    transport = WorkerAudioTransport("session-abc")
    frame = Frame(
        data=b"\x00" * 12, epoch=1, pts_ms=0, codec=FrameCodec.RGB24, width=2, height=2
    )
    asyncio.run(transport.send_frame(frame))
    assert transport.frames_dropped == 1


def test_audio_before_the_stream_is_open_is_dropped_rather_than_raising() -> None:
    """
    A session that starts speaking before the worker's stream exists must not die.

    The alternative is an exception on the audio path during the first turn, which takes the
    whole session with it. Losing the opening chunks degrades that to a slightly late first
    word.
    """
    transport = WorkerAudioTransport("session-abc")
    asyncio.run(transport.send_audio(chunk()))
    assert transport.audio_sent == 0


def test_end_of_turn_and_flush_are_safe_with_no_stream() -> None:
    """Same reasoning as the audio path: teardown and barge-in race the stream existing."""
    transport = WorkerAudioTransport("session-abc")
    transport.end_of_turn()
    asyncio.run(transport.flush_audio())


def test_availability_explains_itself_instead_of_returning_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A reason, not a bool. The failure is a candidate staring at a room where nothing publishes.

    There is no way to detect that from the browser except by waiting, so the runtime has to be
    able to say why before the session starts.
    """
    monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
    monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
    reason = WorkerAudioTransport.available()
    assert reason
    assert "LIVEKIT_API_KEY" in reason or "livekit-agents" in reason


def test_worker_mode_never_builds_the_transport_that_would_collide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The wiring, asserted where it is decided rather than trusted from a comment.

    `server.py` picks one of the two on an `if/elif`, and an edit that turned it into two
    independent `if`s would build both -- producing exactly the identity collision the first
    test in this file describes, in the one configuration nobody runs locally.
    """
    source = (Path(__file__).resolve().parent.parent / "src/avatar/server.py").read_text()
    worker_branch = source.index('if DELIVERY == "worker" and session_id:')
    rtc_branch = source.index("elif session_id and _livekit_available():", worker_branch)
    assert rtc_branch > worker_branch


async def test_the_orchestrator_tells_the_transport_when_a_turn_ends(
    build_session: Callable[..., Any],
    transport: Any,
) -> None:
    """
    The signal the split renderer derives every turn boundary from.

    In-process this is decoration. Across a boundary it is the only thing that ends a segment,
    so a regression here is invisible locally and turns a remote worker's whole interview into
    one unbroken utterance -- mouth never at rest, and nothing for cancellation to count.
    """
    orch = build_session()
    await orch.start("reference.mp4")

    await orch.on_speech_start()
    await orch.on_end_of_turn("Tell me about a failure.")
    await settle(orch)

    assert transport.turn_ends == 1


async def test_a_cancelled_turn_is_not_reported_as_ended(
    build_session: Callable[..., Any],
    transport: Any,
) -> None:
    """
    Barge-in discards a segment; it does not complete one, and the worker treats those
    differently.

    If cancellation also fired `end_of_turn`, the abandoned utterance would be closed as though
    it had finished -- the renderer would return to rest and then be handed the replacement
    turn's audio with no way to tell it apart from a continuation of what it just finished.
    `flush_audio` is the interruption, and it is the one that has to arrive.
    """
    gate = asyncio.Event()
    orch = build_session(llm=ScriptedLLM(["A long answer.", " Still going."], gate=gate))
    await orch.start("reference.mp4")

    await orch.on_speech_start()
    await orch.on_end_of_turn("Tell me about a failure.")
    await asyncio.sleep(0)
    await orch.on_speech_start()  # the candidate speaks over it
    await settle(orch)

    assert transport.flushes >= 1
    assert transport.turn_ends == 0


@pytest.mark.parametrize("missing", ["agent", "face", "reference"])
def test_a_session_that_cannot_be_resolved_says_which_link_is_broken(missing: str) -> None:
    """
    Three ways the chain session -> agent -> face -> reference breaks, each named at its own
    link.

    The generic version of this failed with "no face to render" no matter which link was
    missing, which sent the reader to the wrong screen. The reference case is the one worth the
    most care: a path the runtime can write and the worker cannot read is a working interview
    with an absent face, and nothing in either process logs an error.
    """
    sys.path.insert(0, str(SCRIPTS))
    import avatar_worker

    responses: dict[str, dict[str, Any]] = {
        "/sessions/s": {"id": "s", "agent_id": "" if missing == "agent" else "a"},
        "/agents/a": {"id": "a", "face_id": "" if missing == "face" else "f"},
        "/faces/f": {"id": "f", "reference_path": "/nonexistent/ref.mp4"},
    }

    class FakeResponse:
        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            import json

            return json.dumps(self._payload).encode()

    import urllib.request

    def fake_urlopen(url: str, timeout: int = 0) -> FakeResponse:
        path = url.replace("http://api", "")
        return FakeResponse(responses[path])

    original = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen  # type: ignore[assignment]
    try:
        with pytest.raises(SystemExit) as caught:
            avatar_worker.resolve_session("http://api", "s")
    finally:
        urllib.request.urlopen = original  # type: ignore[assignment]

    message = str(caught.value)
    expected = {"agent": "no agent", "face": "no face", "reference": "cannot read"}[missing]
    assert expected in message
