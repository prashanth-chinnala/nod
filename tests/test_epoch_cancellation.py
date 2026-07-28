"""
Barge-in.

The mechanism under test is a monotonic turn epoch, not task cancellation. What
that buys is checked here directly: an artifact produced under an abandoned turn is
dropped at the consumer, and the drop is observable in telemetry rather than
inferred from the video looking right. M4's acceptance criterion is exactly that
distinction, so it is worth having the unit tests state it first.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from avatar.contracts import Frame
from avatar.orchestrator import SessionOrchestrator
from avatar.renderers.stub import StubRenderer
from avatar.state import FrameSource, State
from tests.conftest import (
    RecordingTransport,
    ScriptedLLM,
    ScriptedTTS,
    reach_speaking,
    run_until,
    settle,
    tick,
)

# -- the transition --------------------------------------------------------


async def test_barge_in_during_speaking_lands_in_listening(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    gate = asyncio.Event()
    orch, _ = await reach_speaking(build_session, gate)
    speaking_epoch = orch.epoch

    await orch.on_speech_start()

    assert orch.state is State.LISTENING
    assert orch.epoch > speaking_epoch, "cancellation must consume an epoch"

    gate.set()
    await settle(orch)


async def test_barge_in_switches_the_frame_source_back_to_idle(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    gate = asyncio.Event()
    orch, _ = await reach_speaking(build_session, gate)
    assert orch._mixer.source is FrameSource.RENDERER

    await orch.on_speech_start()

    # Not "eventually" -- immediately. Leaving rendered frames on screen for the
    # duration of the cancellation is the visible symptom of a laggy barge-in,
    # even when the state machine reacted instantly.
    assert orch._mixer.source is FrameSource.IDLE_LOOP

    gate.set()
    await settle(orch)


async def test_barge_in_resets_the_renderer_and_flushes_the_client(
    build_session: Callable[..., SessionOrchestrator],
    renderer: StubRenderer,
    transport: RecordingTransport,
) -> None:
    gate = asyncio.Event()
    orch, _ = await reach_speaking(build_session, gate)
    assert transport.sent, "precondition: audio reached the transport"

    await orch.on_speech_start()

    assert renderer.sessions_closed == 0, "reset must not tear down the session"
    # A server-side flush alone leaves the client's jitter buffer playing a
    # sentence the avatar has already abandoned.
    assert transport.flushes == 1
    assert transport.sent == []

    gate.set()
    await settle(orch)


# -- stale artifacts -------------------------------------------------------


async def test_frames_from_the_cancelled_turn_are_dropped(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    gate = asyncio.Event()
    orch, _ = await reach_speaking(build_session, gate)
    stale_epoch = orch.epoch

    await orch.on_speech_start()

    accepted = orch._mixer.offer(Frame(data=b"stale", epoch=stale_epoch, pts_ms=0), orch.epoch)

    assert accepted is False
    assert orch._mixer.buffered() == 0
    dropped = [e for e in orch._telemetry.events if e["event"] == "stale_dropped"]
    assert any(e["kind"] == "frame" for e in dropped)

    gate.set()
    await settle(orch)


async def test_stale_audio_chunks_are_not_forwarded_to_transport(
    build_session: Callable[..., SessionOrchestrator], transport: RecordingTransport
) -> None:
    """
    The TTS is paused mid-turn, the turn is cancelled, then the TTS is released.

    Its next chunk still carries the old epoch -- the generator has no idea it was
    abandoned. That chunk must die at the orchestrator rather than reach the client.
    """
    gate = asyncio.Event()
    orch, _ = await reach_speaking(build_session, gate)

    await orch.on_speech_start()
    assert transport.sent == [], "the flush emptied what had already been sent"

    gate.set()
    await settle(orch)

    dropped = [
        e
        for e in orch._telemetry.events
        if e["event"] == "stale_dropped" and e["kind"] == "audio"
    ]
    assert dropped, "the chunk still in the TTS generator was dropped on arrival"
    assert transport.sent == [], "and nothing reached the client after the flush"


async def test_stale_audio_ack_does_not_credit_the_new_turn(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    """
    A late ack from the abandoned turn must not inflate the turn that replaced it.

    Acks are inherently late -- they describe playback that already happened. If
    one for turn N landed on turn N+1, the next truncation would credit the
    candidate with hearing words from a sentence that had not started yet.
    """
    gate = asyncio.Event()
    orch, _ = await reach_speaking(build_session, gate)
    stale_epoch = orch.epoch

    await orch.on_speech_start()
    gate.set()
    await settle(orch)

    await orch.on_end_of_turn("second question")
    new_turn = orch.turn
    assert new_turn is not None and new_turn.epoch != stale_epoch

    orch.on_audio_played(500, stale_epoch)

    assert new_turn.audio_played_ms == 0
    dropped = [
        e
        for e in orch._telemetry.events
        if e["event"] == "stale_dropped" and e["kind"] == "audio_ack"
    ]
    assert dropped

    await settle(orch)


# -- edges -----------------------------------------------------------------


async def test_barge_in_during_thinking_does_not_wedge(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    gate = asyncio.Event()
    tts = ScriptedTTS(chunks_per_sentence=12, gate=gate, gate_after_chunks=1)
    orch = build_session(llm=ScriptedLLM(["Not enough frames yet."]), tts=tts)
    await orch.start("reference.mp4")
    await orch.on_speech_start()
    await orch.on_end_of_turn("Tell me about a failure.")
    await run_until(lambda: tts.emitted == 1, what="first chunk")

    assert orch.state is State.THINKING, "lead-in not satisfied, so no frame shown yet"

    await orch.on_speech_start()

    assert orch.state is State.LISTENING

    gate.set()
    await settle(orch)


async def test_two_rapid_barge_ins_yield_one_clean_listening(
    build_session: Callable[..., SessionOrchestrator], transport: RecordingTransport
) -> None:
    gate = asyncio.Event()
    orch, _ = await reach_speaking(build_session, gate)
    before = orch.epoch

    await orch.on_speech_start()
    await orch.on_speech_start()
    await tick(3)

    assert orch.state is State.LISTENING
    assert orch.epoch == before + 1, "the second barge-in must not consume an epoch"
    assert transport.flushes == 1, "nor flush the client twice"

    gate.set()
    await settle(orch)


async def test_cancelled_turn_does_not_append_generated_text_to_history(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    gate = asyncio.Event()
    orch, _ = await reach_speaking(build_session, gate)

    await orch.on_speech_start()
    gate.set()
    await settle(orch)

    # No playback was ever acknowledged, so from the candidate's side the avatar
    # said nothing, and history must agree with them.
    assert [m["role"] for m in orch.history] == ["user"]


async def test_close_during_a_turn_invalidates_it(
    build_session: Callable[..., SessionOrchestrator], renderer: StubRenderer
) -> None:
    gate = asyncio.Event()
    orch, _ = await reach_speaking(build_session, gate)

    gate.set()
    await orch.close()

    assert orch.state is State.CLOSED
    assert renderer.sessions_closed == 1
