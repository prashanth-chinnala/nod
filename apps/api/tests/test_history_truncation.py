"""
Conversation history reflects what the candidate heard, not what was generated.

This is the failure mode that is invisible in a demo and corrupts every subsequent
turn: the LLM's context claims the interviewer asked a question the candidate never
received, so the next question follows up on something that was never said. The
symptom looks like a bad model, not like a bug in the session layer.

The distinction that carries the whole thing is `audio_played_ms` (acknowledged by
the client) versus `audio_sent_ms` (handed to the transport). The gap between them
is the client's jitter buffer, and a barge-in throws that buffer away.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from avatar.contracts import Turn
from avatar.orchestrator import (
    SessionOrchestrator,
    estimate_duration_ms,
    heard_text,
)
from tests.conftest import CHUNK_MS, ScriptedLLM, reach_speaking, run_until, settle

# Five words at 150wpm is 2000ms, which makes the ratios in these tests readable.
FIVE_WORDS = "one two three four five"


async def park_mid_sentence(
    build_session: Callable[..., SessionOrchestrator],
    gate: asyncio.Event,
    *,
    min_sent_ms: int,
) -> SessionOrchestrator:
    """
    Hold a turn open with at least `min_sent_ms` of audio handed to the transport.

    The threshold matters: these tests are about the gap between sent and played,
    so a scenario where the client acknowledges more audio than was ever sent
    would prove nothing. The gate parks the TTS once enough has gone out.
    """
    chunks = min_sent_ms // CHUNK_MS
    orch, _ = await reach_speaking(
        build_session,
        gate,
        sentences=[FIVE_WORDS],
        chunks_per_sentence=chunks + 10,
        gate_after_chunks=chunks,
    )
    await run_until(
        lambda: orch.turn is not None and orch.turn.audio_sent_ms >= min_sent_ms,
        what=f"{min_sent_ms}ms of audio sent",
    )
    return orch


def turn_with(**kwargs: object) -> Turn:
    base: dict[str, object] = {"epoch": 1, "text_generated": FIVE_WORDS}
    base.update(kwargs)
    return Turn(**base)  # type: ignore[arg-type]


# -- the pure policy -------------------------------------------------------


def test_duration_estimate_scales_with_word_count() -> None:
    assert estimate_duration_ms(FIVE_WORDS) == 2000
    assert estimate_duration_ms("one two") == 800
    assert estimate_duration_ms("   ") == 0
    assert estimate_duration_ms("") == 0


def test_uninterrupted_turn_keeps_the_full_generated_text() -> None:
    turn = turn_with(interrupted=False, audio_played_ms=0)

    # Playback acknowledgement is irrelevant when nothing was cut off: the turn
    # ran to completion, so every word was destined for the candidate.
    assert heard_text(turn) == FIVE_WORDS


def test_interrupted_turn_keeps_only_the_played_prefix() -> None:
    turn = turn_with(interrupted=True, audio_played_ms=800)

    assert heard_text(turn) == "one two"


def test_interruption_before_any_playback_keeps_nothing() -> None:
    turn = turn_with(interrupted=True, audio_played_ms=0)

    assert heard_text(turn) == ""


def test_exact_word_multiples_are_not_rounded_down() -> None:
    # 1200ms is exactly three words at 150wpm, and the character cut lands on the
    # space after "three". Nothing is mid-word, so nothing is dropped.
    assert heard_text(turn_with(interrupted=True, audio_played_ms=1200)) == "one two three"


def test_truncation_lands_on_a_word_boundary() -> None:
    # 1000ms of 2000ms is 50% of the characters, which falls inside "three".
    turn = turn_with(interrupted=True, audio_played_ms=1000)

    result = heard_text(turn)

    assert result == "one two"
    assert not result.endswith("thr"), "a mid-word cut reads as corruption in a transcript"


def test_playback_beyond_the_estimate_is_clamped() -> None:
    turn = turn_with(interrupted=True, audio_played_ms=99_999)

    assert heard_text(turn) == FIVE_WORDS


def test_sent_but_unplayed_audio_is_not_credited() -> None:
    turn = turn_with(interrupted=True, audio_sent_ms=2000, audio_played_ms=400)

    # All five words reached the transport. One was heard. The other four were in
    # the client's buffer when it flushed, so the candidate never received them.
    assert heard_text(turn) == "one"


def test_interrupted_turn_with_no_generated_text_keeps_nothing() -> None:
    turn = turn_with(text_generated="", interrupted=True, audio_played_ms=500)

    assert heard_text(turn) == ""


# -- through the orchestrator ----------------------------------------------


async def test_completed_turn_records_the_full_text(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    orch = build_session(llm=ScriptedLLM(["The billing migration.", " Second year."]))
    await orch.start("reference.mp4")
    await orch.on_speech_start()
    await orch.on_end_of_turn("Tell me about a failure.")
    await settle(orch)

    assert orch.history[-1] == {
        "role": "assistant",
        "content": "The billing migration. Second year.",
    }


async def test_interrupted_turn_records_the_heard_prefix_marked_interrupted(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    gate = asyncio.Event()
    orch = await park_mid_sentence(build_session, gate, min_sent_ms=880)

    orch.on_audio_played(800, orch.epoch)
    await orch.on_speech_start()
    gate.set()
    await settle(orch)

    assert orch.history[-1] == {
        "role": "assistant",
        "content": "one two [interrupted]",
    }


async def test_history_never_contains_text_the_client_did_not_receive(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    gate = asyncio.Event()
    orch = await park_mid_sentence(build_session, gate, min_sent_ms=640)

    orch.on_audio_played(400, orch.epoch)
    await orch.on_speech_start()
    gate.set()
    await settle(orch)

    turn = orch.turn
    assert turn is not None
    assert turn.audio_sent_ms > turn.audio_played_ms, "precondition: audio was buffered"

    recorded = orch.history[-1]["content"].removesuffix(" [interrupted]")
    assert FIVE_WORDS.startswith(recorded)
    assert len(recorded) < len(FIVE_WORDS)


async def test_interrupted_turn_with_no_acknowledged_playback_records_nothing(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    gate = asyncio.Event()
    orch, _ = await reach_speaking(build_session, gate, sentences=[FIVE_WORDS])

    # No acks at all -- the audio was sent but the client never confirmed playing
    # any of it. From the candidate's side the avatar never spoke.
    await orch.on_speech_start()
    gate.set()
    await settle(orch)

    assert [m["role"] for m in orch.history] == ["user"]


async def test_the_llm_only_ever_sees_truncated_history(
    build_session: Callable[..., SessionOrchestrator],
) -> None:
    """
    The point of all of the above: the next turn's prompt must not contain words
    the candidate never heard.
    """
    gate = asyncio.Event()
    llm = ScriptedLLM([FIVE_WORDS])
    orch = await park_mid_sentence(build_session, gate, min_sent_ms=880)

    orch.on_audio_played(800, orch.epoch)
    await orch.on_speech_start()
    gate.set()
    await settle(orch)

    orch._llm = llm
    await orch.on_end_of_turn("sorry, I meant something else")
    await settle(orch)

    assert llm.calls, "the second turn queried the LLM"
    prompt = llm.calls[0]
    assistant_turns = [m for m in prompt if m["role"] == "assistant"]
    assert assistant_turns == [{"role": "assistant", "content": "one two [interrupted]"}]
    assert not any("three" in m["content"] for m in prompt)


@pytest.mark.parametrize("played_ms", [0, 400, 800, 1200, 1600, 2000])
def test_truncation_is_always_a_prefix(played_ms: int) -> None:
    """Whatever the ratio, the result is a prefix of what was generated."""
    turn = turn_with(interrupted=True, audio_played_ms=played_ms)

    assert FIVE_WORDS.startswith(heard_text(turn))
