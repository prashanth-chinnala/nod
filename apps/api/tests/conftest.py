"""
Test doubles and fixtures.

Two principles here, both from the guide's test spec:

  1. No real sleeping. `FakeClock` is the time source and the sleep function, so a
     test that needs "one simulated second" gets it in microseconds. A slow suite
     gets skipped; a flaky suite is worse than no suite.

  2. No real renderer, no real network, no real model. Everything the orchestrator
     talks to is injected, so the whole state machine runs in CI on a machine with
     nothing installed but pytest.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable, Iterable, Sequence

import pytest

from avatar.contracts import AudioChunk, Message
from avatar.mixer import FrameMixer, IdleLoop
from avatar.orchestrator import SessionOrchestrator
from avatar.renderers.stub import StubRenderer
from avatar.state import State
from avatar.telemetry import NullSink, Telemetry

IDLE_FRAME_COUNT = 8
CHUNK_MS = 40


class FakeClock:
    """Monotonic time under test control. Also serves as the sleep function."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds

    async def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)
        # Still yield to the loop: the pacing is fake, the concurrency is not.
        await asyncio.sleep(0)


class RecordingTransport:
    """Records what reached the client, and when it was thrown away."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.sent: list[AudioChunk] = []
        self.flushes = 0

    async def open_track(self) -> None:
        self.opened = True

    async def send_audio(self, chunk: AudioChunk) -> None:
        self.sent.append(chunk)

    async def flush_audio(self) -> None:
        self.flushes += 1
        self.sent.clear()

    async def close_track(self) -> None:
        self.closed = True


class ScriptedLLM:
    """Yields a fixed list of sentences, optionally pausing on a gate."""

    def __init__(
        self,
        sentences: Sequence[str],
        *,
        gate: asyncio.Event | None = None,
        gate_before_index: int | None = None,
        raise_at_index: int | None = None,
    ) -> None:
        self.sentences = list(sentences)
        self.gate = gate
        self.gate_before_index = gate_before_index
        self.raise_at_index = raise_at_index
        self.calls: list[list[Message]] = []

    def __call__(self, history: Sequence[Message]) -> AsyncIterator[str]:
        self.calls.append(list(history))
        return self._generate()

    async def _generate(self) -> AsyncIterator[str]:
        for i, sentence in enumerate(self.sentences):
            if self.raise_at_index == i:
                raise RuntimeError("llm exploded")
            if self.gate is not None and self.gate_before_index == i:
                await self.gate.wait()
            yield sentence
            await asyncio.sleep(0)


class ScriptedTTS:
    """
    Splits each sentence into fixed-duration chunks.

    `gate` pauses the stream after `gate_after_chunks` chunks so a test can hold a
    turn open in a known state and interrupt it deterministically.
    """

    def __init__(
        self,
        *,
        chunks_per_sentence: int = 4,
        chunk_ms: int = CHUNK_MS,
        gate: asyncio.Event | None = None,
        gate_after_chunks: int | None = None,
    ) -> None:
        self.chunks_per_sentence = chunks_per_sentence
        self.chunk_ms = chunk_ms
        self.gate = gate
        self.gate_after_chunks = gate_after_chunks
        self.emitted = 0

    def __call__(self, text: str, epoch: int) -> AsyncIterator[AudioChunk]:
        return self._generate(epoch)

    async def _generate(self, epoch: int) -> AsyncIterator[AudioChunk]:
        for _ in range(self.chunks_per_sentence):
            yield AudioChunk(pcm=b"\x00" * 16, epoch=epoch, duration_ms=self.chunk_ms)
            self.emitted += 1
            if self.gate is not None and self.emitted == self.gate_after_chunks:
                await self.gate.wait()
            await asyncio.sleep(0)


def make_idle_loop(
    count: int = IDLE_FRAME_COUNT, mouth_closed: Iterable[int] | None = None
) -> IdleLoop:
    frames = [f"idle-{i}".encode() for i in range(count)]
    return IdleLoop(frames, range(count) if mouth_closed is None else mouth_closed)


def make_mixer(
    clock: FakeClock, telemetry: Telemetry, idle: IdleLoop | None = None
) -> FrameMixer:
    return FrameMixer(idle or make_idle_loop(), telemetry, clock=clock, sleep=clock.sleep)


async def tick(times: int = 1) -> None:
    """Let pending tasks make progress without advancing simulated time."""
    for _ in range(times):
        await asyncio.sleep(0)


async def settle(orch: SessionOrchestrator) -> None:
    """Wait for the in-flight turn to finish, ignoring cancellation."""
    task = orch.pipeline_task
    if task is None:
        return
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def run_until(
    predicate: Callable[[], bool], *, max_ticks: int = 500, what: str = "condition"
) -> None:
    """Cooperatively advance the loop until `predicate` holds."""
    for _ in range(max_ticks):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError(f"{what} never became true within {max_ticks} ticks")


async def reach_speaking(
    build_session: Callable[..., SessionOrchestrator],
    gate: asyncio.Event,
    *,
    sentences: Sequence[str] = ("A long answer that gets cut off.",),
    chunks_per_sentence: int = 12,
    gate_after_chunks: int = 5,
) -> tuple[SessionOrchestrator, ScriptedTTS]:
    """
    Drive a session to SPEAKING and hold it there.

    The TTS gate is what makes barge-in tests deterministic: the turn is parked at
    a known point with audio in flight, rather than raced against.
    """
    tts = ScriptedTTS(
        chunks_per_sentence=chunks_per_sentence,
        gate=gate,
        gate_after_chunks=gate_after_chunks,
    )
    orch = build_session(llm=ScriptedLLM(list(sentences)), tts=tts)
    await orch.start("reference.mp4")
    await orch.on_speech_start()
    await orch.on_end_of_turn("Tell me about a failure.")
    await run_until(lambda: orch.state is State.SPEAKING, what="SPEAKING")
    return orch, tts


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def telemetry() -> Telemetry:
    return Telemetry(sink=NullSink())


@pytest.fixture
def transport() -> RecordingTransport:
    return RecordingTransport()


@pytest.fixture
def renderer() -> StubRenderer:
    return StubRenderer(width=4, height=4, frame_interval_ms=CHUNK_MS)


@pytest.fixture
def build_session(
    clock: FakeClock,
    telemetry: Telemetry,
    transport: RecordingTransport,
    renderer: StubRenderer,
) -> Callable[..., SessionOrchestrator]:
    """
    Builds an orchestrator with every collaborator faked.

    Returned as a factory rather than an instance so individual tests can vary the
    LLM script, the idle loop's seam annotations, or the lead-in depth without a
    fixture per permutation.
    """

    def _build(
        *,
        llm: ScriptedLLM | None = None,
        tts: ScriptedTTS | None = None,
        idle: IdleLoop | None = None,
        render_lead_in_frames: int = 4,
        idle_reprompt_seconds: float = 12.0,
        seam_wait_max_ms: int = 120,
    ) -> SessionOrchestrator:
        return SessionOrchestrator(
            renderer=renderer,
            mixer=make_mixer(clock, telemetry, idle),
            transport=transport,
            llm=llm or ScriptedLLM(["Tell me about a system you designed."]),
            tts=tts or ScriptedTTS(),
            telemetry=telemetry,
            clock=clock,
            render_lead_in_frames=render_lead_in_frames,
            idle_reprompt_seconds=idle_reprompt_seconds,
            seam_wait_max_ms=seam_wait_max_ms,
        )

    return _build
