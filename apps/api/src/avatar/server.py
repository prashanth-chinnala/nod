"""
FastAPI app. One orchestrator per WebSocket session.

This is the only module that imports a web framework, and the only one that knows a
session is reached over HTTP. The orchestrator receives a `Transport`; it has no idea
whether that is a WebSocket, WebRTC, or a test double.

Three background tasks run for the lifetime of a session, and the split matters:

  frame pump      drains the mixer at a constant cadence, forever. Not driven by
                  turns -- the track carries frames in every state, including idle.
  telemetry relay forwards instrumentation events to the browser as they happen, so
                  a barge-in is visible as an epoch changing rather than as a guess
                  about what the video did.
  silence tick    drives `on_idle_tick`. The orchestrator owns no timer of its own,
                  which is what lets the whole machine be tested on a fake clock.

Concurrency is one session per socket with no pooling: a renderer is constructed and
warmed at connect time and torn down at disconnect. For a GPU renderer that is the
wrong shape -- cold-loading weights per session is exactly the cost §1.4 argues
cannot be paid at conversation start -- and it is deferred to M7 rather than
pretended away.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from avatar.agent_config import ResolvedAgent, resolve_agent
from avatar.api import (
    agents,
    faces,
    guardrails,
    knowledge,
    pronunciations,
    sessions,
    tools,
)
from avatar.audio.stt import build_stt
from avatar.audio.tts import SAMPLE_RATE
from avatar.audio.tts_deepgram import build_tts
from avatar.audio.turn_detection import (
    END_OF_TURN_SILENCE_MS,
    EventKind,
    TurnDetector,
    TurnEvent,
)
from avatar.audio.vad import FRAME_MS, build_vad
from avatar.config import load_env, loaded_files
from avatar.contracts import RendererConfig
from avatar.idle import placeholder_idle_loop
from avatar.knowledge.augment import with_knowledge, with_pronunciation
from avatar.llm_anthropic import build_llm
from avatar.mixer import FRAME_INTERVAL_MS, TARGET_FPS, FrameMixer
from avatar.orchestrator import RENDER_LEAD_IN_FRAMES, SessionOrchestrator
from avatar.renderers import build
from avatar.state import State
from avatar.telemetry import STAGE_TURN_DETECT, Telemetry
from avatar.transport.websocket import WebSocketTransport

# Before any os.environ.get below. Without this, .env was inert: every run needed
# `set -a && . ./.env && set +a` in front of it, and forgetting produced a session that
# silently fell back to every placeholder -- no error, just quietly the wrong system.
_FROM_ENV_FILE = load_env()

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

FRAME_WIDTH = 256
FRAME_HEIGHT = 144
"""
Small on purpose.

Uncompressed BMP at 25fps costs width * height * 3 * 25 bytes/sec -- about 2.7MB/s
at this size. That is fine on localhost and indefensible over a network, and it is
a consequence of having no encoder rather than a considered choice. The real
renderer emits JPEG or WebP in M2 and this constraint disappears; until then, small
frames keep the demo honest about where the bytes go. Recorded in PROCESS.md 3.4.
"""

RENDERER_FIRST_FRAME_DELAY_MS = 200
"""
Audio the stub renderer requires before emitting a frame.

Not arbitrary: real talking-head models need a lookahead window, and setting this to
zero would make the first-frame latency readout meaningless and let the lead-in
buffer look unnecessary.
"""

IDENTITY_REFERENCE = os.environ.get("AVATAR_REFERENCE", "assets/reference.mp4")
RENDERER_NAME = os.environ.get("AVATAR_RENDERER", "stub")
"""
The one-line renderer swap, as an environment variable.

`AVATAR_RENDERER=musetalk uvicorn avatar.server:app` is the whole change once M2
lands. Nothing else in this file mentions a model.
"""

STT_NAME = os.environ.get("AVATAR_STT", "none")
"""
Which transcriber to run. `none` needs nothing; `deepgram` needs a key.

The transcriber never decides when a turn ends -- `audio.turn_detection` does. See the
`Transcriber` docstring in `contracts` for why that split is deliberate.
"""

TTS_NAME = os.environ.get("AVATAR_TTS", "tone")
"""
Which synthesiser to run. `tone` needs nothing; `deepgram` needs a key.

Defaults to `tone` for the same reason as the LLM and the VAD: a clean clone has to run
with no credentials and no network.
"""

LLM_NAME = os.environ.get("AVATAR_LLM", "scripted")
"""
Which interviewer to run. `scripted` needs nothing; `anthropic` needs a key.

Defaults to `scripted` so a clean clone runs with no credentials and no network, which
is what the README promises. `AVATAR_LLM=anthropic` is the whole switch.
"""

VAD_NAME = os.environ.get("AVATAR_VAD", "energy")
"""
Which speech detector to run. `energy` needs nothing; `silero` needs torch.

Turn detection happens server-side rather than in the browser. The trade-off, stated
because it is a real one: the client streams microphone audio continuously, which costs
bandwidth and means candidate audio reaches the server even between turns. In exchange,
the turn-taking policy is one implementation with one set of thresholds that can be
tested and tuned centrally, rather than whatever each browser happened to ship. For an
interview product the second consideration wins; for a consumer toy it might not.
"""

SILENCE_TICK_SECONDS = 1.0
STATS_INTERVAL_SECONDS = 0.5
RELAY_QUEUE_DEPTH = 256

RELAYED_EVENTS = frozenset(
    {"state_change", "latency", "stale_dropped", "session_failure", "counter", "heard", "said"}
)
"""
Which telemetry events reach the browser.

`frame_repeated` is excluded despite being one of the most interesting signals: it
fires up to 25 times a second, and relaying it would spend the socket on
instrumentation instead of video. The count still reaches the page in the stats
message.

`heard` and `said` are the conversation itself, so they belong here rather than only in a
server log. Both are once-per-turn-ish -- `said` is once per sentence -- so the volume
argument that excludes `frame_repeated` does not apply. **This allowlist is easy to forget:
a new event is silently invisible to the client until it is added here, which is exactly
what happened to both of these.**
"""

app = FastAPI(title="nod", docs_url=None, redoc_url=None)

# The console runs on a different origin in development (Next.js on :3000, this on :8000),
# so the browser needs permission to call across. Named origins rather than `*`: with
# credentials disallowed a wildcard is not a data-exfiltration hole, but it does mean any
# page on the internet can drive this API while a developer has it running -- including
# creating agents and deleting sessions. Two literal origins cost nothing and remove that.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["content-type"],
)

# Console CRUD, distinct from the session WebSocket below. These routers manage
# configuration; the socket *is* the runtime. They share this process only because a second
# deployment unit would be overhead at this size -- nothing in the orchestration path imports
# them, so splitting them out later is a routing change rather than a rewrite.
for _resource in (agents, faces, guardrails, knowledge, pronunciations, sessions, tools):
    app.include_router(_resource.router)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/config")
async def config() -> dict[str, object]:
    """
    Which implementation each boundary resolved to, and which came from `.env`.

    Names only, never values -- most of what `.env` holds is a credential. Exists because
    "why is it still using the placeholder voice?" is otherwise answered by reading code.
    """
    return {
        "renderer": RENDERER_NAME,
        "llm": LLM_NAME,
        "llm_model": os.environ.get("AVATAR_LLM_MODEL", "(adapter default)"),
        "llm_base_url": os.environ.get("OPENAI_BASE_URL", "(vendor default)"),
        "tts": TTS_NAME,
        "tts_voice": os.environ.get("AVATAR_TTS_VOICE", "(adapter default)"),
        "stt": STT_NAME,
        "vad": VAD_NAME,
        "env_files_read": loaded_files(),
        "loaded_from_env_file": sorted(_FROM_ENV_FILE),
    }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {
        "status": "ok",
        "renderer": RENDERER_NAME,
        "llm": LLM_NAME,
        "tts": TTS_NAME,
        "stt": STT_NAME,
        "vad": VAD_NAME,
    }


class BrowserSession:
    """One conversational session bound to one WebSocket."""

    def __init__(self, socket: WebSocket) -> None:
        self._socket = socket
        self._relay: asyncio.Queue[Mapping[str, object]] = asyncio.Queue(
            maxsize=RELAY_QUEUE_DEPTH
        )

        # Resolved once per session, not per turn: the retriever indexes its whole corpus
        # here, so a turn pays a scored lookup rather than re-reading documents from disk
        # inside the latency budget.
        self._agent: ResolvedAgent = resolve_agent()

        self._telemetry = Telemetry()
        self._telemetry.subscribe(self._on_telemetry)

        self._vad = build_vad(VAD_NAME)
        self._stt = build_stt(STT_NAME)
        self._detector = TurnDetector(frame_ms=FRAME_MS)
        self._mic = bytearray()
        self._speech_probability = 0.0

        self._transport = WebSocketTransport(socket.send_bytes, socket.send_text)
        self._mixer = FrameMixer(
            placeholder_idle_loop(width=FRAME_WIDTH, height=FRAME_HEIGHT),
            self._telemetry,
        )
        self._orchestrator = SessionOrchestrator(
            renderer=build(
                RendererConfig(
                    name=RENDERER_NAME,
                    options={
                        "width": FRAME_WIDTH,
                        "height": FRAME_HEIGHT,
                        "first_frame_delay_ms": RENDERER_FIRST_FRAME_DELAY_MS,
                        "frame_interval_ms": FRAME_INTERVAL_MS,
                    },
                )
            ),
            mixer=self._mixer,
            transport=self._transport,
            # Both boundaries are wrapped rather than the orchestrator being changed:
            # retrieval augments the prompt, a lexicon rewrites text before synthesis, and
            # neither is a session-lifecycle concern. The state machine cannot tell.
            llm=with_knowledge(build_llm(LLM_NAME), self._agent.retriever),
            tts=with_pronunciation(build_tts(TTS_NAME), self._agent.lexicon),
            telemetry=self._telemetry,
        )

    # -- lifecycle ----------------------------------------------------------

    async def run(self) -> None:
        await self._socket.accept()
        await self._send(
            {
                "type": "hello",
                "sample_rate": SAMPLE_RATE,
                "target_fps": TARGET_FPS,
                "frame_interval_ms": FRAME_INTERVAL_MS,
                "render_lead_in_frames": RENDER_LEAD_IN_FRAMES,
                "renderer": RENDERER_NAME,
                "frame_width": FRAME_WIDTH,
                "frame_height": FRAME_HEIGHT,
                "llm": LLM_NAME,
                "tts": TTS_NAME,
                "stt": STT_NAME,
                "vad": VAD_NAME,
                "vad_frame_ms": FRAME_MS,
                "end_of_turn_silence_ms": END_OF_TURN_SILENCE_MS,
            }
        )
        await self._orchestrator.start(IDENTITY_REFERENCE)

        tasks = [
            # Warmed in the background, deliberately not awaited. The measured ~910ms
            # connect has to be paid before the first turn -- but awaiting it here would
            # delay the video track by the same amount, and the candidate would sit in
            # front of a blank panel at session start. Nothing is lost by overlapping:
            # the microphone is not streaming yet.
            asyncio.create_task(self._warm_transcriber(), name="stt-warm"),
            asyncio.create_task(self._pump_frames(), name="frames"),
            asyncio.create_task(self._pump_relay(), name="relay"),
            asyncio.create_task(self._pump_stats(), name="stats"),
            asyncio.create_task(self._tick_silence(), name="silence"),
        ]
        try:
            await self._receive_loop()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # The socket is very likely already gone by the time we get here, so
            # every teardown step has to tolerate a closed transport.
            with contextlib.suppress(Exception):
                await self._orchestrator.close()
            with contextlib.suppress(Exception):
                await self._stt.aclose()

    async def _receive_loop(self) -> None:
        while True:
            message = await self._socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            audio = message.get("bytes")
            if audio:
                await self._on_mic_audio(audio)
                continue
            text = message.get("text")
            if text is None:
                continue
            try:
                await self._handle(json.loads(text))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                await self._send({"type": "error", "detail": f"{type(exc).__name__}: {exc}"})

    async def _handle(self, message: Mapping[str, Any]) -> None:
        kind = message.get("type")
        if kind == "speech_start":
            await self._orchestrator.on_speech_start()
        elif kind == "speech_retract":
            await self._orchestrator.on_speech_retract()
        elif kind == "end_of_turn":
            # A client-supplied transcript: the typed-answer path, and the one the buttons
            # use. It gets the same `heard` telemetry as the microphone path -- otherwise
            # the transcript pane would show half the conversation, and which half depended
            # on how the turn was started.
            spoken = str(message.get("transcript", ""))
            self._telemetry.heard(
                spoken, epoch=self._orchestrator.epoch, transcribed=bool(spoken)
            )
            await self._orchestrator.on_end_of_turn(spoken)
        elif kind == "audio_played":
            # The client's report of what it actually played. The only input that
            # moves audio_played_ms, and therefore the only evidence history
            # truncation runs on.
            self._orchestrator.on_audio_played(int(message["ms"]), int(message["epoch"]))
        elif kind == "first_paint":
            # Closes the end-to-end measurement. Everything from here back to turn
            # start is visible to the server; the encode-socket-decode-paint tail is
            # not, and this is the only report of it.
            self._orchestrator.on_first_paint(int(message["epoch"]))
        elif kind == "end_session":
            await self._orchestrator.close()
        else:
            await self._send({"type": "error", "detail": f"unknown message {kind!r}"})

    async def _warm_transcriber(self) -> None:
        """Open the STT socket without blocking the video track."""
        connect = getattr(self._stt, "connect", None)
        if connect is not None:
            with contextlib.suppress(Exception):
                await connect()

    # -- microphone ---------------------------------------------------------

    async def _on_mic_audio(self, pcm: bytes) -> None:
        """
        Feed the candidate's microphone through the VAD and the turn policy.

        Frames are fixed-size because Silero requires exactly 512 samples; the buffer
        exists to absorb whatever chunk size the browser happens to deliver. Leftover
        bytes stay buffered rather than being padded out, since a short frame scored as
        a full one reads as a VAD that misses quiet speech.

        Note what is *not* here: no gate on the session state. The microphone is
        processed while the avatar is speaking, because that is precisely when barge-in
        has to work. Keeping the avatar's own voice out of this path is the browser's
        echo cancellation, not ours — and if that fails, the avatar interrupts itself in
        a loop. A different VAD would not fix it.
        """
        # The transcriber gets the raw stream, unframed: it has its own opinion about
        # buffering and does not need the VAD's fixed window. It must never block this
        # path, which is why `push_audio` swallows its own failures.
        await self._stt.push_audio(pcm)

        self._mic.extend(pcm)
        frame_bytes = self._vad.frame_samples * 2
        while len(self._mic) >= frame_bytes:
            frame = bytes(self._mic[:frame_bytes])
            del self._mic[:frame_bytes]
            self._speech_probability = self._vad(frame)
            for event in self._detector.push(self._speech_probability):
                await self._dispatch_turn_event(event)

    async def _dispatch_turn_event(self, event: TurnEvent) -> None:
        if event.kind is EventKind.SPEECH_START:
            await self._orchestrator.on_speech_start()
        elif event.kind is EventKind.SPEECH_RETRACT:
            await self._orchestrator.on_speech_retract()
        elif event.kind is EventKind.END_OF_TURN:
            # End-of-turn detection latency is the silence threshold, by construction.
            # Recorded as a measurement because it occupies a real row in the latency
            # budget -- but it is configuration, not something hardware can improve,
            # and PROCESS.md 1.5 says so.
            self._telemetry.observe_ms(
                STAGE_TURN_DETECT,
                float(self._detector.end_of_turn_silence_ms),
                epoch=self._orchestrator.epoch,
            )
            # Whatever the transcriber has finalised by now. The policy has already
            # decided the turn is over, so this does not wait for a word still in
            # flight -- that gap is the documented cost of keeping turn detection local
            # rather than delegating it to the STT vendor's endpointing.
            transcript = self._stt.take_transcript()
            heard = transcript or f"[{event.speech_ms}ms of speech, no transcript]"
            # Emitted before the turn starts, so the log shows what the LLM was given and
            # not merely what it replied. Without this an empty transcript is invisible:
            # the interviewer still asks a plausible question, it just has nothing to do
            # with the answer, and that looks like a model problem rather than an STT one.
            self._telemetry.heard(
                heard, epoch=self._orchestrator.epoch, transcribed=bool(transcript)
            )
            await self._orchestrator.on_end_of_turn(heard)

    # -- background pumps ---------------------------------------------------

    async def _pump_frames(self) -> None:
        """
        Drain the mixer forever.

        Deliberately not conditional on state: the track carries frames while idle,
        while listening, and while thinking. Gating this on SPEAKING is the bug that
        produces a track which stalls between turns.
        """
        async for frame in self._mixer.stream():
            if self._orchestrator.state is State.CLOSED:
                return
            await self._transport.send_frame(frame)

    async def _pump_relay(self) -> None:
        while True:
            record = await self._relay.get()
            await self._send({"type": "event", **record})

    async def _pump_stats(self) -> None:
        while True:
            await asyncio.sleep(STATS_INTERVAL_SECONDS)
            await self._send(
                {
                    "type": "stats",
                    **self._orchestrator.stats(),
                    "bytes_sent": self._transport.bytes_sent,
                    "latency": self._telemetry.snapshot()["latency"],
                    # The server's own view of the microphone, so the page shows what
                    # the turn policy is acting on rather than what the browser's
                    # separate meter thinks.
                    "speech_probability": round(self._speech_probability, 3),
                    "in_speech": self._detector.in_speech,
                    "speech_ms": self._detector.speech_ms,
                }
            )

    async def _tick_silence(self) -> None:
        while True:
            await asyncio.sleep(SILENCE_TICK_SECONDS)
            await self._orchestrator.on_idle_tick()

    # -- plumbing -----------------------------------------------------------

    def _on_telemetry(self, record: Mapping[str, object]) -> None:
        """
        Called synchronously from inside instrumentation, so it must not block.

        Dropping on a full queue is the right failure: a client too slow to keep up
        with the event stream should lose readouts, not stall the render loop that
        is emitting them.
        """
        if record.get("event") not in RELAYED_EVENTS:
            return
        with contextlib.suppress(asyncio.QueueFull):
            self._relay.put_nowait(record)

    async def _send(self, payload: Mapping[str, object]) -> None:
        with contextlib.suppress(Exception):
            await self._socket.send_text(json.dumps(payload, default=str))


@app.websocket("/session")
async def session_socket(socket: WebSocket) -> None:
    await BrowserSession(socket).run()
