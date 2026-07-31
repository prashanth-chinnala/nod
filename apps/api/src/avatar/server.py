"""
FastAPI app. One orchestrator per WebSocket session.

This is the only module that imports a web framework, and the only one that knows a session is
reached over HTTP. The orchestrator receives a `Transport`; it has no idea whether that is a
WebSocket, WebRTC, or a test double.

Three background tasks run for the lifetime of a session, and the split matters:

  frame pump      drains the mixer at a constant cadence, forever. Not driven by
                  turns -- the track carries frames in every state, including idle.
  telemetry relay forwards instrumentation events to the browser as they happen, so
                  a barge-in is visible as an epoch changing rather than as a guess
                  about what the video did.
  silence tick    drives `on_idle_tick`. The orchestrator owns no timer of its own,
                  which is what lets the whole machine be tested on a fake clock.

Concurrency is one session per socket with no pooling: a renderer is constructed and warmed at
connect time and torn down at disconnect. For a GPU renderer that is the wrong shape --
cold-loading weights per session is exactly the cost §1.4 argues cannot be paid at conversation
start -- and it is deferred to M7 rather than pretended away.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncIterator, Mapping
from typing import Any

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Loaded before every `avatar` import below, and the position is load-bearing rather than tidy.
# Without it .env was inert: every run needed `set -a && . ./.env && set +a` in front of it, and
# forgetting produced a session that silently fell back to every placeholder -- no error, just
# quietly the wrong system.
#
# It has to be *above* the imports because `avatar.store` chooses its backend from
# AVATAR_STORE at import time, and the routers pull that module in transitively. Loading the
# env files after them read AVATAR_STORE too late to matter: the API answered from JSON files
# while reporting no error, and a rubric created through it was invisible in psql.
# `avatar.config` imports nothing from this package, so it is safe to reach for first. The
# alternative -- a lazy proxy around the store -- adds indirection at every call site to solve
# an ordering problem that one line solves.
from avatar.config import load_env, loaded_files

_FROM_ENV_FILE = load_env()

from avatar import warmup
from avatar.agent_config import ResolvedAgent, build_llm_with_tools, resolve_for_session
from avatar.api import (
    agents,
    faces,
    guardrails,
    knowledge,
    pronunciations,
    rubrics,
    sessions,
    tools,
    voices,
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
from avatar.contracts import FRAME_INTERVAL_MS, TARGET_FPS, RendererConfig
from avatar.idle import placeholder_idle_loop
from avatar.knowledge.augment import with_knowledge, with_pronunciation
from avatar.knowledge.guard import with_guardrail
from avatar.mixer import FrameMixer
from avatar.orchestrator import RENDER_LEAD_IN_FRAMES, SessionOrchestrator
from avatar.plan import with_plan
from avatar.renderers import build
from avatar.state import State
from avatar.store import store
from avatar.telemetry import STAGE_TURN_DETECT, Telemetry
from avatar.transport.websocket import WebSocketTransport

FRAME_WIDTH = 256
FRAME_HEIGHT = 144
"""
Small on purpose.

Uncompressed BMP at 25fps costs width * height * 3 * 25 bytes/sec -- about 2.7MB/s at this size.
That is fine on localhost and indefensible over a network, and it is a consequence of having no
encoder rather than a considered choice. The real renderer emits JPEG or WebP in M2 and this
constraint disappears; until then, small frames keep the demo honest about where the bytes go.
Recorded in PROCESS.md 3.4.
"""

RENDERER_FIRST_FRAME_DELAY_MS = 200
"""
Audio the stub renderer requires before emitting a frame.

Not arbitrary: real talking-head models need a lookahead window, and setting this to zero would
make the first-frame latency readout meaningless and let the lead-in buffer look unnecessary.
"""

IDENTITY_REFERENCE = os.environ.get("AVATAR_REFERENCE", "assets/reference.mp4")
RENDERER_NAME = os.environ.get("AVATAR_RENDERER", "stub")
"""
The one-line renderer swap, as an environment variable.

`AVATAR_RENDERER=musetalk uvicorn avatar.server:app` is the whole change once M2 lands. Nothing
else in this file mentions a model.
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

Defaults to `tone` for the same reason as the LLM and the VAD: a clean clone has to run with no
credentials and no network.
"""

LLM_NAME = os.environ.get("AVATAR_LLM", "scripted")
"""
Which interviewer to run. `scripted` needs nothing; `anthropic` needs a key.

Defaults to `scripted` so a clean clone runs with no credentials and no network, which is what
the README promises. `AVATAR_LLM=anthropic` is the whole switch.
"""

VAD_NAME = os.environ.get("AVATAR_VAD", "energy")
"""
Which speech detector to run. `energy` needs nothing; `silero` needs torch.

Turn detection happens server-side rather than in the browser. The trade-off, stated because it
is a real one: the client streams microphone audio continuously, which costs bandwidth and means
candidate audio reaches the server even between turns. In exchange, the turn-taking policy is
one implementation with one set of thresholds that can be tested and tuned centrally, rather
than whatever each browser happened to ship. For an interview product the second consideration
wins; for a consumer toy it might not.
"""

SILENCE_TICK_SECONDS = 1.0
STATS_INTERVAL_SECONDS = 0.5
RELAY_QUEUE_DEPTH = 256

RELAYED_EVENTS = frozenset(
    {
        "state_change",
        "latency",
        "stale_dropped",
        "session_failure",
        "counter",
        "heard",
        "said",
        "plan",
    }
)
"""
Which telemetry events reach the browser.

`frame_repeated` is excluded despite being one of the most interesting signals: it fires up to
25 times a second, and relaying it would spend the socket on instrumentation instead of video.
The count still reaches the page in the stats message.

`heard` and `said` are the conversation itself, so they belong here rather than only in a server
log. Both are once-per-turn-ish -- `said` is once per sentence -- so the volume argument that
excludes `frame_repeated` does not apply. **This allowlist is easy to forget: a new event is
silently invisible to the client until it is added here, which is exactly what happened to both
of these.**
"""

@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """
    Warm the renderer before accepting traffic.

    A lifespan rather than a background task, and awaited rather than fired -- see
    `avatar.warmup` for why. It never raises: a missing GPU becomes a loud log line and a
    server that still serves the console, which an operator needs to find out why.

    """
    await warmup.warm()
    yield


app = FastAPI(title="nod", docs_url=None, redoc_url=None, lifespan=_lifespan)

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
for _resource in (
    agents,
    faces,
    guardrails,
    knowledge,
    pronunciations,
    rubrics,
    sessions,
    tools,
    voices,
):
    app.include_router(_resource.router)


@app.get("/config")
async def config() -> dict[str, object]:
    """
    Which implementation each boundary resolved to, and which came from `.env`.

    Names only, never values -- most of what `.env` holds is a credential. Exists because "why
    is it still using the placeholder voice?" is otherwise answered by reading code.
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
        # Which store the process actually resolved to, not which one was configured. The
        # distinction earned its place: AVATAR_STORE was read after the routers had already
        # imported `avatar.store`, so the API served JSON files while `.env.development`
        # said postgres and nothing anywhere disagreed. Reporting the live object would have
        # made that visible in one request instead of a psql query that came up empty.
        "store": type(store).__name__,
        # So "why was the first session slow" has an answer that is not a guess.
        "warmup": warmup.report.as_dict(),
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

    def __init__(self, socket: WebSocket, *, session_id: str | None = None) -> None:
        self._session_id = session_id
        self._socket = socket
        self._relay: asyncio.Queue[Mapping[str, object]] = asyncio.Queue(
            maxsize=RELAY_QUEUE_DEPTH
        )

        # Resolved once per session, not per turn: the retriever indexes its whole corpus
        # here, so a turn pays a scored lookup rather than re-reading documents from disk
        # inside the latency budget.
        self._agent: ResolvedAgent = resolve_for_session(session_id)

        # One turn's worth of telemetry, accumulated as stages complete and written to the
        # session record when the turn ends. Held here rather than in the orchestrator because
        # persistence is not a state-machine concern -- the orchestrator emits events and does
        # not know a database exists.
        self._turn: dict[str, Any] = {}
        self._turns_written = 0
        # Empty until the plan reads its first turn, and empty for good on an agent with no
        # rubric -- `with_plan` returns the stream untouched in that case, so the callback that
        # would fill this is never wired up.
        self._plan: dict[str, Any] = {}

        self._telemetry = Telemetry()
        self._telemetry.subscribe(self._on_telemetry)

        self._vad = build_vad(VAD_NAME)
        self._stt = build_stt(STT_NAME)
        self._detector = TurnDetector(frame_ms=FRAME_MS)
        self._mic = bytearray()
        self._speech_probability = 0.0

        # WebRTC when the SFU is configured and a session id names a room, WebSocket otherwise.
        # Chosen per session rather than per process, so one deployment can serve both -- and so
        # a clean clone with no SFU still reaches a working prototype, as the README promises.
        #
        # The socket stays open either way. Even on the WebRTC path it carries the candidate's
        # microphone up and the control messages the console reads, because moving those to the
        # data channel too would mean two protocols to debug for no gain while the browser is
        # already connected here.
        self._transport: Any = WebSocketTransport(socket.send_bytes, socket.send_text)
        self._rtc: Any = None
        if session_id and _livekit_available():
            from avatar.transport.livekit import LiveKitTransport

            self._rtc = LiveKitTransport(
                f"session-{session_id}", width=FRAME_WIDTH, height=FRAME_HEIGHT
            )
            self._transport = _Tee(self._rtc, self._transport)
        self._mixer = FrameMixer(
            placeholder_idle_loop(width=FRAME_WIDTH, height=FRAME_HEIGHT),
            self._telemetry,
        )
        self._orchestrator = SessionOrchestrator(
            renderer=build(RendererConfig(name=RENDERER_NAME, options=renderer_options())),
            mixer=self._mixer,
            transport=self._transport,
            # Both boundaries are wrapped rather than the orchestrator being changed:
            # retrieval augments the prompt, a lexicon rewrites text before synthesis, and
            # neither is a session-lifecycle concern. The state machine cannot tell.
            # Order matters: the guardrail wraps the retrieval-augmented stream, so the input
            # check sees the candidate's words and the output check sees what the model said
            # after retrieval influenced it. Reversed, retrieved context would be policed as
            # though the candidate had spoken it.
            # The plan sits innermost, so retrieval keys on the candidate's answer rather than
            # on the brief the plan just appended: `latest_candidate_text` scans backwards for
            # the last user message and the plan's guidance is a system message, so the order is
            # actually safe either way -- but innermost also means retrieval runs against a
            # history the plan has already read, which keeps the two independent. The guardrail
            # stays outermost so its input check still sees the candidate's own words.
            llm=with_guardrail(
                with_knowledge(
                    with_plan(
                        build_llm_with_tools(self._agent),
                        self._agent.plan,
                        on_update=self._on_plan_update,
                    ),
                    self._agent.retriever,
                ),
                self._agent.guardrail,
            ),
            # The agent's own voice decides the engine, falling back to the process default.
            # `clone` needs the reference the agent resolved; the hosted engines ignore it.
            tts=with_pronunciation(
                build_tts(
                    self._agent.voice_provider or TTS_NAME,
                    reference_path=self._agent.voice_reference or "",
                ),
                self._agent.lexicon,
            ),
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
        # The agent's own face, falling back to the environment. Until this line the faces
        # resource was decorative: `face_id` was resolved and then read by nothing, so a face
        # attached in the console prepared successfully and never appeared in a session.
        await self._orchestrator.start(self._agent.face_reference or IDENTITY_REFERENCE)
        self._persist_recording()

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
            # A candidate who closes the tab mid-answer must still leave the turn behind:
            # an abandoned interview is exactly the record worth having.
            self._flush_turn()

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
            # After the orchestrator, not before. Emitting `heard` unconditionally reported a
            # transcript for turns the state machine refused -- so a client that sent an
            # end-of-turn without a preceding `speech_start` saw its answer in the transcript
            # pane and in the session record while the model never received it, and the next
            # question read as the interviewer ignoring them. The refusal is now the loud thing.
            if await self._orchestrator.on_end_of_turn(spoken):
                self._telemetry.heard(
                    spoken, epoch=self._orchestrator.epoch, transcribed=bool(spoken)
                )
            else:
                self._telemetry.increment("turn_refused", state=str(self._orchestrator.state))
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

        Frames are fixed-size because Silero requires exactly 512 samples; the buffer exists to
        absorb whatever chunk size the browser happens to deliver. Leftover bytes stay buffered
        rather than being padded out, since a short frame scored as a full one reads as a VAD
        that misses quiet speech.

        Note what is *not* here: no gate on the session state. The microphone is processed while
        the avatar is speaking, because that is precisely when barge-in has to work. Keeping the
        avatar's own voice out of this path is the browser's echo cancellation, not ours — and
        if that fails, the avatar interrupts itself in a loop. A different VAD would not fix it.
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
            # Emitted only for a turn the machine accepted, and still ahead of any `said`:
            # `on_end_of_turn` spawns the turn rather than awaiting the model, and the LLM's
            # first token measured 1.6-4.7s behind it. So the log keeps showing what the LLM
            # was given before what it replied, without the event ever describing a turn that
            # was refused. Without `heard` at all an empty transcript is invisible -- the
            # interviewer asks a plausible question with no relation to the answer, which reads
            # as a model problem rather than the STT problem it is.
            #
            # This path reaches LISTENING by construction, since the detector only emits
            # END_OF_TURN after an onset. The refusal branch is therefore expected to be dead,
            # and it is here so that if the invariant ever breaks it says so.
            if await self._orchestrator.on_end_of_turn(heard):
                self._telemetry.heard(
                    heard, epoch=self._orchestrator.epoch, transcribed=bool(transcript)
                )
            else:
                self._telemetry.increment("turn_refused", state=str(self._orchestrator.state))

    # -- background pumps ---------------------------------------------------

    async def _pump_frames(self) -> None:
        """
        Drain the mixer forever.

        Deliberately not conditional on state: the track carries frames while idle, while
        listening, and while thinking. Gating this on SPEAKING is the bug that produces a track
        which stalls between turns.
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

        Dropping on a full queue is the right failure: a client too slow to keep up with the
        event stream should lose readouts, not stall the render loop that is emitting them.
        """
        self._accumulate(record)
        if record.get("event") not in RELAYED_EVENTS:
            return
        with contextlib.suppress(asyncio.QueueFull):
            self._relay.put_nowait(record)

    def _accumulate(self, record: Mapping[str, object]) -> None:
        """
        Build up the current turn from the events already being emitted, and persist it when the
        turn ends.

        Reading the existing telemetry rather than adding write calls throughout the
        orchestrator: the events are already the authority on what happened, so a second path
        recording the same facts could disagree with the first, and the transcript would then
        contradict the log. Nothing here can raise — instrumentation must never be able to end a
        conversation, so every failure is swallowed deliberately.
        """
        event = record.get("event")
        try:
            if event == "heard":
                # A new turn starts when the candidate's words arrive. Flush whatever the
                # previous turn accumulated first, in case it was cut off before its own end.
                self._flush_turn()
                # Every field initialised, including the timings that may never arrive. A
                # stage that did not complete must persist as an explicit null rather than an
                # absent key: the console reads these positionally, and `undefined` where a
                # number was expected renders as NaN instead of the honest dash.
                self._turn = {
                    "epoch": int(str(record.get("epoch", 0) or 0)),
                    "heard": str(record.get("text", "")),
                    "said": "",
                    "transcribed": bool(record.get("transcribed")),
                    "interrupted": False,
                    "llm_ttft_ms": None,
                    "tts_first_audio_ms": None,
                    "first_frame_ms": None,
                    "perceived_total_ms": None,
                }
            elif event == "said" and self._turn:
                self._turn["said"] = str(self._turn.get("said", "")) + str(
                    record.get("text", "")
                )
            elif event == "latency" and self._turn:
                stage = str(record.get("stage", ""))
                field = {
                    "llm_ttft": "llm_ttft_ms",
                    "tts_first_audio": "tts_first_audio_ms",
                    "avatar_first_frame": "first_frame_ms",
                    "perceived_total": "perceived_total_ms",
                }.get(stage)
                if field:
                    self._turn[field] = float(str(record.get("ms", 0) or 0))
            elif event == "stale_dropped" and self._turn:
                # Kept, but it is no longer what records a barge-in -- see the CANCELLING branch
                # below. A stale artifact can still arrive during a turn that was cancelled and
                # restarted quickly, and marking it costs nothing.
                self._turn["interrupted"] = True
            elif event == "state_change" and self._turn:
                # The barge-in itself, and this is the only branch that reliably catches it.
                #
                # `stale_dropped` used to be the sole writer of this flag and never fired in
                # time: `_cancel_turn` transitions to CANCELLING one line *before* the epoch
                # bump that makes any artifact stale, so the transition arrives first, the
                # `from=SPEAKING` case below flushes the turn and clears `self._turn`, and the
                # `and self._turn` guard on the stale branch then drops the event. Every
                # mid-speech interruption -- the common case, and the one the whole epoch design
                # exists for -- was stored as `interrupted: false`, and `interview_quality`
                # reported zero barge-ins for every session ever recorded.
                #
                # Entering CANCELLING *is* the barge-in, so it is read directly rather than
                # inferred from a downstream symptom that races the flush.
                if str(record.get("to")) == "CANCELLING":
                    self._turn["interrupted"] = True
                # A turn is over when the machine leaves SPEAKING or CANCELLING. This is the
                # trigger rather than `perceived_total`, which was the first attempt and was
                # wrong: that stage only fires when the *client* reports having painted, so a
                # client that never reports -- a headless driver, a candidate whose tab is
                # backgrounded, anything mid-crash -- would persist no turns at all while the
                # conversation looked completely normal. The state machine always transitions.
                if str(record.get("from")) in ("SPEAKING", "CANCELLING"):
                    self._flush_turn()
        except Exception:  # pragma: no cover - instrumentation must never break a session
            self._turn = {}

    def _persist_recording(self) -> None:
        """
        Store what happened when recording was set up, once, after the transport has connected.

        Written even when the answer is "off" or "unavailable", which is the whole reason it is
        stored at all: a session record that says nothing about recording is indistinguishable
        from one where recording silently failed, and the difference only matters at the moment
        someone asks for the video -- long after anyone could act on it.

        Called after `orchestrator.start`, because that is what opens the transport and
        therefore what creates the room. Reading it earlier would always report the
        constructor's placeholder.
        """
        if self._rtc is None or not self._session_id:
            return
        try:
            store.update("sessions", self._session_id, {"recording": dict(self._rtc.recording)})
        except Exception:  # pragma: no cover - a write failure must not end the interview
            return

    def _on_plan_update(self, snapshot: Mapping[str, object]) -> None:
        """
        Record and relay coverage after the plan has read a turn.

        Written to the session record whole rather than appended, because coverage is cumulative
        session state, not a per-turn event: the current snapshot already contains everything
        the earlier ones said. Appending a snapshot per turn would store the same evidence n
        times and leave a reader to work out which copy is authoritative.

        Persisted here rather than in `_flush_turn` because the two have different clocks. A
        turn flushes when the state machine leaves SPEAKING, which a barge-in during THINKING
        skips entirely -- and a question that was asked and interrupted still consumed one of
        the competency's `max_turns`, so coverage that only survived completed turns would drift
        below what the interview actually spent.

        Wrapped, like every other instrumentation path here: a store write failing must not end
        an interview that is otherwise working.
        """
        self._plan = dict(snapshot)
        try:
            self._telemetry.plan_update(self._plan, epoch=self._orchestrator.epoch)
            if self._session_id:
                store.update("sessions", self._session_id, {"coverage": self._plan})
        except Exception:  # pragma: no cover - instrumentation must never break a session
            return

    def _flush_turn(self) -> None:
        """Append the accumulated turn to the session record, if there is one to append to."""
        if not self._turn or not self._session_id:
            self._turn = {}
            return
        turn, self._turn = self._turn, {}
        try:
            record = store.get("sessions", self._session_id)
            if record.get("ended_at"):
                return  # the record is closed; appending would claim the session continued
            turns = [*(record.get("turns") or []), turn]
            store.update("sessions", self._session_id, {"turns": turns})
            self._turns_written += 1
        except Exception:  # pragma: no cover - a write failure must not end the interview
            return

    async def _send(self, payload: Mapping[str, object]) -> None:
        with contextlib.suppress(Exception):
            await self._socket.send_text(json.dumps(payload, default=str))


def renderer_options() -> dict[str, object]:
    """
    The options every renderer must accept, whichever one `AVATAR_RENDERER` selects.

    A function with a name, rather than a dict literal inlined at the call site, so a test can
    assert that each renderer accepts exactly this. That test is not bookkeeping: selecting
    `musetalk` used to raise `TypeError: unexpected keyword argument 'width'` at the instant a
    candidate opened their interview link, because this dict was shaped entirely by the stub and
    nothing checked the other renderer against it. `isinstance(r, TalkingHeadRenderer)` did not
    catch it -- a runtime-checkable Protocol compares method names, not signatures, and says
    nothing at all about constructors.

    A renderer is free to ignore an option it has no use for. What it may not do is refuse it.
    """
    return {
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "first_frame_delay_ms": RENDERER_FIRST_FRAME_DELAY_MS,
        "frame_interval_ms": FRAME_INTERVAL_MS,
    }


@app.websocket("/session")
async def session_socket(socket: WebSocket, session: str | None = None) -> None:
    """
    `?session=<id>` binds this socket to a stored session record.

    That id is what the candidate's link carries, and it is how configuration reaches the
    runtime without an environment variable: the record names an agent, and the agent names a
    knowledge base, a lexicon, a guardrail and a face. `AVATAR_AGENT` remains as a fallback for
    running the prototype with no console data, which the README promises still works.
    """
    await BrowserSession(socket, session_id=session).run()


def _livekit_available() -> bool:
    """
    Whether an SFU is configured. Checked once per session, not per frame.

    Deliberately only a credentials check, not a connection attempt: probing the SFU here would
    put a network round trip on the path a candidate takes to open their interview, and the
    connection failure surfaces at `open_track` anyway with a better message.
    """
    return bool(os.environ.get("LIVEKIT_API_KEY") and os.environ.get("LIVEKIT_API_SECRET"))


class _Tee:
    """
    Sends to both transports, so a session is watchable over WebRTC *and* instrumented over the
    socket the console already speaks.

    Not a permanent shape. It exists because the browser needs media over WebRTC while the
    console's telemetry, transcript and latency readouts arrive over the WebSocket, and moving
    those to the data channel is a client change rather than a server one. The WebRTC leg is
    primary: if it fails, the socket leg continues regardless, because a session that degrades
    to a working WebSocket beats one that dies.
    """

    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    async def open_track(self) -> None:
        await self._secondary.open_track()
        with contextlib.suppress(Exception):
            await self._primary.open_track()

    async def send_audio(self, chunk: Any) -> None:
        await self._secondary.send_audio(chunk)
        with contextlib.suppress(Exception):
            await self._primary.send_audio(chunk)

    async def send_frame(self, frame: Any) -> None:
        await self._secondary.send_frame(frame)
        with contextlib.suppress(Exception):
            await self._primary.send_frame(frame)

    async def flush_audio(self) -> None:
        await self._secondary.flush_audio()
        with contextlib.suppress(Exception):
            await self._primary.flush_audio()

    async def close_track(self) -> None:
        with contextlib.suppress(Exception):
            await self._primary.close_track()
        await self._secondary.close_track()
