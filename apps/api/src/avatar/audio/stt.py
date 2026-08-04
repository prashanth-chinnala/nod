"""
Speech to text: Deepgram Nova over a persistent WebSocket.

Implements `Transcriber`. Fed the same microphone frames as the VAD, transcribes
continuously, and hands over its text when the turn policy — not Deepgram — says the
turn is finished.

**Why the vendor's endpointing is not used.** Every streaming STT ships its own
utterance detection, and Deepgram's is good. Using it would still be wrong here: it moves
the single most consequential decision in the system into a vendor default that nobody on
this side can see, tune, or test. `audio.turn_detection` is 30 tests over probability
sequences with separately tuned onset, hysteresis, retraction, and end-of-turn
thresholds, each justified by the conversational failure it prevents. Handing that to a
remote service to replace with one number is a bad trade for an interview product, where
cutting a candidate off is worse than almost any other failure.

The cost of that choice, stated plainly: the transcript is whatever had been finalised by
the moment the policy fires, so a word still in flight can be dropped. Interim results are
tracked so the gap is visible rather than silent.

**Measured against real speech** (5.48s of synthesised audio streamed in real time,
`nova-3`, 16kHz mono):

| | |
|---|---|
| connect | ~910ms, once per session |
| first interim | ~800ms into the stream |
| transcript | exact, including punctuation |

Connect cost is why `connect()` is separate from `push_audio` -- the server opens this at
session start, while the candidate is still being greeted, rather than paying ~900ms
inside the first turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Callable

from avatar.audio.vad import SAMPLE_RATE
from avatar.contracts import Transcriber

LISTEN_URL = "wss://api.deepgram.com/v1/listen"

DEFAULT_MODEL = "nova-3"

QUERY = {
    "encoding": "linear16",
    "sample_rate": str(SAMPLE_RATE),
    "channels": "1",
    "punctuate": "true",
    # Interim results are requested even though only finals are used for history. They
    # are the cheapest way to see how far behind the finaliser is running, which is the
    # exact gap this design accepts by keeping turn detection local.
    "interim_results": "true",
    # Endpointing, so Deepgram decides an utterance has ended from the audio rather than
    # waiting to be told. Without it a final only arrives when the stream is finalised or
    # closed -- and this transcriber is never closed mid-interview, so finals never came.
    "endpointing": "300",
    # And a hard backstop: emit a final at least this often even mid-speech. An interview
    # answer can run past any endpointing window, and a transcript that only materialises
    # when someone stops talking is not one a turn boundary can rely on.
    "utterance_end_ms": "1000",
}


class NullTranscriber:
    """
    No transcription. The default, so a clean clone runs with no credentials.

    Returns an empty transcript, which the server renders as a placeholder describing the
    utterance's duration. The orchestrator does not care -- it appends whatever it is
    given to history, which is what makes STT a drop-in.
    """

    async def push_audio(self, pcm: bytes) -> None:
        return None

    def take_transcript(self) -> str:
        return ""

    async def aclose(self) -> None:
        return None


class DeepgramSTT:
    """
    Streaming transcription over one long-lived WebSocket.

    Two properties matter for the audio path and shape the design:

    `push_audio` must never block. It is called from the socket read loop that also feeds
    the VAD, so a stalled or reconnecting transcriber must not stall turn detection. Sends
    are therefore fire-and-forget, and a send failure drops the connection for lazy
    reconnect rather than raising into the caller.

    Transcription is strictly best-effort. If it fails, turns still happen -- they carry a
    placeholder instead of words. Losing the transcript degrades the interview; blocking
    the audio path would end it.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        connect: object | None = None,
        clock: Callable[[], float] | None = None,
        keep_alive_interval: float = 4.0,
        max_reconnect_attempts: int = 6,
        reconnect_base_delay: float = 0.25,
    ) -> None:
        self.model = model
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        if connect is None and not self._api_key:
            raise RuntimeError(
                "DEEPGRAM_API_KEY is not set. Put it in .env (gitignored) and run with "
                "`set -a && . ./.env && set +a`, or export it. "
                "Run with AVATAR_STT=none to skip transcription."
            )
        self._connect = connect
        self._socket: object | None = None
        self._reader: asyncio.Task[None] | None = None
        self._finals: list[str] = []
        self._latest_interim = ""
        """
        The most recent non-final transcript.


        Kept as a fallback because `take_transcript` is called at the turn boundary, and
        Deepgram's finals are not synchronous with it: a final arrives on endpointing or when
        the
        stream is finalised, neither of which is guaranteed to have happened by the moment the
        turn-taking policy declares the turn over. Measured on this exact path -- 3 interim
        results, zero finals, and a turn recorded as `[3.7s of speech, no transcript]` for a
        sentence Deepgram had already transcribed correctly.


        Falling back to an interim is a real trade: it can be a word or two behind the final,
        and
        it is not punctuated as well. Against that, the alternative was throwing the whole
        transcript away, which is what the interview did for 10 of 11 turns.
        
        """
        self.interim_count = 0
        self.bytes_sent = 0
        self.reconnects = 0
        self.keep_alives = 0
        self._reconnector: asyncio.Task[None] | None = None
        self._keeper: asyncio.Task[None] | None = None
        self._clock = clock or time.monotonic
        self._last_audio_at = self._clock()
        self._keep_alive_interval = keep_alive_interval
        """
        Comfortably inside Deepgram's ~10 s idle timeout, and measured against the wrong thing
        if it is tuned by feel: the interval that matters is how long the *candidate* is silent
        while the avatar speaks, which is a whole answer long.
        """
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_base_delay = reconnect_base_delay

    @property
    def url(self) -> str:
        params = "&".join(f"{k}={v}" for k, v in {"model": self.model, **QUERY}.items())
        return f"{LISTEN_URL}?{params}"

    @property
    def connected(self) -> bool:
        return self._socket is not None

    async def connect(self) -> None:
        """
        Open the socket. Called at session start so no turn pays the ~900ms connect.

        Failure is swallowed on purpose: a session without transcription is degraded, a
        session that refuses to start because STT was unreachable is broken.
        """
        if self._socket is not None:
            return
        try:
            self._socket = await self._open()
        except Exception:
            self._socket = None
            # Scheduled even though this is session start, because "unreachable right now" and
            # "unreachable for the whole interview" are different, and the old code treated them
            # as the same by never trying again.
            self._schedule_reconnect()
            return
        self._reader = asyncio.create_task(self._read_loop(), name="stt-reader")
        self._last_audio_at = self._clock()
        # Owned here rather than by the server, so every caller of `connect()` gets it and
        # nothing has to remember. The keepalive is what stops the socket dying of idleness
        # while the avatar talks; the reconnector is what recovers when something else kills it.
        # Both are needed, and neither is sufficient.
        if self._keeper is None or self._keeper.done():
            self._keeper = asyncio.create_task(self.keep_alive(), name="stt-keepalive")

    async def _open(self) -> object:
        if self._connect is not None:
            return await self._connect(self.url)  # type: ignore[operator]
        import websockets

        return await websockets.connect(
            self.url, additional_headers={"Authorization": f"Token {self._api_key}"}
        )

    async def push_audio(self, pcm: bytes) -> None:
        """
        Forward one frame. Never raises, never blocks on a handshake.

        **Repair is scheduled here, not performed here**, and the distinction is the whole
        point. An earlier version of this method dropped the socket and left it dropped: the
        class docstring said "for lazy reconnect" and nothing anywhere reconnected. `connect()`
        runs once at session start, so one dropped socket made the interviewer deaf for the rest
        of the interview.

        That is not a theoretical failure. In a real session two turns transcribed and the
        remaining thirty-eight did not, including one of **10,080 ms of speech** -- the VAD
        detected it correctly, the transcriber was gone, and the interviewer went on asking
        plausible questions of someone it could not hear. Nothing errored, which is why it
        survived this long.

        The original reason for not reconnecting was sound and is preserved: a TLS handshake on
        the audio path would stall the VAD behind it. So the handshake happens in a background
        task, and this method never waits for it. Frames during the gap are lost, which costs a
        word or two at the start of one turn instead of every word for the rest of the session.
        """
        socket = self._socket
        if socket is None:
            self._schedule_reconnect()
            return
        try:
            await socket.send(pcm)  # type: ignore[attr-defined]
            self.bytes_sent += len(pcm)
            self._last_audio_at = self._clock()
        except Exception:
            self._socket = None
            self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        """
        Start a background reconnect unless one is already running. Synchronous and cheap.

        Guarded by the task's own liveness rather than a flag, because a flag and a task are two
        representations of one fact and they drift: the failure mode is a flag left set by a
        task that died, after which nothing ever reconnects again -- the exact bug this method
        exists to fix, reintroduced one level up.
        """
        if self._reconnector is not None and not self._reconnector.done():
            return
        self._reconnector = asyncio.create_task(self._reconnect(), name="stt-reconnect")

    async def _reconnect(self) -> None:
        """
        Reopen the socket, backing off. Gives up after a bounded number of attempts.

        Bounded rather than infinite: an unreachable or unauthorised Deepgram will never come
        back within one interview, and a task retrying it forever would keep a dead session
        warm. `reconnects` counts the successes so `/config` and the session stats can show that
        this happened at all -- a transcript gap that nobody can attribute is how the original
        bug stayed hidden.
        """
        for attempt in range(self._max_reconnect_attempts):
            await asyncio.sleep(min(self._reconnect_base_delay * (2**attempt), 5.0))
            if self._socket is not None:  # a later connect() won the race
                return
            try:
                self._socket = await self._open()
            except Exception:
                continue
            self._reader = asyncio.create_task(self._read_loop(), name="stt-reader")
            self.reconnects += 1
            self._last_audio_at = self._clock()
            return

    async def keep_alive(self) -> None:
        """
        Send Deepgram's `KeepAlive` while no audio is flowing. Prevents the drop entirely.

        **Why this is needed even with reconnection.** Deepgram closes a streaming connection
        that has been idle for about ten seconds, and this transcriber goes idle every time the
        avatar is the one talking -- which on a long answer is easily longer than that. So the
        socket was not dying of network trouble; it was dying of politeness, every single turn,
        and reconnecting after the fact still loses the first words of the candidate's reply.

        Run as a background task for the life of the session. Never raises: a failed keepalive
        is indistinguishable from the socket already being gone, which the reconnector handles.
        """
        while True:
            await asyncio.sleep(self._keep_alive_interval)
            socket = self._socket
            if socket is None:
                continue
            if self._clock() - self._last_audio_at < self._keep_alive_interval:
                continue
            try:
                await socket.send(json.dumps({"type": "KeepAlive"}))  # type: ignore[attr-defined]
                self.keep_alives += 1
            except Exception:
                self._socket = None
                self._schedule_reconnect()

    def take_transcript(self) -> str:
        """
        Everything finalised since the last call, joined and cleared.

        Synchronous by design -- the orchestrator calls it from the turn-boundary path,
        and awaiting a transcriber there would let a slow STT delay the turn the policy
        has already decided is over.
        """
        text = " ".join(self._finals).strip()
        self._finals.clear()
        if not text:
            # No final for this turn. Return what Deepgram had heard rather than nothing.
            text = self._latest_interim.strip()
        self._latest_interim = ""
        return text

    async def _read_loop(self) -> None:
        socket = self._socket
        if socket is None:
            return
        try:
            async for message in socket:  # type: ignore[attr-defined]
                if isinstance(message, bytes):
                    continue
                self._handle(message)
        except Exception:
            # Includes normal closure. The socket is gone either way.
            pass
        finally:
            self._socket = None

    def _handle(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if payload.get("type") != "Results":
            return
        alternatives = payload.get("channel", {}).get("alternatives") or [{}]
        text = (alternatives[0].get("transcript") or "").strip()
        if not text:
            return
        if payload.get("is_final"):
            self._finals.append(text)
            self._latest_interim = ""
        else:
            self.interim_count += 1
            self._latest_interim = text

    async def aclose(self) -> None:
        socket, self._socket = self._socket, None
        # Cancelled before the socket is closed, so neither task can reopen what is being torn
        # down. A reconnector that outlived aclose() would leave a live Deepgram stream behind
        # every finished interview -- billed, and attached to nothing.
        for task in (self._keeper, self._reconnector):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._keeper = self._reconnector = None
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
            self._reader = None
        if socket is not None:
            with contextlib.suppress(Exception):
                # Tells Deepgram to finalise rather than dropping the stream mid-word.
                await socket.send(  # type: ignore[attr-defined]
                    json.dumps({"type": "CloseStream"})
                )
            with contextlib.suppress(Exception):
                await socket.close()  # type: ignore[attr-defined]


def build_stt(name: str = "none") -> Transcriber:
    """
    The one-line STT swap, mirroring the renderer, VAD, LLM, and TTS registries.

    Defaults to `none` so a clean clone runs with no credentials and no network.
    """
    key = name.lower()
    if key in ("none", "null"):
        return NullTranscriber()
    if key == "deepgram":
        return DeepgramSTT()
    raise ValueError(f"unknown STT {name!r}; available: 'none', 'deepgram'")
