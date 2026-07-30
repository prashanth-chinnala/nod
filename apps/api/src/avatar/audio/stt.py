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
            return
        self._reader = asyncio.create_task(self._read_loop(), name="stt-reader")

    async def _open(self) -> object:
        if self._connect is not None:
            return await self._connect(self.url)  # type: ignore[operator]
        import websockets

        return await websockets.connect(
            self.url, additional_headers={"Authorization": f"Token {self._api_key}"}
        )

    async def push_audio(self, pcm: bytes) -> None:
        """
        Forward one frame. Never raises, never blocks on reconnect.

        A dropped connection is not repaired here: doing so would put a TLS handshake on
        the audio path and stall the VAD behind it. `connect()` is retried at the next
        session, and the turn meanwhile carries a placeholder.
        """
        socket = self._socket
        if socket is None:
            return
        try:
            await socket.send(pcm)  # type: ignore[attr-defined]
            self.bytes_sent += len(pcm)
        except Exception:
            self._socket = None
            self.reconnects += 1

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
