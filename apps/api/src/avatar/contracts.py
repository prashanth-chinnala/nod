"""
Data contracts and the interfaces between the four subsystems.

This module is the root of the dependency graph. It imports nothing from the rest
of the package, and every other module imports *from* it. If an import cycle ever
appears involving this file, a boundary has been violated.

It holds declarations only -- dataclasses, Protocols, type aliases. No behaviour.
Policy that operates on these types lives in the module that owns the policy
(history truncation in `orchestrator`, cadence in `mixer`), so that each rule has
exactly one home and one test.

Four boundaries are declared here, and the point of each is that the thing behind
it is replaceable without the state machine noticing:

    TalkingHeadRenderer  the ML model. Knows nothing about turns or sessions.
    Transport            how frames and audio reach the client.
    SentenceStream       the LLM, already chunked into speakable units.
    SpeechStream         the TTS.
    Transcriber          the STT. Accumulates words; decides nothing.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypedDict, runtime_checkable

TARGET_FPS = int(os.environ.get("AVATAR_FPS", 25))
"""
Frames per second the whole pipeline runs at. `AVATAR_FPS` overrides. 25 is the default and the
target.

**Why it is here, and why it is one number.** Three places have to agree: the mixer's cadence,
the interval stamped on each frame, and the `fps=` that Whisper's chunker slices audio features
with. Written down separately they drift, and the failure is not loud -- the mouth slides
against the speech slowly enough to read as bad dubbing. `contracts.py` imports nothing from
this package, so this is the one module every layer can read without any layer depending on
another.

**Why it is configurable at all, which is the part that matters.** A renderer that cannot reach
its target does not degrade gracefully; it fails completely. The mixer drops any frame that
misses its slot, so at 3.3 fps against a 25fps clock every frame is late and every frame is
discarded -- measured on an M1 Pro, three turns rendered 169 frames and delivered zero, and the
candidate watched the placeholder while the interviewer talked. A Tesla T4 measures 8.7 fps,
which fails the same way for the same reason.

So on hardware that cannot hold 25fps, lowering this is the difference between choppy video and
no video. It is a real loss -- lip motion at 8fps is visibly less smooth -- and it is not a
substitute for a faster renderer. It is what makes the renderer's actual output visible instead
of dropped. See MEASUREMENTS.md for the per-device figures.
"""

FRAME_INTERVAL_MS = round(1000 / TARGET_FPS)
"""Milliseconds per frame, derived. 40 at 25fps, 125 at 8fps."""

IDLE_EPOCH = 0
"""
Epoch carried by idle-loop frames.

Turn epochs start at 1, so an idle frame can never collide with a turn and is
never treated as stale. Idle frames belong to no turn: they are correct to show
in any state.
"""


@dataclass(frozen=True, slots=True)
class Frame:
    """One encoded video frame, tagged with the turn that produced it."""

    data: bytes
    epoch: int
    pts_ms: int


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """
    One chunk of synthesised speech, tagged with the turn that produced it.

    `duration_ms` is the wall-clock playback duration, not the byte length. The
    orchestrator uses it to reason about how much of a turn the listener has
    actually heard, so it must reflect real playback time.
    """

    pcm: bytes
    epoch: int
    duration_ms: int


class Message(TypedDict):
    """One entry of conversation history, in the shape chat APIs expect."""

    role: str
    content: str


@dataclass
class Turn:
    """
    One assistant utterance in flight.

    Tracks three quantities that are easy to conflate and must not be:

      text_generated   what the LLM produced
      audio_sent_ms    what was handed to the transport
      audio_played_ms  what the client acknowledged as played

    Only the third is evidence about what the candidate heard. The gap between
    the second and the third is client-side buffer, and on a barge-in that buffer
    is discarded -- so crediting it to history would put words in the
    interviewer's mouth that the candidate never received.
    """

    epoch: int
    text_generated: str = ""
    audio_sent_ms: int = 0
    audio_played_ms: int = 0
    interrupted: bool = False
    started_at: float = 0.0
    first_frame_at: float | None = None
    first_paint_at: float | None = None
    """
    When the client reported painting this turn's first frame.

    Distinct from `first_frame_at`, which is when the server handed the frame to the
    mixer. The gap between them is encode, socket, decode, and a paint -- the part
    of the latency budget a server-side measurement cannot see.
    """


@runtime_checkable
class TalkingHeadRenderer(Protocol):
    """
    The one bounded, swappable ML component.

    Deliberately absent from this interface: any notion of a turn, a session
    state, voice activity, conversation history, or transport. A renderer that
    needed those would be making orchestration decisions, which is exactly the
    coupling this Protocol exists to prevent.

    Implementations may block and may require a GPU. `prepare_identity` is
    explicitly allowed to be slow -- it runs once, offline, per persona.
    """

    def prepare_identity(self, reference_path: str) -> object:
        """Preprocess a reference image/video into a reusable identity artifact."""
        ...

    def start_session(self, identity: object) -> object:
        """Bind an identity to a render session. Should include any warm-up pass."""
        ...

    def push_audio(self, session: object, chunk: AudioChunk) -> None:
        """Feed one chunk of speech. Must accept chunks, never whole files."""
        ...

    def frames(self, session: object) -> Iterator[Frame]:
        """Drain whatever frames are ready now. Must not block waiting for more."""
        ...

    def reset(self, session: object) -> None:
        """
        Drop queued audio and in-flight frames without tearing down the session.

        This is the method that makes barge-in possible. It must be safe to call
        when nothing is in flight, and must not unload weights.
        """
        ...

    def close_session(self, session: object) -> None:
        """Release per-session resources. Must not leak GPU memory across sessions."""
        ...


class Transport(Protocol):
    """How audio and frames reach the client, and how the client acknowledges."""

    async def open_track(self) -> None: ...

    async def send_audio(self, chunk: AudioChunk) -> None: ...

    async def flush_audio(self) -> None:
        """
        Discard unplayed audio, server-side *and* client-side.

        A server-only flush leaves the client's buffer playing a sentence the
        avatar has already abandoned, which reads as a laggy interruption even
        though the state machine reacted instantly.
        """
        ...

    async def close_track(self) -> None: ...


class SentenceStream(Protocol):
    """
    The LLM, wrapped so it yields speakable units rather than tokens.

    Sentence chunking belongs on this side of the boundary: it is what lets TTS
    start before generation finishes, and that overlap is the whole reason a
    sub-second turnaround is achievable.
    """

    def __call__(self, history: Sequence[Message]) -> AsyncGenerator[str, None]:
        """
        Must return a closeable generator, not merely an iterator.

        The orchestrator closes this deterministically when a turn is abandoned, and
        for an HTTP-backed model that close is what aborts the request. An iterator
        with no `aclose` would leave the provider generating a response nobody hears.
        """
        ...


class SpeechStream(Protocol):
    """TTS, streaming audio chunks tagged with the turn that requested them."""

    def __call__(self, text: str, epoch: int) -> AsyncGenerator[AudioChunk, None]:
        """Closeable, for the same reason as `SentenceStream`."""
        ...


class Transcriber(Protocol):
    """
    Speech to text. Accumulates words and nothing else.

    Deliberately *not* a turn detector. Every streaming STT service ships its own
    endpointing, and using it would move the turn-taking decision into a vendor's
    defaults -- replacing a policy with 30 tests over probability sequences
    (`audio.turn_detection`) with a threshold nobody here can see or tune. So the
    transcriber is fed the same audio as the VAD, transcribes continuously, and is
    *asked* for its text when the turn policy says the turn is over.

    The consequence worth naming: the transcript is whatever had been finalised by that
    moment, so a word still in flight can be missed. That is the cost of keeping turn
    detection under local control, and it is the right trade for an interview product
    where a wrong turn boundary is far more damaging than a dropped final word.
    """

    async def push_audio(self, pcm: bytes) -> None:
        """Feed microphone audio. Must not block the caller's audio path."""
        ...

    def take_transcript(self) -> str:
        """Return everything finalised since the last call, and clear it."""
        ...

    async def aclose(self) -> None:
        """Release the connection. Safe to call more than once."""
        ...


Clock = Callable[[], float]
"""Monotonic seconds. Injected so tests can advance time without sleeping."""

Sleep = Callable[[float], Awaitable[None]]
"""Awaitable delay. Injected for the same reason as `Clock`."""


@dataclass(frozen=True, slots=True)
class RendererConfig:
    """
    Which renderer to construct, and with what.

    Swapping the ML model is a change to this value and nothing else -- see
    `avatar.renderers.build`.
    """

    name: str = "stub"
    options: dict[str, object] = field(default_factory=dict)
