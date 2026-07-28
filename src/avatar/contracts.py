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
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypedDict, runtime_checkable

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
