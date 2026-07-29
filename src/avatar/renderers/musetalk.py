"""
MuseTalk behind `TalkingHeadRenderer`.

The model is a single-step latent inpainter: a frozen VAE encodes reference frames, a
frozen `whisper-tiny` encodes audio, and a Stable-Diffusion-v1.4-shaped U-Net fuses them by
cross-attention to repaint the mouth region. Nothing here diffuses -- there is one forward
pass per frame, which is the only reason 25fps is arguable at all.

**The friction this module exists to absorb.** MuseTalk's own realtime script is built
around a whole audio *file*: `get_audio_feature(audio_path)` reads a path, and
`get_whisper_chunk()` segments the entire clip up front. "Realtime" there means *faster than
realtime over a file*, not *streaming from a live microphone*. Our Protocol requires
`push_audio(chunk)` and a non-blocking `frames()`, because a conversation does not have a
file. Bridging those two is most of the code below, and it is a real architectural cost of
this model choice that belongs in the memo rather than hidden in a wrapper.

**Why there is a backend seam.** Every torch/CUDA call goes through `MuseTalkBackend`. That
is not indirection for its own sake: it is the only way the buffering, windowing, epoch
tagging, reset semantics, and frame pacing get tested at all, on a machine with no GPU. The
logic most likely to be wrong is the streaming glue, not the matrix multiplication, and the
glue is exactly what a fake backend exercises. The same reasoning as the stub renderer, one
level down.

Nothing in this module is imported by the orchestration layer, and every heavy import is
inside the function that needs it -- so `import avatar` stays GPU-free and CI stays green
without torch.

**Status: written against MuseTalk's documented API and unit-tested against a fake backend.
The real backend has never executed** -- there is no CUDA device in the development
environment, and the model spike has not yet produced a working install. Every number this
would produce is therefore still `NOT YET MEASURED`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from avatar.contracts import IDLE_EPOCH, AudioChunk, Frame

FRAME_INTERVAL_MS = 40
"""25fps. MuseTalk's reference cycle and Whisper's feature stride both assume it."""

SAMPLE_RATE = 16_000
"""
Fixed by two things at once, which is a convenience rather than a coincidence.

`whisper-tiny` expects 16kHz, and the transport already carries 16kHz mono PCM16 because
the voice-activity detector wants the same. So no resampling happens anywhere in the audio
path, and a resampler is one fewer thing to get subtly wrong.
"""

BYTES_PER_SAMPLE = 2
BYTES_PER_FRAME = SAMPLE_RATE * BYTES_PER_SAMPLE * FRAME_INTERVAL_MS // 1000  # 1280

WINDOW_FRAMES = 16
"""
How many frames' worth of audio to accumulate before running a batch.

The floor is set by Whisper needing context either side of a frame to encode it well; the
ceiling by latency, since a frame cannot emit until its window is full. 16 frames is 640ms
of audio, which at a batch size of 8 is two U-Net passes.

This is the single knob that trades first-frame latency against throughput, and it is
`NOT YET MEASURED` on real hardware. It should be tuned once the backend runs, not guessed
at repeatedly.
"""

CONTEXT_FRAMES = 2
"""
Frames of already-consumed audio to prepend to each window.

Without it, every window boundary is a discontinuity in the audio features and the mouth
visibly jumps at the window rate -- the characteristic streaming-talking-head artifact, and
periodic enough to be unmistakable once seen. Overlapping the context costs two redundant
feature extractions per window and removes the seam.
"""


@runtime_checkable
class MuseTalkBackend(Protocol):
    """
    Every GPU-touching operation, isolated so the streaming logic can be tested without one.

    An implementation may block and may take seconds to load. It must not know what a turn,
    an epoch, or a session is -- that is this module's job, and the reason the split exists.
    """

    def load(self) -> None:
        """Load weights onto the device. Called once per process, not per session."""
        ...

    def prepare(self, reference_path: str) -> object:
        """
        Face detection, parsing, and VAE encoding of the reference frames.

        Returns whatever the render step needs, opaque to this module. Allowed to be slow:
        it runs once per persona, offline.
        """
        ...

    def render(
        self, prepared: object, pcm: bytes, *, start_frame: int, count: int
    ) -> list[bytes]:
        """
        Render `count` frames beginning at `start_frame`, driven by `pcm`.

        `start_frame` indexes the reference cycle, so consecutive calls continue the body
        motion rather than restarting it. Returns encoded image bytes, one per frame, in
        order. May return fewer than `count` if the audio ran short.
        """
        ...

    def unload(self) -> None:
        """Release device memory. Must be safe to call twice."""
        ...


@dataclass(frozen=True, slots=True)
class MuseTalkIdentity:
    """A prepared persona. Reusable across sessions, which is the point of §1.2."""

    reference_path: str
    prepared: object


@dataclass
class MuseTalkSession:
    identity: MuseTalkIdentity
    pcm: bytearray = field(default_factory=bytearray, repr=False)
    context: bytes = b""
    """Tail of the previously rendered audio, prepended to the next window."""
    frames_emitted: int = 0
    epoch: int = IDLE_EPOCH
    closed: bool = False
    resets: int = 0


class MuseTalkRenderer:
    """
    Satisfies `TalkingHeadRenderer` structurally, like the stub does.

    Conformance is asserted in the test suite rather than by inheritance, so this cannot
    subclass its way to compliance while quietly taking different arguments.
    """

    def __init__(
        self,
        *,
        backend: MuseTalkBackend | None = None,
        window_frames: int = WINDOW_FRAMES,
        context_frames: int = CONTEXT_FRAMES,
        frame_interval_ms: int = FRAME_INTERVAL_MS,
    ) -> None:
        if window_frames < 1:
            raise ValueError(f"window_frames must be positive, got {window_frames}")
        if context_frames < 0:
            raise ValueError(f"context_frames must not be negative, got {context_frames}")
        self.window_frames = window_frames
        self.context_frames = context_frames
        self.frame_interval_ms = frame_interval_ms
        self._backend = backend
        self._loaded = False

    # -- backend wiring -----------------------------------------------------

    @property
    def backend(self) -> MuseTalkBackend:
        """
        The real backend, constructed on first use.

        Deferred because constructing it imports torch and loads several GB of weights.
        Doing that at module import would make `import avatar` fail on a laptop and drag
        CUDA into CI -- the two things the module boundary exists to prevent.
        """
        if self._backend is None:
            from avatar.renderers.musetalk_torch import TorchMuseTalkBackend

            self._backend = TorchMuseTalkBackend()
        if not self._loaded:
            self._backend.load()
            self._loaded = True
        return self._backend

    # -- the Protocol -------------------------------------------------------

    def prepare_identity(self, reference_path: str) -> object:
        return MuseTalkIdentity(
            reference_path=reference_path,
            prepared=self.backend.prepare(reference_path),
        )

    def start_session(self, identity: object) -> object:
        """
        Bind a prepared identity to a session.

        No warm-up pass here, deliberately. The first `render` call is slower than the rest
        -- CUDA context, cuDNN autotune, lazy weight loads -- and hiding that inside session
        start would move the cost somewhere it is not measured. It belongs in the first
        turn's first-frame latency, where §1.5 can see it.
        """
        if not isinstance(identity, MuseTalkIdentity):
            raise TypeError(f"expected a prepared MuseTalkIdentity, got {type(identity)!r}")
        return MuseTalkSession(identity=identity)

    def push_audio(self, session: object, chunk: AudioChunk) -> None:
        state = _as_session(session)
        if chunk.epoch != state.epoch:
            # A new turn. Drop whatever the previous one left buffered rather than
            # rendering the join between two unrelated utterances, and restart the
            # feature context -- carrying it across a turn boundary would condition the
            # first frames of this turn on the last words of the abandoned one.
            state.pcm.clear()
            state.context = b""
            state.epoch = chunk.epoch
            state.frames_emitted = 0
        state.pcm += chunk.pcm

    def frames(self, session: object) -> Iterator[Frame]:
        """
        Render every complete window currently buffered, then stop.

        Never blocks waiting for more audio: a window that is not yet full stays buffered
        and is picked up on the next call. That is what lets the mixer keep its cadence
        while the renderer is mid-utterance.
        """
        state = _as_session(session)
        window_bytes = self.window_frames * BYTES_PER_FRAME

        while len(state.pcm) >= window_bytes:
            window = bytes(state.pcm[:window_bytes])
            del state.pcm[:window_bytes]

            images = self.backend.render(
                state.identity.prepared,
                state.context + window,
                start_frame=state.frames_emitted,
                count=self.window_frames,
            )
            # Keep the tail as context for the next window, so the feature extraction
            # either side of the boundary overlaps and the mouth does not jump.
            if self.context_frames:
                state.context = window[-self.context_frames * BYTES_PER_FRAME :]

            for image in images:
                yield Frame(
                    data=image,
                    epoch=state.epoch,
                    # The mixer restamps this. A renderer's own pts is only useful for
                    # spotting one that has lost count of its own output.
                    pts_ms=state.frames_emitted * self.frame_interval_ms,
                )
                state.frames_emitted += 1

    def reset(self, session: object) -> None:
        """
        Abandon buffered audio and feature context. Weights stay loaded.

        The method barge-in depends on. It does not cancel a `render` call already inside
        the backend -- that pass completes and its frames are returned, then dropped at the
        consumer because their epoch is stale. Cancellation is an integer write, not a
        kill, and this is where that design becomes concrete for a GPU model.
        """
        state = _as_session(session)
        state.pcm.clear()
        state.context = b""
        state.frames_emitted = 0
        state.epoch = IDLE_EPOCH
        state.resets += 1

    def close_session(self, session: object) -> None:
        """
        Release per-session state. Weights are process-scoped and stay put.

        Unloading the model here would make every session pay the cold-start cost §1.4
        argues cannot be paid at conversation start.
        """
        state = _as_session(session)
        state.pcm.clear()
        state.context = b""
        state.closed = True


def _as_session(session: object) -> MuseTalkSession:
    if not isinstance(session, MuseTalkSession):
        raise TypeError(f"expected a MuseTalkSession, got {type(session)!r}")
    if session.closed:
        raise RuntimeError("session is closed")
    return session
