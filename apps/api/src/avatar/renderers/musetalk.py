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

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from avatar.contracts import (
    FRAME_INTERVAL_MS as CONTRACT_FRAME_INTERVAL_MS,
)
from avatar.contracts import (
    IDLE_EPOCH,
    AudioChunk,
    Frame,
)
from avatar.contracts import (
    TARGET_FPS as CONTRACT_TARGET_FPS,
)
from avatar.mixer import IdleLoop

TARGET_FPS = CONTRACT_TARGET_FPS
"""
Frames per second the renderer aims to produce. `AVATAR_MUSETALK_FPS` overrides.

**Why this is configurable, and why it is not a quality setting.** 25 is what MuseTalk's
reference cycle and Whisper's feature stride assume, and it is the right target. But a renderer
that cannot reach its target does not degrade gracefully -- it fails completely. The mixer drops
any frame that misses its slot, so at 3.3 fps against a 25fps clock *every* frame is late and
*every* frame is discarded: measured on the M1 Pro, three turns produced 169 rendered frames and
delivered zero. The candidate watched the placeholder while the interviewer talked.

Rendering at a rate the hardware can sustain is the difference between choppy video and no
video. It is not free -- lip motion at 8fps is visibly less smooth -- but it is a real
picture of a real face, which the alternative is not.

This value has to reach three places that must agree: the mixer's cadence, the frame interval
stamped on each frame, and the `fps=` argument Whisper's chunker slices audio features with. If
they disagree, the mouth drifts against the speech -- slowly, so it reads as bad dubbing rather
than as a bug. So it is derived from one number here rather than written down three times.
"""

FRAME_INTERVAL_MS = CONTRACT_FRAME_INTERVAL_MS
"""Milliseconds per frame. Re-exported so this module's constants read as a set."""

SAMPLE_RATE = 16_000
"""
Fixed by two things at once, which is a convenience rather than a coincidence.

`whisper-tiny` expects 16kHz, and the transport already carries 16kHz mono PCM16 because
the voice-activity detector wants the same. So no resampling happens anywhere in the audio
path, and a resampler is one fewer thing to get subtly wrong.
"""

BYTES_PER_SAMPLE = 2
BYTES_PER_FRAME = SAMPLE_RATE * BYTES_PER_SAMPLE * FRAME_INTERVAL_MS // 1000  # 1280

WINDOW_MS = 640
"""
How much audio to accumulate before rendering a batch, in milliseconds.


**Milliseconds, not frames, and that distinction was a real bug.** This was `WINDOW_FRAMES =
16`, which is 640ms at 25fps -- correct. But a frame count means the window's *duration* moves
with the frame rate, so dropping to 8fps to match what the hardware could sustain silently
stretched it to 2000ms. Nothing looked wrong: the renderer produced correct frames at a
sustainable rate, and the first one still arrived 4.2 seconds into the turn, by which time the
avatar's audio had nearly finished playing and `SPEAKING -> IDLE` drained the queue. 16 frames
rendered, 16 discarded, zero shown. The window is a latency budget, and a latency budget
denominated in frames is not one.


The floor is Whisper needing context either side of a frame to encode it; the ceiling is
latency, since no frame emits until its window is full. 640ms is upstream's, and it is the whole
of the first-frame cost that this stage controls.
"""

WINDOW_FRAMES = max(1, round(WINDOW_MS / FRAME_INTERVAL_MS))
"""The window in frames, derived. 16 at 25fps, 5 at 8fps -- 640ms either way."""

CONTEXT_MS = 80
"""
Already-consumed audio prepended to each window, in milliseconds. Same reasoning as `WINDOW_MS`.


Without it, every window boundary is a discontinuity in the audio features and the mouth visibly
jumps at the window rate -- the characteristic streaming-talking-head artifact, and periodic
enough to be unmistakable once seen.
"""

CONTEXT_FRAMES = max(1, round(CONTEXT_MS / FRAME_INTERVAL_MS))
"""Context in frames, derived. 2 at 25fps, 1 at 8fps."""


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

    frame_count: int | None = None
    """
    Usable reference frames -- the ones a face was found in, before cycling.

    Reported here rather than counted by the API, which would mean decoding the clip a second
    time to learn something the renderer already knows. `None` from a backend that does not
    report it, so the console shows an empty cell rather than a number nothing measured.

    It is the *usable* count, not the source count: a frame with no detected face is dropped
    during preparation, so a 150-frame clip where the subject turns away can legitimately
    enroll fewer. The difference is worth seeing.
    """


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


IDENTITY_CACHE_SIZE = 2
"""
Prepared identities held in memory at once. `AVATAR_IDENTITY_CACHE` overrides.

Two, not more: each one is roughly a gigabyte for a 150-frame reference, and this process also
holds several GB of model weights. One would thrash whenever two agents alternate.
"""

_IDENTITIES: dict[str, MuseTalkIdentity] = {}
"""
Prepared identities, shared by every renderer instance in the process.

Module-level rather than per-instance, and the reason is specific: `renderers.build`
constructs a fresh renderer for each caller, so `POST /faces/{id}/prepare` and a session that
then uses that face were separate objects with separate caches. Enrollment ran, measured 109s,
cached its result -- and the session it was meant to serve threw the artifact away and did it
again. Enrollment that does not warm the thing it enrolls for is a status field, not a feature.

The cost of sharing, stated, because it bit immediately: this is process-global, so a test that
prepares the same reference twice sees one call. `conftest.py` clears it around every test via
`reset_identity_cache()`, and nothing in the runtime should call that.
"""


def reset_identity_cache() -> None:
    """Drop every prepared identity. For tests, and for freeing memory deliberately."""
    _IDENTITIES.clear()


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
        width: int = 0,
        height: int = 0,
        first_frame_delay_ms: int = 0,
    ) -> None:
        """
        `width`, `height` and `first_frame_delay_ms` are accepted and not honoured, which needs
        saying out loud rather than being discovered.

        They exist because `renderers.build` passes one options dict to whichever renderer is
        selected, and until now that dict was shaped entirely by the stub -- so choosing
        `musetalk` raised `TypeError: unexpected keyword argument 'width'` at the moment a
        candidate opened their interview. The class docstring claimed conformance was asserted
        so this "cannot quietly take different arguments"; the conformance test checks methods,
        not the constructor, so it did.

        Why not honoured:

        * `first_frame_delay_ms` is how long the *stub* waits before its first frame, to
          simulate a cost this renderer actually pays. Simulating it here would add latency on
          top of the real thing.
        * `width`/`height` default to 256x144 -- deliberately tiny, deliberately 16:9, because
          they describe a placeholder canvas. A reference clip of a person is portrait, and
          forcing a face into 16:9 would stretch it. Output is scaled by height alone, aspect
          preserved, capped by `AVATAR_MUSETALK_MAX_HEIGHT`.

        The residual, stated because it is visible: the idle loop is still the 16:9 placeholder,
        so switching between idle and speaking changes aspect ratio on screen. The right fix is
        an idle loop built from the reference frames -- the reference *is* the person sitting
        still, which is exactly what an idle loop should be -- and that is a change to how the
        mixer is constructed, not to this class.
        """
        if window_frames < 1:
            raise ValueError(f"window_frames must be positive, got {window_frames}")
        if context_frames < 0:
            raise ValueError(f"context_frames must not be negative, got {context_frames}")
        self.window_frames = window_frames
        self.context_frames = context_frames
        self.frame_interval_ms = frame_interval_ms
        self.placeholder_geometry = (width, height)
        self._identity_cache_size = int(
            os.environ.get("AVATAR_IDENTITY_CACHE", IDENTITY_CACHE_SIZE)
        )
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
        """
        Prepare a reference, or return the artifact prepared from it earlier.

        The cache is what makes §1.2's claim -- "reusable across sessions" -- true rather than
        aspirational. Without it every session re-ran preparation from scratch: 109s of face
        detection and VAE encoding per candidate, for a result identical to the one the previous
        candidate's session had just computed and thrown away. Enrollment existed, was measured,
        and bought nothing.

        Keyed by reference path, because that is what the artifact is a function of. Not by face
        id: two faces pointing at the same clip should share, and a face whose clip is replaced
        must not.

        Shared across every renderer instance in the process -- see `_IDENTITIES`. Deliberately
        tiny: one identity for a 150-frame 576x768 reference holds roughly a gigabyte of cycled
        frames, masks and latents, so this trades memory for time at a rate where two entries is
        already a real cost. A persistent cache would mean writing that gigabyte somewhere and
        inventing an invalidation rule for it; a restart re-prepares instead, which is honest,
        because the artifact is derived data.
        """
        cached = _IDENTITIES.get(reference_path)
        if cached is not None:
            return cached

        prepared = self.backend.prepare(reference_path)
        usable = prepared.get("usable_frames") if isinstance(prepared, dict) else None
        identity = MuseTalkIdentity(
            reference_path=reference_path,
            prepared=prepared,
            frame_count=usable if isinstance(usable, int) else None,
        )
        # Evict the oldest rather than growing. `dict` preserves insertion order, so this is a
        # FIFO -- not an LRU, because with a capacity of two the distinction is theoretical and
        # the tracking is not free.
        while len(_IDENTITIES) >= self._identity_cache_size:
            _IDENTITIES.pop(next(iter(_IDENTITIES)))
        _IDENTITIES[reference_path] = identity
        return identity

    def load(self) -> None:
        """
        Load the models without preparing anything.

        Exists for warm-up. Loading is otherwise triggered lazily by the first
        `prepare_identity`, with two consequences worth removing: the load's cost is
        attributed to that face rather than reported separately, and an installation with no
        face attached to any agent warms nothing -- so the first candidate still pays 27s for
        weights even though there was nothing to enroll.

        Idempotent; the backend returns immediately if it is already loaded.
        """
        self.backend.load()
        self._loaded = True

    def idle_loop(self, identity: object) -> IdleLoop | None:
        """
        An idle loop built from this persona's own reference frames, or None.


        **Why the renderer supplies this, not the server.** Between turns the mixer shows
        `placeholder_idle_loop` -- a 256x144 grey rectangle with two eyes. So a candidate saw a
        placeholder, then a real face when the interviewer spoke, then the placeholder again:
        the
        persona appeared only while talking, and the canvas changed aspect ratio each way, which
        reads as the product switching between two different things.


        The reference clip already *is* the person sitting still and looking ahead -- the upload
        guidance asks for exactly that -- so it is the correct idle loop, and nothing has to be
        generated. Between turns the candidate now sees the same face at the same size, not
        speaking. Which is what standing by looks like.


        Returns None when the artifact has no encoded frames, so an older prepared identity or a
        different backend degrades to the placeholder rather than failing.
        
        """
        if not isinstance(identity, MuseTalkIdentity):
            return None
        prepared = identity.prepared
        if not isinstance(prepared, dict):
            return None
        frames = prepared.get("idle_jpegs")
        closed = prepared.get("idle_mouth_closed")
        if not frames or not closed:
            return None
        return IdleLoop(frames, closed)

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
