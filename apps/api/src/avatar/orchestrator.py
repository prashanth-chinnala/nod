"""
Session orchestration for a real-time conversational avatar.

Design rules this file exists to enforce:

  1. The video track is continuous. It opens at session start and closes at
     session end. State changes swap the *frame source*, never the track.

  2. Cancellation is an integer, not a task kill. Every artifact carries the turn
     epoch it was produced under; stale epochs are dropped at the consumer
     boundary. This makes barge-in race-free and unit-testable.

  3. The ML model sits behind `TalkingHeadRenderer` and knows nothing about
     sessions, turns, VAD, or transport. All lifecycle logic lives here.

  4. Conversation history is truncated to what the candidate actually HEARD, not
     what the LLM generated. See `heard_text`.

Why an epoch and not `task.cancel()`: cancelling a task tells you when the
coroutine stopped, not when its already-dispatched side effects stop arriving. A
GPU forward pass in flight still returns frames; a transport write already awaited
still lands. An integer write happens instantly and is observable from every
consumer, so "was this artifact produced under the current turn?" has one answer
that every component agrees on. The cost is bounded latency, not correctness: an
abandoned turn runs until its next epoch check, which is one audio chunk.

Nothing here imports torch, CUDA, or a renderer implementation. That is a graded
boundary, enforced by `tests/test_boundaries.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from contextlib import aclosing

from avatar.contracts import (
    FRAME_INTERVAL_MS,
    AudioChunk,
    Clock,
    Message,
    SentenceStream,
    SpeechStream,
    TalkingHeadRenderer,
    Transport,
    Turn,
)
from avatar.mixer import FrameMixer
from avatar.state import (
    FRAME_SOURCE,
    InvalidTransition,
    State,
    can_transition,
)
from avatar.telemetry import STAGE_LLM_TTFT, STAGE_TTS_FIRST_AUDIO, Telemetry

RENDER_LEAD_IN_FRAMES = 4
"""
Rendered frames buffered before the mixer switches away from the idle loop.

Trades first-frame latency for stutter resistance: at 25fps this is 160ms of
cushion bought for 160ms of added first-frame delay. Renderers that emit in bursts
rather than steadily need it; a genuinely steady renderer does not. Tune against
`FrameMixer.frames_repeated` and report the value used.
"""

SEAM_WAIT_MAX_MS = 120
"""
How long the handover will wait for a mouth-closed idle frame to cut from.

Unbounded waiting would let a sparsely-annotated idle clip delay speech
indefinitely, which trades a visible artifact for an audible one -- the worse
deal. On expiry the switch happens anyway and `seam_forced` increments, so the
frequency is measurable rather than invisible.
"""

IDLE_REPROMPT_SECONDS = 12.0
"""How long the avatar sits silent before re-prompting the candidate."""

REPROMPT_TRANSCRIPT = "[candidate silent - re-prompt]"

WORDS_PER_MINUTE = 150.0


def estimate_duration_ms(text: str) -> int:
    """
    Rough speaking duration for a span of text.

    A deliberate approximation, and the weakest link in history truncation. The
    production answer is word-level timestamps from the TTS engine, which most
    expose; this stands in until the TTS adapter lands in M4, and the error it
    introduces is a word or two at the truncation point.
    """
    if not text.strip():
        return 0
    words = len(text.split())
    return int(words / WORDS_PER_MINUTE * 60_000)


def heard_text(turn: Turn) -> str:
    """
    The prefix of a turn the candidate actually heard.

    Keyed on `audio_played_ms` -- client-acknowledged playback -- not
    `audio_sent_ms`. The difference is the client's jitter buffer, which a barge-in
    discards. Crediting sent-but-unplayed audio to history would leave the LLM
    believing it asked a question that never reached the candidate, and every
    later turn inherits that error silently.

    Truncation lands on a word boundary. Mid-word cuts survive fine as history
    text, but they read as corruption in a transcript a human reviews later.
    """
    if not turn.interrupted:
        return turn.text_generated
    if not turn.text_generated or turn.audio_played_ms <= 0:
        return ""

    total_ms = estimate_duration_ms(turn.text_generated)
    if total_ms <= 0:
        return ""

    ratio = min(1.0, turn.audio_played_ms / total_ms)
    cutoff = int(len(turn.text_generated) * ratio)
    prefix = turn.text_generated[:cutoff]
    if ratio < 1.0 and not turn.text_generated[cutoff : cutoff + 1].isspace():
        # We cut mid-word; drop the partial.
        head, _, tail = prefix.rpartition(" ")
        prefix = head if head else ("" if tail else prefix)
    return prefix.rstrip()


class SessionOrchestrator:
    """
    Deterministic state machine. Owns every transition.

    The renderer, LLM, TTS, and transport are all things this calls. None of them
    call back into state, and none of them are allowed to know what state the
    session is in. Every collaborator is injected, which is what lets the whole
    machine be exercised in CI with no GPU, no network, and no real clock.
    """

    def __init__(
        self,
        *,
        renderer: TalkingHeadRenderer,
        mixer: FrameMixer,
        transport: Transport,
        llm: SentenceStream,
        tts: SpeechStream,
        telemetry: Telemetry,
        clock: Clock = time.monotonic,
        render_lead_in_frames: int = RENDER_LEAD_IN_FRAMES,
        idle_reprompt_seconds: float = IDLE_REPROMPT_SECONDS,
        seam_wait_max_ms: int = SEAM_WAIT_MAX_MS,
    ) -> None:
        self._renderer = renderer
        self._mixer = mixer
        self._transport = transport
        self._llm = llm
        self._tts = tts
        self._telemetry = telemetry
        self._clock = clock
        self._render_lead_in_frames = render_lead_in_frames
        self._idle_reprompt_seconds = idle_reprompt_seconds
        self._seam_wait_max_ms = seam_wait_max_ms

        self.state = State.INITIALIZING
        self.epoch = 0
        self.turn: Turn | None = None
        self.history: list[Message] = []

        self._render_session: object | None = None
        self._pipeline_task: asyncio.Task[None] | None = None
        self._last_activity = clock()
        self._lead_in_satisfied_at: float | None = None

    @property
    def pipeline_task(self) -> asyncio.Task[None] | None:
        """
        The in-flight turn, if any.

        Exposed so the server can surface a failed turn instead of letting the
        exception disappear into an unretrieved task result, and so tests can wait
        for a turn to settle without reaching into private state.
        """
        return self._pipeline_task

    # -- transitions --------------------------------------------------------

    def _transition(self, new: State) -> None:
        """
        The only place `self.state` changes, and the only place the mixer's frame
        source changes. Both facts are load-bearing: the source is a pure function
        of the state via `FRAME_SOURCE`, so there is no way to reach a state whose
        visual has not been decided.
        """
        if new is self.state:
            return
        if not can_transition(self.state, new):
            raise InvalidTransition(self.state, new)
        old, self.state = self.state, new
        self._mixer.set_source(FRAME_SOURCE[new])
        self._telemetry.state_change(old, new, self.epoch)

    # -- lifecycle ----------------------------------------------------------

    async def start(self, identity_reference: str) -> None:
        """
        Prepare the identity and open the track.

        `prepare_identity` goes to a worker thread, and that is not a micro-optimisation. It is
        synchronous and a real renderer makes it expensive -- measured at 109s for a 150-frame
        reference -- so calling it inline blocked the event loop for the whole enrollment. The
        symptom was not a slow session: it was `TimeoutError: timed out during opening
        handshake`, because the loop could not finish the WebSocket handshake it was in the
        middle of. Every other session in the process stalls with it.

        No torch crosses this line. `asyncio.to_thread` is stdlib, and the renderer is still
        reached only through the protocol -- which is what rule 3 protects.
        """
        identity = await asyncio.to_thread(self._renderer.prepare_identity, identity_reference)

        # A renderer that can show the persona standing by is asked for one. Optional by
        # design: the stub has no face to stand by with, so it does not implement this and
        # the placeholder remains. `getattr` rather than a required protocol method for
        # exactly that reason.
        offer = getattr(self._renderer, "idle_loop", None)
        if callable(offer):
            idle = offer(identity)
            if idle is not None:
                self._mixer.set_idle(idle)
        self._render_session = self._renderer.start_session(identity)
        await self._transport.open_track()
        self._last_activity = self._clock()
        self._transition(State.IDLE)

    async def close(self) -> None:
        if self.state is State.CLOSED:
            return
        # Invalidate everything in flight before tearing anything down, so a
        # late-arriving frame cannot touch a closed render session.
        self.epoch += 1
        if self._pipeline_task is not None:
            self._pipeline_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._pipeline_task
            self._pipeline_task = None
        if self._render_session is not None:
            self._renderer.close_session(self._render_session)
            self._render_session = None
        await self._transport.close_track()
        self._transition(State.CLOSED)

    # -- inbound audio events -----------------------------------------------

    async def on_speech_start(self) -> None:
        """VAD detected the candidate speaking."""
        self._last_activity = self._clock()
        if self.state in (State.SPEAKING, State.THINKING):
            await self._cancel_turn()
        elif self.state is State.IDLE:
            self._transition(State.LISTENING)

    async def on_speech_retract(self) -> None:
        """VAD false positive -- background noise, not speech."""
        if self.state is State.LISTENING:
            self._transition(State.IDLE)

    async def on_end_of_turn(self, transcript: str) -> bool:
        """
        Accept the candidate's turn, or refuse it. Returns whether it was accepted.

        The guard stays: an end-of-turn from anywhere but LISTENING is not a turn. What
        changed is that refusal is now *reported* rather than silent, and the reason is a
        defect this cost a debugging session to find.

        The caller emits `heard` telemetry for the transcript. When this returned `None`
        unconditionally, a refused turn still produced a `heard` event -- so the transcript
        pane and the session record both showed the candidate saying something the model
        never received, and the interviewer's next question looked like it was ignoring them.
        That is the exact failure class `heard` was introduced to make visible, and it was
        being caused by the instrumentation itself. A caller can now tell the difference.
        """
        if self.state is not State.LISTENING:
            return False
        self.history.append({"role": "user", "content": transcript})
        self._begin_turn()
        return True

    async def on_idle_tick(self) -> None:
        """
        Silence watchdog. Must be driven from outside; the orchestrator does not
        own a timer of its own, so that tests can advance time explicitly.
        """
        if self.state is not State.IDLE:
            return
        if self._clock() - self._last_activity <= self._idle_reprompt_seconds:
            return
        # Deliberately not routed through `on_end_of_turn`: that guard requires
        # LISTENING, and a silence re-prompt by definition happens from IDLE. The
        # earlier sketch chained the two and the re-prompt could never fire.
        self.history.append({"role": "user", "content": REPROMPT_TRANSCRIPT})
        self._begin_turn()

    def on_audio_played(self, duration_ms: int, epoch: int) -> None:
        """
        Client acknowledged playing `duration_ms` of audio from turn `epoch`.

        This is the only input that moves `audio_played_ms`, and therefore the
        only evidence used for history truncation. Acks for a turn that has
        already been abandoned are dropped -- they describe audio the client threw
        away when it flushed.
        """
        turn = self.turn
        if turn is None or turn.epoch != epoch:
            self._telemetry.stale_artifact_dropped(
                "audio_ack", stale_epoch=epoch, current=self.epoch
            )
            return
        turn.audio_played_ms += duration_ms

    def on_first_paint(self, epoch: int) -> None:
        """
        Client painted the first frame of turn `epoch`.

        This closes the only measurement the server cannot take for itself. A
        timestamp at ingress and a timestamp at browser paint are very different
        numbers: the gap holds encode, socket, decode, and a rendering frame, and
        every one of those is a term the latency budget has to account for.
        Reporting `avatar_first_frame` as if it were the perceived total would
        understate the truth by exactly the amount that is hardest to fix.

        Recorded once per turn. A second report for the same turn is ignored rather
        than averaged -- there is only one first frame.
        """
        turn = self.turn
        if turn is None or turn.epoch != epoch or turn.first_paint_at is not None:
            return
        now = self._clock()
        turn.first_paint_at = now
        self._telemetry.turn_latency(now - turn.started_at, epoch=epoch)

    # -- the generate/speak pipeline ----------------------------------------

    def _begin_turn(self) -> None:
        """
        Start a turn.

        The epoch is bumped here, synchronously, before the task exists. The
        earlier sketch bumped it inside the task body, which left a window between
        `create_task` and first execution in which a barge-in bumped the epoch and
        the task then bumped past it and carried on generating a turn that had
        already been cancelled.
        """
        self.epoch += 1
        my_epoch = self.epoch
        self.turn = Turn(epoch=my_epoch, started_at=self._clock())
        self._lead_in_satisfied_at = None
        self._transition(State.THINKING)
        self._pipeline_task = asyncio.create_task(self._run_turn(my_epoch))

    async def _run_turn(self, my_epoch: int) -> None:
        turn = self.turn
        assert turn is not None and turn.epoch == my_epoch

        try:
            llm_started = self._clock()
            saw_sentence = False

            # `aclosing` is load-bearing, not tidiness. Returning out of an `async for`
            # does not close the generator -- Python leaves that to the garbage
            # collector, which may be seconds later or never. For a real LLM or TTS
            # backed by an HTTP stream, that means a barge-in abandons the turn
            # logically while the provider keeps generating, and billing, a response
            # nobody will ever hear. Closing deterministically runs the generator's
            # `finally`, which aborts the request.
            async with aclosing(self._llm(list(self.history))) as sentences:
                async for sentence in sentences:
                    if my_epoch != self.epoch:
                        return
                    if not saw_sentence:
                        saw_sentence = True
                        self._telemetry.observe_ms(
                            STAGE_LLM_TTFT,
                            (self._clock() - llm_started) * 1000,
                            epoch=my_epoch,
                        )
                    turn.text_generated += sentence
                    # Before synthesis, so the words appear even if a barge-in cuts this
                    # sentence off mid-air. An interrupted question is the one most worth
                    # being able to read back.
                    self._telemetry.said(sentence, epoch=my_epoch)

                    tts_started = self._clock()
                    saw_audio = False
                    async with aclosing(self._tts(sentence, my_epoch)) as chunks:
                        async for chunk in chunks:
                            if my_epoch != self.epoch:
                                self._telemetry.stale_artifact_dropped(
                                    "audio", stale_epoch=my_epoch, current=self.epoch
                                )
                                return
                            if not saw_audio:
                                saw_audio = True
                                self._telemetry.observe_ms(
                                    STAGE_TTS_FIRST_AUDIO,
                                    (self._clock() - tts_started) * 1000,
                                    epoch=my_epoch,
                                )
                            await self._speak(turn, chunk, my_epoch)

            if my_epoch == self.epoch:
                await self._finish_turn(turn)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Never leave the session wedged mid-turn. A failed turn is recoverable;
            # a session stuck in THINKING forever is not.
            self._telemetry.session_failure(type(exc).__name__, epoch=my_epoch)
            if my_epoch == self.epoch:
                self._transition(State.IDLE)
                self._last_activity = self._clock()
            raise

    async def _speak(self, turn: Turn, chunk: AudioChunk, my_epoch: int) -> None:
        self._renderer.push_audio(self._render_session, chunk)
        await self._transport.send_audio(chunk)
        turn.audio_sent_ms += chunk.duration_ms
        self._pump_frames(my_epoch)
        self._maybe_start_speaking(turn, my_epoch)

    def _pump_frames(self, my_epoch: int) -> None:
        """
        Drain whatever the renderer has ready.

        Synchronous and non-blocking by contract: `frames()` returns what exists
        now. A renderer that blocks here would stall the audio path, which is the
        one thing that must not be starved.
        """
        assert self._render_session is not None
        for frame in self._renderer.frames(self._render_session):
            self._mixer.offer(frame, my_epoch)

    def _maybe_start_speaking(self, turn: Turn, my_epoch: int) -> None:
        if self.state is not State.THINKING:
            return
        if self._mixer.buffered() < self._render_lead_in_frames:
            return

        now = self._clock()
        if self._lead_in_satisfied_at is None:
            self._lead_in_satisfied_at = now

        waited_ms = (now - self._lead_in_satisfied_at) * 1000
        if not self._mixer.at_clean_exit():
            if waited_ms < self._seam_wait_max_ms:
                return
            self._telemetry.increment("seam_forced")

        self._transition(State.SPEAKING)
        turn.first_frame_at = now
        self._telemetry.first_frame_latency(now - turn.started_at, epoch=my_epoch)
        # The perceived total is deliberately NOT emitted here. It is the same
        # instant under a more flattering name, and reporting it as end-to-end would
        # silently drop encode, socket, decode, and paint from the budget. It is
        # emitted from `on_first_paint`, when the client says it actually saw it.

    async def _finish_turn(self, turn: Turn) -> None:
        self.history.append({"role": "assistant", "content": turn.text_generated})
        self._transition(State.IDLE)
        self._last_activity = self._clock()

    # -- barge-in -----------------------------------------------------------

    async def _cancel_turn(self) -> None:
        """
        The entire cancellation.

        One integer bump invalidates every in-flight artifact; nothing downstream
        can act on a stale epoch after that line. Everything after it is cleanup
        that is safe to do at leisure.
        """
        started = self._clock()
        self._transition(State.CANCELLING)

        self.epoch += 1  # <-- the actual cancellation

        turn = self.turn
        if turn is not None:
            turn.interrupted = True
            heard = heard_text(turn)
            if heard:
                self.history.append({"role": "assistant", "content": heard + " [interrupted]"})
            # If nothing was heard, nothing is recorded. From the candidate's side
            # the avatar never spoke, and history should agree with them.

        if self._render_session is not None:
            self._renderer.reset(self._render_session)
        await self._transport.flush_audio()

        self._transition(State.LISTENING)
        self._last_activity = self._clock()
        self._telemetry.interrupt_latency(self._clock() - started, epoch=self.epoch)

    # -- introspection for the demo page and measurement script -------------

    def stats(self) -> dict[str, object]:
        return {
            "state": str(self.state),
            "epoch": self.epoch,
            "frames_emitted": self._mixer.frames_emitted,
            "frames_repeated": self._mixer.frames_repeated,
            "frames_discarded": self._mixer.frames_discarded,
            "buffer_depth": self._mixer.buffered(),
            "frame_interval_ms": FRAME_INTERVAL_MS,
            "history_len": len(self.history),
        }
