"""
Instrumentation hooks.

In the prototype these print one structured JSON object per event to a sink and
accumulate in-process histograms that `scripts/measure_latency.py` reads back. In
production the same call sites become OTel spans and metrics; the call sites are
the part that matters and the part that is expensive to retrofit, so they go in
now rather than in M5.

Every event carries `epoch`, which is also the per-turn trace correlator: one
conversational turn spans STT, LLM, TTS, render, and transport, and without a
single id threaded through all five, "the avatar felt laggy" is unfalsifiable.

Nothing here imports a metrics library. `Telemetry` is injected into the
orchestrator and the mixer, so tests assert on recorded events instead of parsing
stdout.
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TextIO

if TYPE_CHECKING:
    from avatar.state import State

# Stage names, aligned with the PROCESS.md 1.5 latency budget table so the
# measured column can be filled from a snapshot without a mapping step.
STAGE_TURN_DETECT = "turn_detect"
STAGE_STT = "stt_finalize"
STAGE_LLM_TTFT = "llm_ttft"
STAGE_TTS_FIRST_AUDIO = "tts_first_audio"
STAGE_FIRST_FRAME = "avatar_first_frame"
STAGE_TRANSPORT = "transport"
STAGE_PERCEIVED_TOTAL = "perceived_total"
STAGE_INTERRUPT_TO_SILENT = "interrupt_to_silent"


class Sink(Protocol):
    def write(self, line: str) -> None: ...


Observer = Callable[[Mapping[str, object]], None]
"""
A live subscriber to the event stream.

The browser client is one: it renders state changes, stale-frame drops, and
first-frame latency as they happen rather than on a poll, which is what makes a
barge-in visible as an integer changing rather than as a guess about what the video
did. Observers must not raise -- see `_emit`.
"""


class _StreamSink:
    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def write(self, line: str) -> None:
        self._stream.write(line + "\n")
        self._stream.flush()


class NullSink:
    """Discards events. Used by tests that only care about the histograms."""

    def write(self, line: str) -> None:
        return None


@dataclass
class Histogram:
    """
    Sorted-on-read sample buffer.

    Unbounded, which is fine for a prototype measuring tens of turns and wrong
    for production -- a real deployment reports into a bucketed histogram it
    never has to hold in memory. Noted rather than fixed, because fixing it here
    would mean shipping a metrics backend for no measurement gain.
    """

    name: str
    samples: list[float] = field(default_factory=list)

    def observe(self, value_ms: float) -> None:
        self.samples.append(value_ms)

    def percentile(self, p: float) -> float | None:
        if not self.samples:
            return None
        ordered = sorted(self.samples)
        if len(ordered) == 1:
            return ordered[0]
        # Nearest-rank on a small n. Interpolating would imply a precision that
        # twenty samples do not have.
        rank = math.ceil(p / 100 * len(ordered)) - 1
        return ordered[min(max(rank, 0), len(ordered) - 1)]

    def summary(self) -> dict[str, float | int | None]:
        return {
            "count": len(self.samples),
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "max": max(self.samples) if self.samples else None,
        }


class Telemetry:
    """Records events and latency samples. Cheap enough to call on every frame."""

    def __init__(self, sink: Sink | None = None, *, retain_events: bool = True) -> None:
        self._sink: Sink = sink if sink is not None else _StreamSink(sys.stdout)
        self._observers: list[Observer] = []
        # Retaining every event is what the tests assert against, and it is an
        # unbounded list -- fine for a suite and for a demo session, wrong for a
        # long-lived server. Off by default nowhere, but switchable.
        self._retain_events = retain_events
        self.histograms: dict[str, Histogram] = {}
        self.counters: dict[str, int] = defaultdict(int)
        self.events: list[dict[str, object]] = []

    def subscribe(self, observer: Observer) -> None:
        self._observers.append(observer)

    # -- emit ---------------------------------------------------------------

    def _emit(self, event: str, **fields: object) -> None:
        record: dict[str, object] = {"event": event, **fields}
        if self._retain_events:
            self.events.append(record)
        self._sink.write(json.dumps(record, default=str))
        for observer in self._observers:
            # An observer that raises must not break the thing it is observing.
            # Losing a readout in the browser is a cosmetic failure; taking down a
            # live session because of one is not.
            with contextlib.suppress(Exception):
                observer(record)

    def observe_ms(self, stage: str, value_ms: float, *, epoch: int) -> None:
        hist = self.histograms.setdefault(stage, Histogram(stage))
        hist.observe(value_ms)
        self._emit("latency", stage=stage, ms=round(value_ms, 2), epoch=epoch)

    def increment(self, counter: str, *, amount: int = 1, **labels: object) -> None:
        key = counter if not labels else f"{counter}{sorted(labels.items())}"
        self.counters[key] += amount
        self._emit("counter", counter=counter, amount=amount, **labels)

    # -- named call sites ---------------------------------------------------
    #
    # Named methods rather than raw `observe_ms` calls at the call site, so that
    # renaming a stage is one edit here instead of a grep across the package.

    def state_change(self, old: State, new: State, epoch: int) -> None:
        self._emit("state_change", **{"from": str(old), "to": str(new), "epoch": epoch})

    def first_frame_latency(self, seconds: float, *, epoch: int) -> None:
        self.observe_ms(STAGE_FIRST_FRAME, seconds * 1000, epoch=epoch)

    def interrupt_latency(self, seconds: float, *, epoch: int) -> None:
        self.observe_ms(STAGE_INTERRUPT_TO_SILENT, seconds * 1000, epoch=epoch)

    def turn_latency(self, seconds: float, *, epoch: int) -> None:
        self.observe_ms(STAGE_PERCEIVED_TOTAL, seconds * 1000, epoch=epoch)

    def stale_artifact_dropped(self, kind: str, *, stale_epoch: int, current: int) -> None:
        """
        A frame or audio chunk from a cancelled turn was discarded.

        This is the observable proof that barge-in worked by invalidation rather
        than by luck. M4's acceptance criterion is verified from this event, not
        from the video looking right.
        """
        self.increment("stale_dropped", kind=kind)
        self._emit("stale_dropped", kind=kind, stale_epoch=stale_epoch, current_epoch=current)

    def heard(
        self, text: str, *, epoch: int, transcribed: bool, silent: bool = False
    ) -> None:
        """
        What the transcriber produced for a turn, and whether it produced anything.

        This exists because its absence was a real defect. The transcript is the input to
        the LLM, and it was going into conversation history without ever being logged or
        shown -- so an empty transcript was indistinguishable from a working one. The
        visible symptom was an interviewer that asked a reasonable-sounding question with
        no relation to the answer just given, which reads as "the model is ignoring me"
        when the actual fault is upstream and total: no words reached the model at all.

        `transcribed=False` marks the fallback path, where a turn is known to have
        contained speech but no text came back. That case must be loud, because the
        conversation continues plausibly without it and nothing else reveals the gap.

        `silent=True` is the different case: no speech at all, and the silence watchdog
        opened this turn instead of the candidate. It has to be a separate flag rather than
        an empty `text`, because an empty `text` already means the fallback path above --
        and a quiet candidate must not read as broken transcription.
        """
        self.increment(
            "heard", transcribed=str(transcribed).lower(), silent=str(silent).lower()
        )
        self._emit(
            "heard", text=text, epoch=epoch, transcribed=transcribed, silent=silent
        )

    def said(self, sentence: str, *, epoch: int) -> None:
        """
        One sentence the avatar is about to speak.

        Emitted per sentence rather than per turn, and *before* its audio is synthesised,
        so a reader sees the words in the order they are spoken and sees them for a turn
        that a barge-in later cuts short. Pairing with `heard` gives a complete two-sided
        transcript without the client having to transcribe audio it just played.

        Deliberately not the whole turn's text: waiting for the turn to finish would mean
        an interrupted question never appears at all, and an interrupted question is
        exactly the one worth reading.
        """
        self._emit("said", text=sentence, epoch=epoch)

    def plan_update(self, snapshot: Mapping[str, object], *, epoch: int) -> None:
        """
        Coverage against the competency plan, after a turn has been read into it.

        Emitted as an event rather than only written to the session record because the operator
        watching a live interview is the person who most needs it: knowing that four of six
        competencies are evidenced with two turns left is actionable *during* the call, and
        useless afterwards. The record is for the report; this is for the room.

        No histogram. Coverage is not a duration, and forcing it into `observe_ms` to reuse the
        plumbing would put a meaningless percentile in the latency snapshot.
        """
        self._emit("plan", epoch=epoch, **snapshot)

    def frame_repeated(self, *, total: int) -> None:
        self.increment("frames_repeated")
        self._emit("frame_repeated", total=total)

    def session_failure(self, cause: str, *, epoch: int) -> None:
        self.increment("session_failure", cause=cause)
        self._emit("session_failure", cause=cause, epoch=epoch)

    # -- read back ----------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        return {
            "latency": {name: h.summary() for name, h in sorted(self.histograms.items())},
            "counters": dict(sorted(self.counters.items())),
        }
