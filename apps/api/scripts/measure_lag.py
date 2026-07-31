#!/usr/bin/env python3
"""
Measure how far the video trails the audio in a live turn, and where the gap comes from.

**Why this script exists rather than another reading of the throughput benchmark.**
`bench_renderer.py` answers "how fast can this card turn audio into frames" and it now answers
78.4 ms/frame. That is not the number a candidate experiences. What they experience is a mouth
still moving after the sentence finished, and that has three separate causes the throughput
figure cannot separate:

  1. **Start-up.** The first frame of a turn cannot exist before one render window of audio
     does, so a window is a floor on how late video begins.
  2. **Throughput.** If the renderer produces fewer frames per second than the mixer emits, the
     gap grows for the whole turn instead of staying constant.
  3. **Drain.** When the audio ends, frames already queued still have to go somewhere. `offer()`
     discards those rather than showing them, which is correct -- a mouth moving in silence is
     worse than no mouth -- but the count is a direct measure of how much video was behind.

Two earlier attempts at this failed on harness plumbing, so this one deliberately reuses
`smoke_session.py`'s socket, mic and observer rather than reimplementing them. The only thing
added is a clock on each arrival.

**What it reports and what it cannot.** Arrival timestamps are taken in this process, so they
include the socket but not a browser's decode or compositor. That makes every figure a lower
bound on what a person sees, which is the safe direction for a number used to decide whether to
spend money on a GPU. `first_paint` is acknowledged the way the browser does it, so the server's
own `avatar_first_frame` is reported alongside and the two can be compared.

    uvicorn avatar.server:app &
    python scripts/measure_lag.py --turns 3 --json lag.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import websockets
except ModuleNotFoundError:  # pragma: no cover - operator error, not a code path
    sys.exit("needs the server extras: pip install -e '.[dev,server]'")

from smoke_session import (  # noqa: E402  (path is set above)
    MIC_SILENCE_FRAMES,
    MIC_SPEECH_FRAMES,
    URL,
    speak_into_the_mic,
)

from avatar.contracts import FRAME_INTERVAL_MS, TARGET_FPS  # noqa: E402
from avatar.transport.websocket import Kind, decode  # noqa: E402


@dataclass
class Arrival:
    """One artifact off the wire, with the moment it got here."""

    at: float
    epoch: int
    pts_ms: int
    audio_ms: float = 0.0


@dataclass
class Cadence:
    """
    The server's frame cadence, taken from `hello`.

    Read from the server rather than from this process's `AVATAR_FPS`, because the probe and the
    server are different processes and the probe's environment says nothing about the server's. The
    first version imported `TARGET_FPS` locally and reported "delivered 8.4 fps against 25" at
    a server configured for 8 -- a fabricated target, and the `need` column was wrong by the
    same factor. `hello` has carried both numbers all along.
    """

    fps: int = TARGET_FPS
    interval_ms: float = FRAME_INTERVAL_MS
    from_server: bool = False


@dataclass
class Turn:
    """Everything observed for one epoch."""

    epoch: int
    video: list[Arrival] = field(default_factory=list)
    audio: list[Arrival] = field(default_factory=list)
    first_frame_ms: float | None = None
    discarded: int = 0

    def report(self, cadence: Cadence) -> dict[str, object]:
        """
        The three causes, separated.

        `trailing_gap_ms` is the headline: how long video kept arriving after the last audio
        did. Positive means the candidate saw a mouth moving in silence, or would have if those
        frames were not discarded. Negative means video finished first, the healthy direction.
        """
        if not self.video or not self.audio:
            return {"epoch": self.epoch, "usable": False}

        audio_first, audio_last = self.audio[0].at, self.audio[-1].at
        video_first, video_last = self.video[0].at, self.video[-1].at
        spoken_ms = sum(a.audio_ms for a in self.audio)
        wall_ms = (video_last - video_first) * 1000

        return {
            "epoch": self.epoch,
            "usable": True,
            "audio_chunks": len(self.audio),
            "video_frames": len(self.video),
            "frames_discarded": self.discarded,
            "spoken_ms": round(spoken_ms),
            # Video that arrived after the audio stopped. The number the whole script is for.
            "trailing_gap_ms": round((video_last - audio_last) * 1000),
            # Cause 1: how much later than the first audio the first frame appeared.
            "video_start_lag_ms": round((video_first - audio_first) * 1000),
            # Cause 2: frames per second actually delivered across the turn, against the target.
            # A ratio below 1.0 means the gap grew for the whole turn rather than staying put.
            "delivered_fps": round(len(self.video) / max(wall_ms / 1000, 1e-9), 1),
            "fps_target": cadence.fps,
            "server_first_frame_ms": self.first_frame_ms,
            # Frames needed to cover the speech, against frames that arrived. A shortfall is
            # video the mixer had to repeat or the turn simply never got.
            "frames_for_the_speech": round(spoken_ms / cadence.interval_ms),
        }


async def observe(
    socket: object,
    turns: dict[int, Turn],
    stop: asyncio.Event,
    deadline: float,
    cadence: Cadence,
) -> None:
    """
    Consume everything the server sends, stamping arrivals.

    Deliberately a near-copy of `smoke_session.observe` rather than an import: that one exists
    to assert, and bolting a second purpose onto it would make the assertions harder to read.
    The one behaviour that must not diverge is the acknowledgements -- without `audio_played`
    the server has no evidence anything was heard, and without `first_paint` there is no
    `avatar_first_frame`.
    """
    sample_rate = 16_000
    while not stop.is_set() and time.monotonic() < deadline:
        try:
            message = await asyncio.wait_for(socket.recv(), timeout=0.25)  # type: ignore[attr-defined]
        except TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosed:
            return

        now = time.monotonic()
        if isinstance(message, bytes):
            envelope = decode(message)
            turn = turns.setdefault(envelope.epoch, Turn(epoch=envelope.epoch))
            if envelope.kind is Kind.VIDEO:
                turn.video.append(Arrival(at=now, epoch=envelope.epoch, pts_ms=envelope.pts_ms))
                if len(turn.video) == 1 and envelope.epoch:
                    await socket.send(  # type: ignore[attr-defined]
                        json.dumps({"type": "first_paint", "epoch": envelope.epoch})
                    )
            else:
                chunk_ms = len(envelope.payload) / 2 / sample_rate * 1000
                turn.audio.append(
                    Arrival(
                        at=now,
                        epoch=envelope.epoch,
                        pts_ms=envelope.pts_ms,
                        audio_ms=chunk_ms,
                    )
                )
                await socket.send(  # type: ignore[attr-defined]
                    json.dumps(
                        {
                            "type": "audio_played",
                            "ms": round(chunk_ms),
                            "epoch": envelope.epoch,
                        }
                    )
                )
            continue

        payload = json.loads(message)
        kind = payload.get("type")
        if kind == "hello":
            sample_rate = int(payload.get("sample_rate", 16_000))
            if "target_fps" in payload:
                cadence.fps = int(payload["target_fps"])
                cadence.interval_ms = float(
                    payload.get("frame_interval_ms", 1000 / max(cadence.fps, 1))
                )
                cadence.from_server = True
        elif kind == "stats":
            # Session-wide rather than per-turn, so it is attributed to whichever turn is open.
            # Recorded as a delta so the last turn does not inherit every earlier discard.
            total = int(payload["frames_discarded"])
            already = sum(t.discarded for t in turns.values())
            if turns and total > already:
                latest = turns[max(turns)]
                latest.discarded += total - already
        elif (
            kind == "event"
            and payload.get("event") == "latency"
            and str(payload["stage"]) == "avatar_first_frame"
        ):
            epoch = int(payload.get("epoch", 0))
            turns.setdefault(epoch, Turn(epoch=epoch)).first_frame_ms = float(payload["ms"])


async def speak_turns(socket: object, count: int, spacing: float) -> None:
    """
    Speak `count` turns, `spacing` seconds apart, on a background task.

    A background task and a fixed spacing, rather than waiting for each turn to go quiet. Two
    earlier versions of this script waited for quiet and measured nothing: the idle loop never
    goes quiet -- it delivers a frame every `FRAME_INTERVAL_MS` for the whole session -- so
    "nothing has arrived recently" is never true and the wait either ran to its timeout or,
    worse, looked like a settled turn when it was a stalled one. Spacing has to exceed the
    slowest turn observed (LLM ~4.5 s, then speech), or a turn is cut off by the next one and
    its numbers are a barge-in rather than a turn.
    """
    for _ in range(count):
        await speak_into_the_mic(socket, MIC_SPEECH_FRAMES, MIC_SILENCE_FRAMES)
        await asyncio.sleep(spacing)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=3, help="how many turns to speak")
    parser.add_argument("--url", default=URL)
    parser.add_argument(
        "--agent",
        help=(
            "agent id to interview as. Required with the real renderer: without a session the "
            "server falls back to a bundled reference that a GPU renderer has no file for, and "
            "the socket closes mid-handshake with the reason only in the server log."
        ),
    )
    parser.add_argument(
        "--api",
        default="http://127.0.0.1:8000",
        help="where to create the session, if --agent is given",
    )
    parser.add_argument("--json", help="also write the full result here")
    parser.add_argument(
        "--settle",
        type=float,
        default=6.0,
        help="extra seconds to keep listening after the last turn was spoken",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=28.0,
        help=(
            "seconds between turns. Must exceed the slowest turn or a turn is cut off by the "
            "next one and its figures describe a barge-in"
        ),
    )
    args = parser.parse_args()

    turns: dict[int, Turn] = {}
    stop = asyncio.Event()
    cadence = Cadence()

    # A session, so the face under measurement is the one an agent actually uses. The lag
    # depends on the reference -- a 550-frame clip and a 100-frame one are different amounts of
    # work per window -- so measuring against a default would measure the wrong face.
    url, start = args.url, {"type": "start", "reference": "reference.mp4"}
    if args.agent:
        import urllib.request

        request = urllib.request.Request(
            f"{args.api}/sessions",
            data=json.dumps({"agent_id": args.agent}).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            session_id = json.load(response)["id"]
        print(f"session {session_id} as {args.agent}", flush=True)
        url = f"{args.url}?session={session_id}"
        # No `start`: a session id means the server starts on its own, and sending one is
        # answered with `unknown message 'start'`.
        start = {}

    async with websockets.connect(url, max_size=None) as socket:
        if start:
            await socket.send(json.dumps(start))
        speaker = asyncio.ensure_future(speak_turns(socket, args.turns, args.spacing))
        # The receive loop is the main loop, not a task, so the run cannot end while frames are
        # still arriving -- which is what made the earlier version report zero usable turns.
        deadline = time.monotonic() + args.turns * args.spacing + args.settle
        try:
            await observe(socket, turns, stop, deadline, cadence)
        finally:
            speaker.cancel()
            stop.set()

    # Epoch 0 is the idle loop, which belongs to no turn and would drag every average toward
    # whatever the standing-by animation does.
    measured = [turn.report(cadence) for epoch, turn in sorted(turns.items()) if epoch]
    usable = [r for r in measured if r["usable"]]

    source = "from the server" if cadence.from_server else "ASSUMED -- no hello seen"
    print(f"\n=== {len(usable)} turn(s) measured, target {cadence.fps} fps ({source}) ===\n")
    header = (
        f"{'epoch':>6} {'spoken':>8} {'frames':>7} {'need':>6} {'fps':>6} "
        f"{'start lag':>10} {'TRAILING':>10} {'discarded':>10}"
    )
    print(header)
    for r in usable:
        print(
            f"{r['epoch']:>6} {r['spoken_ms']:>7}ms {r['video_frames']:>7} "
            f"{r['frames_for_the_speech']:>6} {r['delivered_fps']:>6} "
            f"{r['video_start_lag_ms']:>9}ms {r['trailing_gap_ms']:>9}ms "
            f"{r['frames_discarded']:>10}"
        )

    result: dict[str, object] = {
        "turns": measured,
        "fps_target": cadence.fps,
        "fps_target_from_server": cadence.from_server,
    }
    if usable:
        trailing = [float(r["trailing_gap_ms"]) for r in usable]  # type: ignore[arg-type]
        fps = [float(r["delivered_fps"]) for r in usable]  # type: ignore[arg-type]
        start = [float(r["video_start_lag_ms"]) for r in usable]  # type: ignore[arg-type]
        result["summary"] = {
            "trailing_gap_ms_median": round(statistics.median(trailing)),
            "trailing_gap_ms_worst": round(max(trailing)),
            "delivered_fps_median": round(statistics.median(fps), 1),
            "video_start_lag_ms_median": round(statistics.median(start)),
        }
        summary = result["summary"]
        assert isinstance(summary, dict)
        print(
            f"\nmedian trailing gap {summary['trailing_gap_ms_median']} ms "
            f"(worst {summary['trailing_gap_ms_worst']}), "
            f"delivered {summary['delivered_fps_median']} fps against {cadence.fps}, "
            f"video started {summary['video_start_lag_ms_median']} ms after audio"
        )
        # Stated as a verdict, because "1900 ms" means nothing without a threshold. Lip-sync
        # tolerance in broadcast is about 100 ms of video lag before it is noticeable.
        gap = float(summary["trailing_gap_ms_median"])
        if gap <= 100:
            print("      -> within the ~100 ms a viewer does not notice.")
        else:
            print(f"      -> {gap / 100:.0f}x the ~100 ms a viewer does not notice.")
    else:
        print("!! no usable turn: no turn produced both audio and video", file=sys.stderr)

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.json}")
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
