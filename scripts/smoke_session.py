#!/usr/bin/env python3
"""
Drive a real session over a real WebSocket and assert the mechanics happened.

Needs the server extras and a running server:

    pip install -e ".[dev,server]"
    uvicorn avatar.server:app &
    python scripts/smoke_session.py

Why this exists separately from the test suite: the suite proves the state machine
is correct against test doubles, which is the right thing for CI to run. It cannot
prove that the bytes on the wire are the bytes the browser expects, that the frame
pump keeps cadence under a real event loop, or that a barge-in reaches the client as
a flush. This does, headlessly, so the claim does not rest on someone watching a
video and forming an impression.

It is not in CI: that would mean a web stack in the CI dependency set to test a layer
whose logic is already covered. Run it before recording the Loom.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

try:
    import websockets
except ModuleNotFoundError:  # pragma: no cover - operator error, not a code path
    sys.exit("needs the server extras: pip install -e '.[dev,server]'")

from avatar.mixer import TARGET_FPS
from avatar.transport.websocket import Kind, decode

URL = "ws://127.0.0.1:8000/session"

SPEAK_SETTLE_SECONDS = 2.0
"""Long enough for TTFT + first audio + the lead-in buffer, with room to spare."""

BARGE_IN_SETTLE_SECONDS = 1.0


@dataclass
class Observed:
    states: list[str] = field(default_factory=list)
    events: Counter[str] = field(default_factory=Counter)
    video_frames: int = 0
    audio_chunks: int = 0
    audio_ms: int = 0
    frame_pts: list[int] = field(default_factory=list)
    stale_kinds: Counter[str] = field(default_factory=Counter)
    latencies: dict[str, float] = field(default_factory=dict)
    flushes: int = 0
    hello: dict[str, object] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    painted_epoch: int = 0
    frames_repeated: int = 0
    frames_discarded: int = 0


async def observe(socket: object, seen: Observed, stop: asyncio.Event) -> None:
    """Consume everything the server sends until told to stop."""
    while not stop.is_set():
        try:
            message = await asyncio.wait_for(socket.recv(), timeout=0.25)  # type: ignore[attr-defined]
        except TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosed:
            return

        if isinstance(message, bytes):
            envelope = decode(message)
            if envelope.kind is Kind.VIDEO:
                seen.video_frames += 1
                seen.frame_pts.append(envelope.pts_ms)
                # Stand in for the browser's paint report, which is what closes the
                # end-to-end measurement. Epoch 0 is the idle loop and belongs to no
                # turn.
                if envelope.epoch > seen.painted_epoch:
                    seen.painted_epoch = envelope.epoch
                    await socket.send(  # type: ignore[attr-defined]
                        json.dumps({"type": "first_paint", "epoch": envelope.epoch})
                    )
            else:
                seen.audio_chunks += 1
                # 16-bit mono at the declared rate.
                rate = int(seen.hello.get("sample_rate", 16_000))
                chunk_ms = round(len(envelope.payload) / 2 / rate * 1000)
                seen.audio_ms += chunk_ms
                # Acknowledge playback the way the browser does. Without this the
                # server has no evidence anything was heard, and an interrupted turn
                # would correctly record nothing.
                await socket.send(  # type: ignore[attr-defined]
                    json.dumps(
                        {"type": "audio_played", "ms": chunk_ms, "epoch": envelope.epoch}
                    )
                )
            continue

        payload = json.loads(message)
        kind = payload.get("type")
        if kind == "hello":
            seen.hello = payload
        elif kind == "flush_audio":
            seen.flushes += 1
        elif kind == "error":
            seen.errors.append(str(payload.get("detail")))
        elif kind == "stats":
            seen.frames_repeated = int(payload["frames_repeated"])
            seen.frames_discarded = int(payload["frames_discarded"])
        elif kind == "event":
            event = str(payload.get("event"))
            seen.events[event] += 1
            if event == "state_change":
                seen.states.append(str(payload["to"]))
            elif event == "stale_dropped":
                seen.stale_kinds[str(payload["kind"])] += 1
            elif event == "latency":
                seen.latencies[str(payload["stage"])] = float(payload["ms"])


def check(label: str, condition: bool, detail: object = "") -> bool:
    mark = "ok  " if condition else "FAIL"
    suffix = f"  -- {detail}" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return condition


async def main() -> int:
    seen = Observed()
    stop = asyncio.Event()

    async with websockets.connect(URL, max_size=None) as socket:
        connected_at = time.monotonic()
        watcher = asyncio.create_task(observe(socket, seen, stop))

        await asyncio.sleep(0.6)  # let the idle loop run so the track is proven live
        idle_frames = seen.video_frames

        print("\nturn 1: candidate asks, avatar answers to completion")
        await socket.send(json.dumps({"type": "speech_start"}))
        await asyncio.sleep(0.1)
        await socket.send(json.dumps({"type": "end_of_turn", "transcript": "smoke test"}))
        started = time.monotonic()
        await asyncio.sleep(SPEAK_SETTLE_SECONDS)
        spoke_for = time.monotonic() - started
        audio_after_turn1 = seen.audio_ms

        print("\nturn 2: candidate barges in mid-sentence")
        await socket.send(json.dumps({"type": "speech_start"}))
        await asyncio.sleep(0.1)
        await socket.send(json.dumps({"type": "end_of_turn", "transcript": "second"}))
        await asyncio.sleep(0.8)
        await socket.send(json.dumps({"type": "speech_start"}))  # <-- barge-in
        await asyncio.sleep(BARGE_IN_SETTLE_SECONDS)

        elapsed = time.monotonic() - connected_at
        stop.set()
        await watcher

    print("\n--- observed ---")
    print(f"  states      {' -> '.join(seen.states)}")
    print(f"  video       {seen.video_frames} frames, {idle_frames} before any turn")
    print(f"  audio       {seen.audio_chunks} chunks, {seen.audio_ms}ms")
    print(f"  stale drops {dict(seen.stale_kinds) or 'none'}")
    print(f"  mixer       {seen.frames_repeated} repeated, {seen.frames_discarded} discarded")
    print(f"  flushes     {seen.flushes}")
    print(f"  latency     {ered(seen.latencies)}")

    print("\n--- assertions ---")
    results = [
        check(
            "handshake declares the wire format",
            bool(seen.hello),
            str(seen.hello.get("renderer")),
        ),
        check(
            "track carries frames before any turn starts",
            idle_frames > 5,
            f"{idle_frames} idle frames in 0.6s",
        ),
        check(
            "frame cadence is near target",
            0.7 <= (seen.video_frames / elapsed) / TARGET_FPS <= 1.1,
            f"{seen.video_frames / elapsed:.1f}fps over ~{elapsed:.1f}s",
        ),
        check(
            "presentation timestamps are strictly monotonic",
            all(b > a for a, b in zip(seen.frame_pts, seen.frame_pts[1:], strict=False)),
        ),
        check("the avatar reached SPEAKING", "SPEAKING" in seen.states),
        check(
            "audio reached the client",
            seen.audio_chunks > 0,
            f"{audio_after_turn1}ms in turn 1",
        ),
        check(
            "first-frame latency was measured",
            "avatar_first_frame" in seen.latencies,
            f"{seen.latencies.get('avatar_first_frame', 0):.0f}ms",
        ),
        check(
            "LLM and TTS stages were measured",
            {"llm_ttft", "tts_first_audio"} <= seen.latencies.keys(),
        ),
        check("barge-in returned the session to LISTENING", seen.states[-1] == "LISTENING"),
        check("barge-in flushed the client's audio", seen.flushes >= 1, f"{seen.flushes}"),
        check(
            "stale audio was dropped at the epoch check, not merely overtaken",
            seen.stale_kinds["audio"] > 0,
            dict(seen.stale_kinds),
        ),
        check(
            "rendered frames for the cancelled turn were discarded",
            seen.frames_discarded > 0,
            f"{seen.frames_discarded} frames",
        ),
        check(
            "interruption-to-silence was measured",
            "interrupt_to_silent" in seen.latencies,
            f"{seen.latencies.get('interrupt_to_silent', 0):.1f}ms server-side only",
        ),
        check(
            "end-to-end was measured to browser paint, not to socket write",
            "perceived_total" in seen.latencies
            and seen.latencies["perceived_total"]
            >= seen.latencies.get("avatar_first_frame", 0),
            f"paint={seen.latencies.get('perceived_total', 0):.0f}ms vs "
            f"first frame={seen.latencies.get('avatar_first_frame', 0):.0f}ms",
        ),
        check("no protocol errors", not seen.errors, "; ".join(seen.errors)),
    ]

    passed = sum(results)
    print(f"\n{passed}/{len(results)} assertions passed")
    print(f"(spoke for {spoke_for:.1f}s wall clock)")
    return 0 if passed == len(results) else 1


def ered(latencies: dict[str, float]) -> str:
    if not latencies:
        return "none"
    return ", ".join(f"{k}={v:.0f}ms" for k, v in sorted(latencies.items()))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
