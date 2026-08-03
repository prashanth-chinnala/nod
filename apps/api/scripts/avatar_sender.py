#!/usr/bin/env python3
"""
The runtime half of the split: stream audio to an avatar worker, and interrupt it.

**What this is for.** `avatar_worker.py --audio stream` waits for audio on `lk.audio_stream`
from another participant. This is that participant. Together they exercise the arrangement the
whole migration is aiming at: the renderer in its own process, joined to the room, fed audio
over a data stream, and interrupted by an RPC rather than by reaching into its memory.

**The barge-in is the point, not a bonus.** `--interrupt-after` fires `lk.clear_buffer` partway
through an utterance. That is the regression the notes call the one that matters most:
in-process, cancellation is one integer write and cannot fail; across a boundary it is a round
trip, and the question is whether the frames actually stop. Run with and without it and compare
what the worker reports.

    # terminal 1
    python scripts/avatar_worker.py --audio stream --seconds 30 --room split-check
    # terminal 2
    python scripts/avatar_sender.py --room split-check --seconds 8 --interrupt-after 3
"""

from __future__ import annotations

import argparse
import asyncio
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from avatar.config import load_env  # noqa: E402

load_env()

SAMPLE_RATE = 16_000
CHUNK_MS = 80
"""
Deepgram delivers roughly this much audio per chunk, so the wire sees a realistic shape.

It matters: a chunk larger than one frame interval is what exposed the pairing bug in
`AvStream` -- a budget that looked like time but behaved like a count. Sending 20 ms here would
have hidden it.
"""


def chunk_pcm(index: int) -> bytes:
    """One chunk of a speech-shaped tone. Amplitude varies so silence is distinguishable."""
    per = SAMPLE_RATE * CHUNK_MS // 1000
    samples = []
    for n in range(per):
        t = (index * per + n) / SAMPLE_RATE
        envelope = 0.35 * (1.0 + math.sin(2 * math.pi * 1.3 * t)) / 2.0
        samples.append(int(32767 * envelope * math.sin(2 * math.pi * 190.0 * t)))
    return struct.pack(f"<{per}h", *samples)


async def run(args: argparse.Namespace) -> int:
    from livekit import rtc
    from livekit.agents.voice.avatar import DataStreamAudioOutput

    from avatar.transport.livekit import credentials, room_token

    url, _, _ = credentials()
    room = rtc.Room()
    await room.connect(url, room_token(args.room, args.identity, name="Runtime"))
    print(f"-- connected to {args.room!r} as {args.identity!r}")

    # Wait for the worker, because `DataStreamAudioOutput` addresses it by identity and a stream
    # to an absent participant is silently dropped rather than refused.
    def joined() -> bool:
        return args.worker in {p.identity for p in room.remote_participants.values()}

    deadline = args.wait
    while deadline > 0 and not joined():
        await asyncio.sleep(0.25)
        deadline -= 0.25
    present = joined()
    print(f"-- worker {args.worker!r} present: {present}")
    if not present:
        print("!! the worker never joined; start avatar_worker.py --audio stream first")
        await room.disconnect()
        return 1

    # `wait_remote_track` is the library's own answer to a real race, and without it the first
    # run of this looked like a silent failure: the sender wrote 100 chunks, the worker reported
    # zero received, and neither logged anything. The worker registers its byte-stream handler
    # as part of starting, and LiveKit drops a stream announced to a participant with no handler
    # for that topic -- so a sender that begins the moment the worker *joins* is too early.
    # Waiting for the worker's audio track means waiting until it can actually receive.
    out = DataStreamAudioOutput(
        room,
        destination_identity=args.worker,
        sample_rate=SAMPLE_RATE,
        wait_remote_track=rtc.TrackKind.KIND_AUDIO,
    )

    sent = 0
    interrupted = False
    total = int(args.seconds * 1000 / CHUNK_MS)
    print(f"-- sending {args.seconds}s as {total} chunks of {CHUNK_MS}ms")
    for index in range(total):
        elapsed = index * CHUNK_MS / 1000.0
        if args.interrupt_after and not interrupted and elapsed >= args.interrupt_after:
            # The barge-in. Sent mid-utterance, exactly as a candidate speaking over the avatar
            # would produce -- and deliberately *before* the remaining audio, so a worker that
            # ignored it would go on speaking and the difference would be visible in its counts.
            print(f"-- INTERRUPT at {elapsed:.1f}s: firing lk.clear_buffer")
            out.clear_buffer()
            interrupted = True
            if args.stop_on_interrupt:
                break
        await out.capture_frame(
            rtc.AudioFrame(
                data=chunk_pcm(index),
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=SAMPLE_RATE * CHUNK_MS // 1000,
            )
        )
        sent += 1
        await asyncio.sleep(CHUNK_MS / 1000.0)

    out.flush()
    print(f"-- flushed after {sent} chunks ({sent * CHUNK_MS / 1000.0:.1f}s of audio)")
    # A moment for the worker to drain what is in flight before the room closes under it.
    await asyncio.sleep(1.5)
    await room.disconnect()
    print("-- done")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", default="split-check")
    parser.add_argument("--identity", default="runtime-sender")
    parser.add_argument("--worker", default="avatar-agent")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument(
        "--interrupt-after",
        type=float,
        default=0.0,
        help="fire lk.clear_buffer this many seconds in; 0 disables",
    )
    parser.add_argument(
        "--stop-on-interrupt",
        action="store_true",
        help="stop sending after the interrupt, the way an abandoned turn actually stops",
    )
    parser.add_argument("--wait", type=float, default=20.0)
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
