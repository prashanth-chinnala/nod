#!/usr/bin/env python3
"""
Answer "is this demoable right now" in one command, checking the things that fail silently.

**Why this exists.** An hour before a demo, three things were wrong and none of them produced an
error anywhere: the SFU was advertising a LAN address the machine had not had for five days, the
egress container was holding a connection to the SFU it started with, and recording had been off
since a restart. Every screen loaded, every endpoint returned 200, and the test suite was green.

So this checks the *environment*, not the code. The code has 844 tests; what it does not have is
any way to notice that the network changed underneath it.

Each check prints `ok`, `WARN`, or `FAIL` and says what to do about it. Exit code is non-zero if
anything FAILs, so it can gate a demo script.

    python scripts/preflight.py
    python scripts/preflight.py --api http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from avatar.config import load_env  # noqa: E402

load_env()

OK, WARN, FAIL = "ok  ", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def record(level: str, what: str, detail: str) -> None:
    results.append((level, what, detail))
    print(f"  [{level}] {what}  --  {detail}")


def lan_ip() -> str:
    """This machine's LAN address, or empty. Tries the interfaces a laptop actually uses."""
    for interface in ("en0", "en1", "en5", "eth0"):
        try:
            out = subprocess.run(
                ["ipconfig", "getifaddr", interface], capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return ""


def check_node_ip() -> None:
    """
    The one that cost the most to find, and the one nothing else can detect.

    LiveKit hands clients the node IP as an ICE candidate. If it names an address the machine no
    longer has, every WebRTC connection fails to establish media and falls back or dies -- the
    browser shows a black video element, the egress recorder gives up with `Start signal not
    received`, and the SFU logs a healthy room the whole time. A laptop that moved networks is
    all it takes.
    """
    import os

    configured = os.environ.get("LIVEKIT_NODE_IP", "")
    actual = lan_ip()
    if not configured:
        record(WARN, "LIVEKIT_NODE_IP", "not set; defaults to 127.0.0.1, which to a container is "
                                        "itself. Recording and containerised clients will fail.")
        return
    if configured in ("127.0.0.1", "localhost"):
        record(FAIL, "LIVEKIT_NODE_IP", f"is {configured}. A container reaching this resolves it to "
                                        "itself. Set it to a LAN address.")
        return
    if actual and configured != actual:
        record(FAIL, "LIVEKIT_NODE_IP", f"is {configured} but this machine is on {actual}. WebRTC "
                                        f"and recording will fail silently. Fix .env.development "
                                        f"and: docker compose --env-file .env.development up -d "
                                        f"--force-recreate")
        return
    record(OK, "LIVEKIT_NODE_IP", f"{configured} matches this machine")


def check_containers() -> None:
    """
    Running is not the same as *current*. A container that predates a config change still holds
    the old one, which is how the egress kept talking to an SFU that had been restarted under
    it.
    """
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.CreatedAt}}"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        record(FAIL, "docker", f"cannot list containers: {exc}")
        return
    rows = [line.split("\t") for line in out.stdout.strip().splitlines() if line.strip()]
    names = {row[0]: row for row in rows}
    for needed in ("livekit", "egress", "redis"):
        match = next((n for n in names if needed in n), None)
        if match is None:
            record(FAIL, f"container {needed}", "not running. docker compose --env-file "
                                                ".env.development up -d")
        else:
            record(OK, f"container {needed}", names[match][1])


def get(url: str, timeout: float = 10.0) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def check_runtime(api: str) -> None:
    try:
        config = get(f"{api}/config")
    except (OSError, urllib.error.HTTPError) as exc:
        record(FAIL, "runtime", f"unreachable at {api}: {exc}. cd apps/api && python -m uvicorn "
                                f"avatar.server:app")
        return
    assert isinstance(config, dict)
    record(OK, "runtime", f"renderer={config.get('renderer')} stt={config.get('stt')} "
                          f"tts={config.get('tts')} llm={config.get('llm')} "
                          f"delivery={config.get('delivery')}")

    problems = config.get("schema_problems") or []
    if problems:
        record(FAIL, "database schema", f"{problems}. Apply the migrations in apps/api/migrations.")
    else:
        record(OK, "database schema", "no problems reported")

    reason = config.get("worker_reason") or ""
    if reason:
        record(WARN, "worker delivery", f"asked for but unavailable: {reason}")

    # A demo where the interviewer speaks in a tone rather than a voice is a different demo.
    for name, value in (("stt", config.get("stt")), ("tts", config.get("tts")),
                        ("llm", config.get("llm"))):
        if value in ("none", "tone", "scripted"):
            record(WARN, f"{name} is a placeholder",
                   f"{name}={value}. Fine for a clean clone, wrong for a demo of the real thing.")


def check_content(api: str) -> None:
    """Empty collections mean a console with nothing to walk through."""
    for path, label in (("/agents", "interviewers"), ("/candidates", "candidates"),
                        ("/knowledge", "knowledge bases"), ("/faces", "faces"),
                        ("/voices", "voices"), ("/rubrics", "rubrics")):
        try:
            items = get(f"{api}{path}")
        except (OSError, urllib.error.HTTPError) as exc:
            record(FAIL, f"{path}", f"{exc}")
            continue
        count = len(items) if isinstance(items, list) else 0
        if count == 0:
            record(WARN, f"{label}", "none. python scripts/seed_demo.py")
        else:
            record(OK, f"{label}", f"{count}")


def check_recording(api: str) -> None:
    """
    Whether a recording would be produced, which is not the same as whether one was requested.

    `/rtc` can only honestly say `requested` -- it creates the room with an egress config and
    cannot see whether a recorder ever picked the job up. So this reports what the runtime
    intends and points at the only real evidence, which is a file.
    """
    import os

    if os.environ.get("AVATAR_RECORD", "").strip().lower() not in ("1", "true", "yes", "on"):
        record(WARN, "recording", "AVATAR_RECORD is not set, so no recording will be requested.")
        return
    recordings = ROOT.parent.parent / "recordings"
    files = sorted(recordings.glob("*.mp4"), key=lambda p: p.stat().st_mtime) if (
        recordings.exists()) else []
    if not files:
        record(WARN, "recording", "requested, but no .mp4 has ever been produced here. Recording "
                                  "needs a real WebRTC participant -- drive one session from a "
                                  "browser and check recordings/.")
        return
    newest = files[-1]
    size_mb = newest.stat().st_size / 1_000_000
    record(OK, "recording", f"enabled; newest is {newest.name} at {size_mb:.1f} MB")


def check_stt_liveness() -> None:
    """
    The transcriber surviving a quiet stretch, which is what it does during every avatar turn.

    Deepgram closes an idle stream after ~10 s. This holds one open for 12 with no audio, which
    is the exact condition that used to kill it -- and did, silently, for 38 turns of a real
    interview.
    """
    import asyncio
    import os

    if os.environ.get("AVATAR_STT", "none") != "deepgram":
        record(WARN, "transcriber", "AVATAR_STT is not deepgram; nothing will be transcribed.")
        return

    async def probe() -> tuple[bool, int]:
        from avatar.audio.stt import DeepgramSTT

        stt = DeepgramSTT()
        await stt.connect()
        if not stt.connected:
            return False, 0
        await asyncio.sleep(12)
        alive, keeps = stt.connected, stt.keep_alives
        await stt.aclose()
        return alive, keeps

    try:
        alive, keeps = asyncio.run(probe())
    except Exception as exc:
        record(FAIL, "transcriber", f"could not connect: {exc}")
        return
    if not alive:
        record(FAIL, "transcriber", "died during 12 s of silence. The keepalive is not working, "
                                    "and every avatar turn longer than ~10 s will deafen it.")
    else:
        record(OK, "transcriber", f"survived 12 s idle, {keeps} keepalives sent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-stt", action="store_true", help="skip the 12 s transcriber probe")
    args = parser.parse_args()

    print("-- environment")
    check_node_ip()
    check_containers()
    print("-- runtime")
    check_runtime(args.api)
    print("-- content")
    check_content(args.api)
    print("-- media")
    check_recording(args.api)
    if not args.skip_stt:
        check_stt_liveness()

    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print(f"\n-- {len(results) - len(fails) - len(warns)} ok, {len(warns)} warn, {len(fails)} fail")
    if fails:
        print("!! not demoable until these are fixed:")
        for _, what, detail in fails:
            print(f"   {what}: {detail}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
