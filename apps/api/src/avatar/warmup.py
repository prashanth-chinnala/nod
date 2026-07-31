"""
Pay the renderer's start-up cost before a candidate arrives, not while one waits.

**The problem this solves, with the measured numbers.** Two costs are unavoidable the first time
a real renderer is used in a process: loading five models (27s on a T4) and preparing an
identity from a reference clip (44-155s depending on its length). Nothing triggered either until
a session opened, so the first candidate to join after a restart waited **70-150 seconds**
looking at a placeholder. Every session after that started in 1.5s, because both costs are
cached process-wide.

That is the worst behaviour in the product, and it is entirely a scheduling problem: the work is
identical, it was just being done at the least useful moment.

**Why not simply call it at import.** Three reasons, each of which shaped the design:

* It must not block the event loop. `prepare_identity` is synchronous and slow, and a loop
  stuck inside it cannot finish a WebSocket handshake -- which is how this first surfaced,
  as `TimeoutError: timed out during opening handshake` rather than as slowness.
* It must not stop the process serving. A missing GPU, or weights never fetched, should
  degrade to a loud log line -- not a server that refuses to start. The console has to work
  for an operator to find out why.
* It must not be paid when useless. The stub has nothing to warm, and a developer
  restarting the API on every edit does not want two minutes of enrollment each time.

**Why it warms faces and not just models.** Loading the models is the smaller half. The identity
is the expensive part, and it is per-face, so warming the models alone would still leave the
first candidate waiting a minute. Which faces to warm is decided by which agents actually
reference one -- an unattached face is not going to be used by anybody.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

from avatar.contracts import RendererConfig

WARM_ENV = "AVATAR_WARM"
"""
Set to `0`/`false`/`no` to skip warming entirely.

Present because the cost is real and sometimes unwanted: a developer restarting the API on every
edit does not want to re-enroll a 550-frame reference each time, and would rather pay it once on
the first session.
"""


@dataclass
class WarmupReport:
    """
    What warming did, for `/config` to report.

    Kept rather than logged and forgotten, because "why is the first session slow" is a
    question an operator will ask, and the answer is one of three things: warming is off,
    warming failed and here is the reason, or warming succeeded and something else is slow.
    Those are three different investigations, and guessing wastes an afternoon.

    """

    skipped: str = ""
    """Why nothing was warmed. Empty when warming ran."""

    models_ms: int | None = None
    faces_warmed: list[str] = field(default_factory=list)
    faces_failed: list[str] = field(default_factory=list)
    total_ms: int | None = None
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        if self.skipped:
            return {"warm": False, "skipped": self.skipped}
        return {
            "warm": not self.error,
            "models_ms": self.models_ms,
            "faces_warmed": self.faces_warmed,
            "faces_failed": self.faces_failed,
            "total_ms": self.total_ms,
            **({"error": self.error} if self.error else {}),
        }


report = WarmupReport(skipped="not started")
"""Module-level, so `/config` can read it without the server holding a reference."""


def _disabled() -> bool:
    return os.environ.get(WARM_ENV, "1").strip().lower() in ("0", "false", "no")


def _references_to_warm(limit: int) -> list[tuple[str, str]]:
    """
    `(face_id, reference_path)` for the faces worth warming, most recently updated first.

    Only faces an agent actually references: warming one nothing points at spends minutes to
    populate a cache entry that will be evicted before use.

    Bounded by the identity cache's own size. Warming more than fits would enroll a
    reference and immediately evict it -- pure cost, and worse than nothing because it
    delays the faces that *will* be used. The bound is the cache's size rather than a
    number of our own, so one place decides how many identities exist at once.

    """
    from avatar.store import store

    try:
        agents = store.list("agents")
        faces = {face["id"]: face for face in store.list("faces")}
    except Exception:
        return []

    seen: dict[str, str] = {}
    for agent in sorted(agents, key=lambda a: str(a.get("updated_at") or ""), reverse=True):
        face_id = agent.get("face_id")
        if not face_id or face_id in seen:
            continue
        face = faces.get(str(face_id))
        # `ready` only. A queued face has never been enrolled, and enrolling it here would
        # hide a failure the operator needs to see on the Faces screen.
        if not face or face.get("status") != "ready":
            continue
        path = face.get("reference_path")
        if path:
            seen[str(face_id)] = str(path)
        if len(seen) >= limit:
            break
    return list(seen.items())


def _warm_blocking() -> WarmupReport:
    """The slow part, on a worker thread. Every failure is caught and reported, never raised."""
    from avatar.renderers import build

    result = WarmupReport()
    started = time.perf_counter()
    name = os.environ.get("AVATAR_RENDERER", "stub")

    try:
        renderer = build(RendererConfig(name=name))
    except Exception as exc:
        result.error = f"could not construct the {name!r} renderer: {type(exc).__name__}: {exc}"
        return result

    loader = getattr(renderer, "load", None)
    if callable(loader):
        mark = time.perf_counter()
        try:
            loader()
        except Exception as exc:
            result.error = f"model load failed: {type(exc).__name__}: {exc}"
            result.total_ms = round((time.perf_counter() - started) * 1000)
            return result
        result.models_ms = round((time.perf_counter() - mark) * 1000)
        print(f"warmup: models loaded in {result.models_ms} ms", flush=True)

    limit = int(os.environ.get("AVATAR_IDENTITY_CACHE", 2))
    for face_id, path in _references_to_warm(limit):
        mark = time.perf_counter()
        try:
            renderer.prepare_identity(path)
        except Exception as exc:
            result.faces_failed.append(f"{face_id}: {type(exc).__name__}: {exc}")
            print(f"warmup: {face_id} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        result.faces_warmed.append(face_id)
        print(
            f"warmup: {face_id} prepared in {round((time.perf_counter() - mark) * 1000)} ms",
            flush=True,
        )

    result.total_ms = round((time.perf_counter() - started) * 1000)
    return result


async def warm() -> None:
    """
    Warm the renderer, off the event loop, without ever failing startup.

    Awaited from the app's lifespan rather than fired as a background task. A background task
    would let the first session race the warming and pay the cost anyway, in a process that
    also has a second copy of the models loading beside it. Serving a few seconds later with a
    warm cache beats serving at once and being slow for the first candidate.

    """
    global report

    if _disabled():
        report = WarmupReport(skipped=f"{WARM_ENV} is off")
        return
    if os.environ.get("AVATAR_RENDERER", "stub") == "stub":
        # Not a failure. The stub loads nothing and prepares nothing, so there is nothing
        # to warm, and saying so is more useful than reporting a 0 ms success.
        report = WarmupReport(skipped="renderer is the stub; nothing to warm")
        return

    print("warmup: loading models and preparing attached faces...", flush=True)
    report = await asyncio.to_thread(_warm_blocking)
    if report.error:
        print(
            f"!! warmup: {report.error}\n"
            "   The server is running. The first session will pay this cost instead, "
            "or fail the same way.",
            flush=True,
        )
    else:
        print(
            f"warmup: ready in {report.total_ms} ms "
            f"({len(report.faces_warmed)} face(s) warm, {len(report.faces_failed)} failed)",
            flush=True,
        )
