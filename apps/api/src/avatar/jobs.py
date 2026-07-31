"""
Background work that outlives an HTTP request, with its state in the store.

**The problem.** Enrollment takes minutes -- 126s for a 550-frame reference, and animating a
photograph first adds another two. `POST /faces/{id}/prepare` held the connection open for all
of it, which fails in three separate ways: a proxy or a browser times out and the operator sees
an error for work that is still running fine; nothing reports progress; and a process killed
midway leaves a row claiming `preparing` for ever, which `PREPARABLE` will not accept again, so
the face is permanently unenrollable and the only fix is deleting it.

**Why this is not Redis.** Redis is already running for egress, and a queue on it would be the
conventional answer. It would also be the wrong one at this size: there is one API process and
one GPU, so a distributed queue buys nothing but a second failure mode. What the problem
actually needs is (a) the request returning immediately, (b) status an operator can poll, and
(c) recovery from a crash. Those are three small things, and inventing a broker to get them
would be the clever choice rather than the correct one.

**What this does not do**, stated so nobody discovers it later: no cross-process dispatch, no
retries, no priorities, no fairness. One worker thread per job, bounded by a semaphore. The
moment a second API process or a GPU pool exists, this becomes a real queue and the store's
`status` field is already the contract that migration would preserve.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from avatar.store import NotFound, Store

MAX_CONCURRENT = int(os.environ.get("AVATAR_JOB_CONCURRENCY", 1))
"""
How many enrollment jobs run at once. One, by default, and that is a GPU decision.

Two concurrent MuseTalk preparations on one card do not halve the wall clock -- they contend,
the same way the renderer and the voice cloner do, and the measured cost of that contention was
a 10x regression. Serialising is faster than sharing here.
"""

STALE_AFTER = timedelta(minutes=int(os.environ.get("AVATAR_JOB_STALE_MINUTES", 30)))
"""
How long a job may claim to be running before a restart calls it dead.

Generously above the slowest real job measured (126s enrollment plus ~124s animation), because
marking a live job dead is worse than leaving a dead one a while longer: the first produces two
workers writing the same record.
"""

_slots = threading.Semaphore(MAX_CONCURRENT)
_running: dict[str, float] = {}
"""Job ids currently held by a worker in this process, with their start time. For `/config`."""


def running() -> dict[str, float]:
    """What this process is working on, and for how long. Copied, so callers cannot mutate."""
    now = time.time()
    return {key: round(now - started, 1) for key, started in _running.items()}


def submit(
    data: Store,
    collection: str,
    record_id: str,
    work: Callable[[], dict[str, Any]],
    *,
    label: str = "",
) -> None:
    """
    Run `work` on a worker thread and write its outcome to the record.

    `work` returns the patch to apply on success and raises on failure; it must not touch the
    store
    itself, so that success and failure are recorded in exactly one place and cannot disagree.

    The store is passed in rather than imported. An earlier version bound the module-level
    `store`
    at import, which meant a caller using a different one -- every test, and anything that swaps
    backends after import -- had its writes silently sent to the default location instead. That
    is
    the same failure as reading `AVATAR_STORE` too late, and the fix is the same: do not capture
    a
    global that someone else owns.

    Fire-and-forget by design -- the caller has already answered the request. Which means every
    failure has to be caught here: an exception escaping a worker thread would be printed to
    stderr
    and lost, leaving the record in `preparing` with no reason, which is the state this module
    exists
    to eliminate.
    """

    def run() -> None:
        key = f"{collection}/{record_id}"
        # Acquired inside the thread, not before it starts, so the request returns immediately
        # even
        # when a job is already running. The queue is the blocked threads.
        with _slots:
            _running[key] = time.time()
            try:
                patch = work()
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                print(f"job {key} failed: {reason}\n{traceback.format_exc()}", flush=True)
                _finish(
                    data, collection, record_id,
                    {"status": "failed", "failure_reason": reason},
                )
            else:
                _finish(data, collection, record_id, {"failure_reason": None, **patch})
            finally:
                _running.pop(key, None)

    threading.Thread(target=run, name=label or f"job-{record_id}", daemon=True).start()


def _finish(data: Store, collection: str, record_id: str, patch: dict[str, Any]) -> None:
    """
    Write the outcome, tolerating a record deleted while the job ran.

    An operator who deletes a face mid-enrollment has answered the question; recreating the row
    from
    a background thread would be the surprising behaviour, and a traceback in the log about a
    `NotFound` that nobody needs to act on is noise.
    """
    try:
        data.update(collection, record_id, patch)
    except NotFound:
        print(f"job {collection}/{record_id}: record deleted before it finished", flush=True)


def claim(
    data: Store, collection: str, record_id: str, started_field: str = "job_started_at"
) -> None:
    """
    Mark a record as running, with a timestamp, before any work begins.

    The timestamp is what makes recovery possible: without it a `preparing` row is
    indistinguishable
    from one whose worker died, and the only safe action is to leave it stuck for ever -- which
    is
    exactly what happened before this existed.
    """
    data.update(
        collection,
        record_id,
        {"status": "preparing", started_field: datetime.now(UTC).isoformat()},
    )


def reap(
    data: Store, collection: str, started_field: str = "job_started_at"
) -> list[str]:
    """
    Fail any record still claiming to run from a process that is gone. Returns the ids.

    Called at startup, because that is the one moment this process knows no job of its own is in
    flight -- so anything marked `preparing` belongs to a previous life and cannot be adopted.

    A row with no timestamp is failed too. It predates this module, which means it has been
    stuck
    since whenever it was written, and leaving it is the behaviour being fixed.
    """
    reaped: list[str] = []
    cutoff = datetime.now(UTC) - STALE_AFTER
    try:
        records = data.list(collection)
    except Exception:
        return reaped

    for record in records:
        if record.get("status") != "preparing":
            continue
        started = str(record.get(started_field) or "")
        try:
            fresh = datetime.fromisoformat(started) > cutoff if started else False
        except ValueError:
            fresh = False
        if fresh:
            # A restart that fast means the timestamp is younger than the stale window, so this
            # might be a legitimately long job whose process is *also* still running -- during a
            # rolling restart, for instance. Left alone: the cost is a stuck row that the next
            # restart reaps, against the cost of two workers on one record.
            continue
        _finish(
            data,
            collection,
            record["id"],
            {
                "status": "failed",
                "failure_reason": (
                    "the process handling this stopped before it finished"
                    + (f" (claimed at {started})" if started else "")
                    + ". Nothing was corrupted -- press Prepare again."
                ),
            },
        )
        reaped.append(str(record["id"]))
    return reaped


def wait_for_idle(timeout: float = 60.0) -> bool:
    """
    Block until no job is running in this process. Returns False on timeout.

    For tests and for a graceful shutdown. Polling rather than a condition variable because the
    thing being waited on is a dict mutated by daemon threads, and a 20ms poll is invisible next
    to
    jobs measured in minutes.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _running:
            return True
        time.sleep(0.02)
    return not _running


async def reap_all() -> None:
    """Reap every collection that runs jobs. Awaited from the app's lifespan."""
    from avatar.store import store

    for collection in ("faces",):
        reaped = await asyncio.to_thread(reap, store, collection)
        if reaped:
            print(
                f"jobs: failed {len(reaped)} stale {collection} record(s) left by a previous "
                f"process: {', '.join(reaped)}",
                flush=True,
            )
