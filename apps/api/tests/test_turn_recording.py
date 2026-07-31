"""
Every turn that happens reaches the record -- including the ones nobody spoke in.

**The gap this closes.** The server builds turns from the telemetry stream rather than from
explicit write calls, which is the right choice: the events are already the authority on what
happened, so a second recording path could disagree with the first. The cost is that a turn
only exists if a `heard` event announced it, and the silence watchdog did not emit one.
`on_idle_tick` appended its marker to history and called `_begin_turn` directly -- correctly,
because a re-prompt happens from IDLE and the end-of-turn guard requires LISTENING -- so the
re-prompt was generated, spoken aloud, heard by the candidate, and stored nowhere.

What a reviewer saw was a transcript that jumped from one answer to the next with no sign that
twelve seconds of silence and a nudge had happened in between. Not a crash, and not a wrong
number: an absence, which is the hardest kind of defect to notice in a record that otherwise
looks complete.

These tests drive `_accumulate` with the telemetry records the orchestrator actually emits, and
assert on what lands in the store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from avatar import server as server_module
from avatar.server import BrowserSession
from avatar.store import Store


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Store:
    """
    A throwaway store, patched over the one `_flush_turn` reaches for.

    `_flush_turn` uses the module-level store rather than an injected one, so the patch is on
    the module. That is the same reason `avatar.jobs` takes its store as a parameter: a
    module-level binding a test cannot redirect silently sends writes somewhere real.
    """
    replacement = Store(tmp_path)
    monkeypatch.setattr(server_module, "store", replacement)
    return replacement


@pytest.fixture
def session(store: Store) -> BrowserSession:
    """
    A `BrowserSession` with only the two attributes the turn recorder touches.

    Constructed without a socket, a renderer or an orchestrator on purpose. `_accumulate` and
    `_flush_turn` read `self._turn`, `self._session_id` and the module-level store and nothing
    else, so building a real session here would mean a WebSocket and a renderer for a test about
    bookkeeping -- and would hide which state the recorder actually depends on.
    """
    record = store.create("sessions", "sess", {"agent_id": None})
    instance = BrowserSession.__new__(BrowserSession)
    instance._turn = {}
    instance._session_id = str(record["id"])
    instance._turns_written = 0
    return instance


def heard(text: str, *, epoch: int = 1, transcribed: bool = True, silent: bool = False) -> dict:
    return {
        "event": "heard",
        "text": text,
        "epoch": epoch,
        "transcribed": transcribed,
        "silent": silent,
    }


def said(text: str) -> dict:
    return {"event": "said", "text": text}


def stored(store: Store, session: BrowserSession) -> list[dict[str, Any]]:
    return list(store.get("sessions", session._session_id)["turns"])


def test_a_silence_reprompt_is_recorded_as_a_turn(
    store: Store, session: BrowserSession
) -> None:
    """
    The defect, directly. A re-prompt must produce a turn holding the question it asked.

    Before the fix this list was empty and the question was lost.
    """
    session._accumulate(heard("", epoch=3, transcribed=False, silent=True))
    session._accumulate(said("Take your time -- shall I rephrase the question?"))
    session._flush_turn()

    turns = stored(store, session)
    assert len(turns) == 1
    assert turns[0]["said"] == "Take your time -- shall I rephrase the question?"
    assert turns[0]["silent"] is True
    assert turns[0]["epoch"] == 3


def test_a_silent_turn_stores_no_words_for_the_candidate(
    store: Store, session: BrowserSession
) -> None:
    """
    `heard` stays empty, because the candidate said nothing.

    The orchestrator does put `REPROMPT_TRANSCRIPT` into conversation history -- the LLM needs
    to know why it is being asked to speak again -- but that marker must not reach `heard`,
    which is what the scorer quotes the candidate from. A quote attributed to someone who was
    silent is worse than a missing turn.
    """
    session._accumulate(heard("", transcribed=False, silent=True))
    session._flush_turn()

    assert stored(store, session)[0]["heard"] == ""


def test_silence_is_distinguishable_from_transcription_that_failed(
    store: Store, session: BrowserSession
) -> None:
    """
    The reason `silent` is its own field rather than an empty `heard`.

    `transcribed=False` with no text already means something specific and urgent: speech was
    detected and the transcriber returned nothing, which is a broken-STT signal. If silence
    reused that shape, a quiet candidate and a misconfigured Deepgram key would look identical
    in the record -- and the second one needs someone paged.
    """
    session._accumulate(heard("", transcribed=False, silent=True))
    session._accumulate(said("Still there?"))
    session._accumulate(heard("", epoch=2, transcribed=False, silent=False))
    session._accumulate(said("Sorry, I did not catch that."))
    session._flush_turn()

    turns = stored(store, session)
    assert [t["silent"] for t in turns] == [True, False]
    assert [t["transcribed"] for t in turns] == [False, False]


def test_an_ordinary_turn_is_not_marked_silent(
    store: Store, session: BrowserSession
) -> None:
    """The flag defaults off, so nothing that was already working changes meaning."""
    session._accumulate(heard("We rewrote the ingest path."))
    session._accumulate(said("What broke first?"))
    session._flush_turn()

    turn = stored(store, session)[0]
    assert turn["silent"] is False
    assert turn["heard"] == "We rewrote the ingest path."


def test_a_reprompt_after_an_answer_does_not_swallow_the_answer(
    store: Store, session: BrowserSession
) -> None:
    """
    Two turns, in order: the answer, then the silence.

    `heard` flushes the previous turn before starting a new one, so the re-prompt arriving as a
    `heard` event is what *keeps* the preceding turn intact rather than what threatens it -- but
    that is worth asserting rather than assuming, because a flush is exactly the operation that
    could drop the thing being flushed.
    """
    session._accumulate(heard("We shipped it behind a flag.", epoch=1))
    session._accumulate(said("Why a flag?"))
    session._accumulate(heard("", epoch=2, transcribed=False, silent=True))
    session._accumulate(said("Would you like me to move on?"))
    session._flush_turn()

    turns = stored(store, session)
    assert [t["heard"] for t in turns] == ["We shipped it behind a flag.", ""]
    assert [t["silent"] for t in turns] == [False, True]
    assert [t["said"] for t in turns] == ["Why a flag?", "Would you like me to move on?"]


def test_timings_from_a_silent_turn_land_on_the_silent_turn(
    store: Store, session: BrowserSession
) -> None:
    """
    A re-prompt is a real generation with real latency, and it belongs to its own turn.

    Before the fix these timings were attributed to whichever turn happened to be open, which
    silently mixed re-prompt latencies into the answer latencies the report quotes.
    """
    session._accumulate(heard("A first answer.", epoch=1))
    session._accumulate({"event": "latency", "stage": "llm_ttft", "ms": 900})
    session._accumulate(heard("", epoch=2, transcribed=False, silent=True))
    session._accumulate({"event": "latency", "stage": "llm_ttft", "ms": 2400})
    session._flush_turn()

    turns = stored(store, session)
    assert [t["llm_ttft_ms"] for t in turns] == [900.0, 2400.0]
