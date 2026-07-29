"""
Tool execution and the tool-call loop.

Every test here is about a failure that would otherwise be invisible or expensive. A tool call
happens *inside* a turn the candidate is waiting through, so the interesting behaviours are the
bounds: does a slow tool end the turn, does a looping model hold it open, does a failure surface
as something the model can talk about.

The loop is exercised against a fake client, not the network. A live model may or may not choose
to call a function on any given turn, so a test that depended on it would be a coin flip — and
the logic worth testing is the accumulation of fragmented arguments and the round bound, neither
of which involves a provider.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from avatar.tools import (
    MAX_CALLS_PER_TURN,
    TIMEOUT_CEILING_MS,
    Tool,
    ToolExecutor,
    tools_from_records,
)


def tool(**overrides: Any) -> Tool:
    base: dict[str, Any] = {
        "name": "score_answer",
        "description": "Score the last answer",
        "parameters_schema": {"type": "object", "properties": {"score": {"type": "integer"}}},
    }
    base.update(overrides)
    return Tool(**base)


# -- records to tools ------------------------------------------------------


def test_a_disabled_tool_is_never_built() -> None:
    """
    Filtered here rather than at the call site, so a tool switched off in the console cannot be
    offered to the model by a caller that forgot to check.
    """
    built = tools_from_records(
        [
            {"name": "on", "enabled": True},
            {"name": "off", "enabled": False},
        ]
    )

    assert [t.name for t in built] == ["on"]


def test_a_record_with_no_name_is_skipped() -> None:
    """A nameless function cannot be called, and offering it would waste a round trip."""
    assert tools_from_records([{"name": "   "}, {"description": "no name"}]) == []


def test_enabled_defaults_to_true_for_older_records() -> None:
    """A record written before the field existed must not silently disappear."""
    assert len(tools_from_records([{"name": "legacy"}])) == 1


# -- the deadline ----------------------------------------------------------


def test_a_timeout_is_clamped_to_the_ceiling() -> None:
    """
    The console enforces this at write time, and so does this module — a ceiling that only
    exists at the write boundary is not a ceiling. A hand-edited record must not be able to
    hold a turn open for a minute.
    """
    assert tool(timeout_ms=60_000).deadline_seconds == TIMEOUT_CEILING_MS / 1000


def test_a_zero_or_negative_timeout_becomes_the_smallest_positive_one() -> None:
    """`0` would make every call fail instantly while looking configured."""
    assert tool(timeout_ms=0).deadline_seconds > 0
    assert tool(timeout_ms=-5).deadline_seconds > 0


async def test_a_slow_tool_returns_an_error_the_model_can_use() -> None:
    """
    The defect this prevents: a scoring endpoint being slow ends the turn, and the candidate
    hears nothing at all. A tool result saying "continue without it" lets the interview proceed.
    """

    @dataclass
    class Slow:
        name: str = "slow"

    class SlowExecutor(ToolExecutor):
        async def _dispatch(self, tool: Tool, arguments: str) -> str:
            await asyncio.sleep(1)
            return "never reached"

    executor = SlowExecutor([tool(name="slow", timeout_ms=20)])

    result = await executor.run("slow", "{}")

    assert "did not respond within 20ms" in result
    assert "Continue without it" in result


async def test_a_raising_tool_returns_an_error_rather_than_propagating() -> None:
    class Exploding(ToolExecutor):
        async def _dispatch(self, tool: Tool, arguments: str) -> str:
            raise RuntimeError("boom")

    result = await Exploding([tool(name="bang")]).run("bang", "{}")

    assert "bang failed (RuntimeError)" in result


async def test_an_unknown_tool_is_answered_not_ignored() -> None:
    """
    Models do occasionally invent a function. Telling it so is what makes it stop; silence
    invites a retry, and a retry is another round trip inside the turn.
    """
    result = await ToolExecutor([tool()]).run("invented_function", "{}")

    assert "no tool named 'invented_function'" in result


async def test_every_call_is_timed_even_when_it_fails() -> None:
    """A tool taking 900ms is a product decision, and it cannot be made without the number —
    including for the calls that time out, which are the ones worth seeing."""

    class Exploding(ToolExecutor):
        async def _dispatch(self, tool: Tool, arguments: str) -> str:
            raise RuntimeError("boom")

    executor = Exploding([tool(name="bang")])
    await executor.run("bang", "{}")

    assert len(executor.durations["bang"]) == 1


# -- builtins --------------------------------------------------------------


async def test_a_builtin_says_plainly_that_nothing_was_persisted() -> None:
    """
    Storing a score needs a decision about who may read it mid-interview. A builtin that
    silently dropped the data while reporting success would be the worst of the three options,
    so it reports the truth instead.
    """
    result = await ToolExecutor([tool()]).run("score_answer", '{"score": 4}')

    assert "score=4" in result
    assert "Not persisted" in result


async def test_malformed_arguments_are_reported_rather_than_crashing() -> None:
    result = await ToolExecutor([tool()]).run("score_answer", "{not json")

    assert "not valid JSON" in result


async def test_an_http_tool_with_no_url_is_reported() -> None:
    """A tool that cannot be called is a silent no-op mid-conversation unless it says so."""
    result = await ToolExecutor([tool(name="remote", kind="http", url=None)]).run("remote", "{}")

    assert "no url configured" in result


# -- the wire shape --------------------------------------------------------


def test_the_openai_spec_carries_a_schema_even_when_none_was_given() -> None:
    """An absent `parameters` is rejected by the API, so an empty object stands in."""
    spec = tool(parameters_schema={}).to_openai()

    assert spec["function"]["parameters"] == {"type": "object", "properties": {}}


def test_specs_are_only_offered_when_tools_exist() -> None:
    """`available` gates whether `tools` is sent at all — an empty list is not the same as
    omitting the field, and some providers reject it."""
    assert ToolExecutor([]).available is False
    assert ToolExecutor([tool()]).available is True


def test_the_round_bound_is_small_enough_to_matter() -> None:
    """
    Documented as a bound rather than a constant to tune: each round is another round trip
    inside a turn already measuring seconds, so the value has to stay small enough that three
    of them cannot dominate the budget.
    """
    assert 1 <= MAX_CALLS_PER_TURN <= 5


# -- the loop, against a fake provider ------------------------------------


class FakeStream:
    """Mimics the provider's async-context streaming object over a scripted list of chunks."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def __aiter__(self) -> Any:
        async def gen() -> Any:
            for chunk in self._chunks:
                yield chunk

        return gen()


def text_chunk(content: str) -> Any:
    delta = type("D", (), {"content": content, "tool_calls": None})()
    choice = type("C", (), {"delta": delta})()
    return type("Chunk", (), {"choices": [choice]})()


def tool_chunk(index: int, call_id: str, name: str, arguments: str) -> Any:
    function = type("F", (), {"name": name, "arguments": arguments})()
    fragment = type("T", (), {"index": index, "id": call_id, "function": function})()
    delta = type("D", (), {"content": None, "tool_calls": [fragment]})()
    choice = type("C", (), {"delta": delta})()
    return type("Chunk", (), {"choices": [choice]})()


class FakeClient:
    """Returns a scripted stream per request, and records what it was asked."""

    def __init__(self, *rounds: list[Any]) -> None:
        self._rounds = list(rounds)
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Completions:
            async def create(self, **kwargs: Any) -> FakeStream:
                outer.requests.append(kwargs)
                return FakeStream(outer._rounds.pop(0) if outer._rounds else [])

        self.chat = type("Chat", (), {"completions": Completions()})()


def interviewer(client: FakeClient, tools: list[Tool] | None = None) -> Any:
    pytest.importorskip("openai")
    from avatar.llm_openai import OpenAIInterviewer

    return OpenAIInterviewer(
        client=client,
        executor=ToolExecutor(tools) if tools is not None else None,
    )


async def drain(stream: Any) -> str:
    return "".join([chunk async for chunk in stream])


async def test_no_tools_means_no_tools_field_on_the_request() -> None:
    """Some providers reject an empty list, and it is a meaningful signal to the model that
    there is nothing to call."""
    client = FakeClient([text_chunk("A question.")])

    await drain(interviewer(client)([{"role": "user", "content": "hi"}]))

    assert "tools" not in client.requests[0]


async def test_fragmented_tool_arguments_are_reassembled_before_execution() -> None:
    """
    Arguments arrive split across deltas exactly like content does. Parsing them incrementally
    would hand the executor half a JSON object, which fails in a way that looks like the model
    emitting nonsense.
    """
    client = FakeClient(
        [
            tool_chunk(0, "call_1", "score_answer", '{"sco'),
            tool_chunk(0, "", "", 're": 4}'),
        ],
        [text_chunk("Thanks, noted.")],
    )
    executor = ToolExecutor([tool()])
    from avatar.llm_openai import OpenAIInterviewer

    llm = OpenAIInterviewer(client=client, executor=executor)

    out = await drain(llm([{"role": "user", "content": "my answer"}]))

    assert out == "Thanks, noted."
    # The reassembled arguments reached the tool: the acknowledgement quotes the parsed value.
    tool_message = [m for m in client.requests[1]["messages"] if m.get("role") == "tool"]
    assert "score=4" in tool_message[0]["content"]


async def test_the_loop_stops_at_the_round_bound_and_says_so() -> None:
    """
    A model can alternate between two functions indefinitely. The bound is told to the model
    rather than silently enforced, because a request that vanishes gets repeated.
    """
    forever = [[tool_chunk(0, f"call_{i}", "score_answer", "{}")] for i in range(6)]
    client = FakeClient(*forever)
    llm = interviewer(client, [tool()])

    await drain(llm([{"role": "user", "content": "hi"}]))

    assert client.requests, "at least one request was made"
    assert len(client.requests) <= MAX_CALLS_PER_TURN + 2
    last = client.requests[-1]["messages"][-1]
    assert "tool budget" in str(last.get("content", ""))
