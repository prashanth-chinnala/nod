"""
Executing the functions an agent may call mid-interview.

**The hazard here is latency, not correctness.** A tool call inserts a round trip *inside* a
conversational turn that already measures 2.7-5.8s against a sub-second target, and while it
runs the candidate is sitting in front of a silent avatar with no way to tell whether it is
thinking or dead. So every rule below is about bounding that:

  * every call has a hard deadline, capped at the value the console enforces;
  * a timeout returns a result the model can use rather than raising, because a turn that dies
    because a scoring endpoint was slow is worse than a turn that continues without the score;
  * the number of calls per turn is bounded, because a model that loops between two tools would
    otherwise hold the turn open indefinitely.

**Failures are values, not exceptions.** Every failure path returns a short string the model
reads as a tool result. That is deliberate: the model can say "I could not retrieve that" and
carry on, whereas an exception ends the turn and the candidate hears nothing at all. The
distinction matters most when the tool is non-essential, which is the common case — a scoring
call failing should never cost the interview.

**Builtins do nothing on purpose.** `score_answer` and `flag_for_review` acknowledge and return.
Persisting a score belongs with the session record and needs a decision about who may read it;
pretending to store it would be worse than saying it is not stored.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

TIMEOUT_CEILING_MS = 5000
"""
Outer wall on any single call, matching what the console's Tool schema enforces.

Duplicated deliberately rather than imported: this module must bound a call even if it is
handed a record written by an older build or edited by hand on disk. A ceiling that only exists
at the write boundary is not a ceiling.
"""

MAX_CALLS_PER_TURN = 3
"""
How many tool calls one turn may make.

A model can be talked into alternating between two tools indefinitely, and each iteration is
another round trip inside a turn the candidate is waiting on. Three is enough for
"score, then flag", and the cap is reported to the model when reached so it stops asking rather
than being silently ignored.
"""


@dataclass(frozen=True, slots=True)
class Tool:
    """One callable function, as the console stores it."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    kind: str = "builtin"
    url: str | None = None
    timeout_ms: int = 1500

    @property
    def deadline_seconds(self) -> float:
        """Clamped, so a bad record cannot hold a turn open past the ceiling."""
        return min(max(self.timeout_ms, 1), TIMEOUT_CEILING_MS) / 1000

    def to_openai(self) -> dict[str, Any]:
        """The wire shape a chat-completions request expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema or {"type": "object", "properties": {}},
            },
        }


def tools_from_records(records: list[dict[str, Any]]) -> list[Tool]:
    """
    Build tools from stored records, skipping the disabled ones.

    Disabled tools are dropped here rather than filtered at the call site, so a tool switched
    off in the console cannot be offered to the model by a caller that forgot to check.
    """
    built: list[Tool] = []
    for record in records:
        if not record.get("enabled", True):
            continue
        name = str(record.get("name") or "").strip()
        if not name:
            continue
        built.append(
            Tool(
                name=name,
                description=str(record.get("description") or ""),
                parameters_schema=dict(record.get("parameters_schema") or {}),
                kind=str(record.get("kind") or "builtin"),
                url=record.get("url"),
                timeout_ms=int(record.get("timeout_ms") or 1500),
            )
        )
    return built


class ToolExecutor:
    """
    Runs a named tool and returns a string for the model.

    Records per-tool durations so the console can show a measured p95. A tool taking 900ms is a
    product decision, and it cannot be made without the number.
    """

    def __init__(self, tools: list[Tool]) -> None:
        self._by_name = {tool.name: tool for tool in tools}
        self.durations: dict[str, list[float]] = {}
        self.calls = 0

    @property
    def specs(self) -> list[dict[str, Any]]:
        return [tool.to_openai() for tool in self._by_name.values()]

    @property
    def available(self) -> bool:
        return bool(self._by_name)

    async def run(self, name: str, arguments: str) -> str:
        """
        Execute by name. Never raises.

        An unknown name is answered rather than ignored: models do occasionally invent a
        function, and telling it so is what makes it stop, whereas silence invites a retry that
        costs another round trip.
        """
        tool = self._by_name.get(name)
        if tool is None:
            return f"error: no tool named {name!r} is available"

        self.calls += 1
        started = asyncio.get_running_loop().time()
        try:
            async with asyncio.timeout(tool.deadline_seconds):
                result = await self._dispatch(tool, arguments)
        except TimeoutError:
            result = (
                f"error: {tool.name} did not respond within {tool.timeout_ms}ms. "
                "Continue without it."
            )
        except Exception as exc:
            result = f"error: {tool.name} failed ({type(exc).__name__}). Continue without it."
        finally:
            elapsed = (asyncio.get_running_loop().time() - started) * 1000
            self.durations.setdefault(name, []).append(elapsed)
        return result

    async def _dispatch(self, tool: Tool, arguments: str) -> str:
        if tool.kind == "http":
            return await self._http(tool, arguments)
        return self._builtin(tool, arguments)

    async def _http(self, tool: Tool, arguments: str) -> str:
        if not tool.url:
            return f"error: {tool.name} is an http tool with no url configured"
        import httpx

        # A client per call rather than a shared pool. At a few calls per interview the
        # connection setup is not the term that matters, and a shared client would need a
        # lifecycle tied to the session -- more to get wrong than it saves.
        async with httpx.AsyncClient(timeout=tool.deadline_seconds) as client:
            response = await client.post(
                tool.url,
                content=arguments or "{}",
                headers={"content-type": "application/json"},
            )
        # The body is handed to the model as text whatever the status. A 4xx with an
        # explanation is more useful to it than a raised exception, and it can say so aloud.
        return response.text[:2000] or f"{tool.name} returned {response.status_code}"

    def _builtin(self, tool: Tool, arguments: str) -> str:
        """
        Acknowledge and return. Nothing is persisted, and that is stated rather than implied.

        Storing a score belongs with the session record and needs a decision about who may read
        it during an interview; a builtin that silently dropped the data while reporting success
        would be the worst of the three options.
        """
        try:
            parsed = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            return f"error: {tool.name} received arguments that were not valid JSON"
        summary = ", ".join(f"{key}={value!r}" for key, value in sorted(parsed.items()))
        return f"{tool.name} acknowledged ({summary or 'no arguments'}). Not persisted."
