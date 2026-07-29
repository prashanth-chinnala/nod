"""
The assistant's HTTP surface: one streaming chat endpoint and a capability listing.

**Its own service, on its own port.** Not merged into the runtime on :8000, because that process
holds live interviews — one orchestrator per socket, with a latency budget it already misses. An
assistant question that spends eight tool rounds and several seconds of model time has no
business sharing an event loop with a conversation that is trying to answer in under a second.
Separate process, separate dependency set, and the interview keeps running if this crashes.

**SSE rather than a WebSocket.** The traffic is one-directional once a question is asked: tokens
and tool activity flow out, nothing flows back until the next question. SSE is a plain HTTP
response that reconnects on its own, which is materially less to get wrong than a socket — and
the runtime already demonstrates how much care a socket needs.

**Tool activity is streamed, not just tokens.** A question like "audit consistency across these
six candidates" spends most of its time in tool calls with nothing to show, and a silent spinner
for ten seconds reads as broken. Emitting which tool is running turns the wait into progress,
and it also makes the assistant legible: the operator can see it looked at interview quality
before commenting on a low score.

**No authentication, stated plainly.** This endpoint will read any interview transcript in the
store and it is bound to loopback with CORS limited to the console's two origins, which is a
development posture and not a security one. It is the sharpest instance of a gap the whole
product has, and it gets sharper the moment anything here is deployed. Written down rather than
left to be discovered.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from avatar.config import load_env, loaded_files
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

# Before anything reads the environment, matching the runtime. Without it the model
# configuration is silently absent and the assistant answers every question by failing to
# reach a provider.
#
# It also carries AVATAR_DATA_DIR, which has to be absolute and shared. `avatar.store` defaults
# to a relative "data", so this service started from its own directory opened an empty store and
# answered "no interviews have been scored" about a pipeline with several -- confidently, and
# with nothing wrong in any log.
_FROM_ENV_FILE = load_env()

app = FastAPI(title="nod assistant", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)

MAX_HISTORY = 40
"""
How many prior messages a client may replay.

This service is stateless -- the client owns the conversation -- so history arrives with every
request, and without a bound a long thread eventually exceeds the model's context and fails at
the provider rather than here. Truncation keeps the most recent turns, which is where the
question is.
"""


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(pattern="^(user|assistant)$")
    content: str


class Screen(BaseModel):
    """
    What the operator is looking at, derived by the client from its own URL.

    Sent so "this session" and "why did she score badly" resolve without the operator pasting an
    id. Derived from the route rather than reported by each page: a registry every page has to
    remember to update goes stale the first time someone adds a screen, and a stale context is
    worse than none -- it would answer confidently about the wrong candidate.

    Advisory, not authoritative. It is injected as a hint the model may use to resolve a
    pronoun; every fact still comes from a tool call against the store, so a wrong context
    produces a question about the wrong id rather than an invented answer about the right one.
    """

    model_config = ConfigDict(extra="forbid")

    route: str = ""
    label: str = ""
    session_id: str | None = None
    rubric_id: str | None = None
    agent_id: str | None = None


class Ask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    history: list[Turn] = Field(default_factory=list)
    actor: str = "unknown"
    screen: Screen | None = None
    """
    Who is asking, as claimed by the caller.

    Unverified, because there is no auth. It is passed into write tools so a proposal carries a
    name, which makes the trail complete in shape from now rather than starting on the day
    accounts exist. Nothing is authorised on the basis of it.
    """


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/capabilities")
async def capabilities() -> dict[str, Any]:
    """
    What the assistant can read and what it can change.

    Served rather than documented, so the console can show it and so the answer cannot drift
    from the code: both lists are generated from the tool sets the graph is actually built with.
    The read/write split is the security model, and this is where a reviewer checks it without
    reading Python.
    """
    from assistant.tools_read import READ_TOOLS
    from assistant.tools_write import WRITE_TOOLS

    def describe(tools: list[Any]) -> list[dict[str, str]]:
        return [
            {"name": t.name, "summary": (t.description or "").strip().split("\n")[0]}
            for t in tools
        ]

    return {
        "reads": describe(READ_TOOLS),
        "writes": describe(WRITE_TOOLS),
        "model": os.environ.get("AVATAR_ASSISTANT_MODEL")
        or os.environ.get("AVATAR_LLM_MODEL", "not configured"),
        "env_files_read": loaded_files(),
        "writes_are_proposals": True,
        "auth": "none — this service will read any transcript in the store",
        "note": (
            "Every write records a proposal and returns its id. Nothing here edits a rubric, a "
            "rating or a weight; a human applies a proposal in the console."
        ),
    }


def _to_messages(body: Ask) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage

    history = body.history[-MAX_HISTORY:]
    messages: list[Any] = [
        HumanMessage(content=turn.content)
        if turn.role == "user"
        else AIMessage(content=turn.content)
        for turn in history
    ]
    messages.append(HumanMessage(content=body.message))
    return messages


def _context_note(body: Ask) -> str:
    """
    Who is asking and what they are looking at, as one message.

    Injected per request rather than baked into the system prompt, because the prompt is built
    once and both of these change every time. Phrased as context the model may use rather than
    as instructions, and it says explicitly that the ids are unconfirmed -- otherwise a model
    asked "summarise this" on a stale tab will happily describe whatever id it was handed
    without checking it exists.
    """
    lines = [
        f"The person asking is: {body.actor}. "
        "Pass this as `actor` to any write tool you use."
    ]
    screen = body.screen
    if screen and (screen.route or screen.session_id or screen.rubric_id):
        where = screen.label or screen.route or "the console"
        lines.append(
            f"They are currently looking at {where}"
            + (f" (route {screen.route})" if screen.route else "")
            + "."
        )
        ids = {
            "session_id": screen.session_id,
            "rubric_id": screen.rubric_id,
            "agent_id": screen.agent_id,
        }
        named = {key: value for key, value in ids.items() if value}
        if named:
            lines.append(
                "On-screen ids, for resolving words like \"this\" or \"her\": "
                + ", ".join(f"{key}={value}" for key, value in named.items())
                + ". Verify with a tool before describing any of them; do not assume the "
                "record exists, or that it is the one they mean if the question names "
                "something else."
            )
    return "\n".join(lines)


async def _stream(body: Ask) -> AsyncIterator[dict[str, str]]:
    """
    Run the graph and emit tokens, tool activity, and a terminal event.

    Every event is a JSON object with a `type`, rather than raw text, so the client can
    distinguish a token from a tool call from an error without parsing prose. `astream_events`
    is what makes the tool activity visible; without it the transport would carry only the final
    answer and a long silence.

    Failures are emitted as an event and then the stream closes normally. An SSE connection that
    drops on error looks identical to a network problem from the browser, and the actual cause
    -- usually an unreachable model provider -- would be invisible to the person who can fix it.
    """
    from langchain_core.messages import HumanMessage

    from assistant.graph import build_graph

    try:
        graph = build_graph()
    except Exception as exc:
        yield {
            "event": "message",
            "data": json.dumps(
                {
                    "type": "error",
                    "detail": (
                        f"the assistant could not start its model "
                        f"({type(exc).__name__}: {exc}). "
                        "Set AVATAR_LLM_MODEL and OPENAI_BASE_URL, or OPENAI_API_KEY."
                    ),
                }
            ),
        }
        yield {"event": "message", "data": json.dumps({"type": "done"})}
        return

    messages = _to_messages(body)
    messages.insert(max(0, len(messages) - 1), HumanMessage(content=_context_note(body)))

    try:
        async for event in graph.astream_events({"messages": messages}, version="v2"):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                text = getattr(chunk, "content", "")
                if text:
                    yield {
                        "event": "message",
                        "data": json.dumps({"type": "token", "text": text}),
                    }
            elif kind == "on_tool_start":
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {"type": "tool", "name": event.get("name", ""), "state": "start"}
                    ),
                }
            elif kind == "on_tool_end":
                # The result is deliberately not streamed. Tool output here contains
                # verbatim transcript text, and shipping all of it to the browser would put
                # candidate data in a place nobody asked for it -- the model's answer is what
                # the operator reads.
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {"type": "tool", "name": event.get("name", ""), "state": "end"}
                    ),
                }
    except Exception as exc:
        yield {
            "event": "message",
            "data": json.dumps({"type": "error", "detail": f"{type(exc).__name__}: {exc}"}),
        }

    yield {"event": "message", "data": json.dumps({"type": "done"})}


@app.post("/ask")
async def ask(body: Ask) -> EventSourceResponse:
    """
    Ask a question. Responds as a stream of events, not one JSON body.

    POST rather than GET despite being SSE: the question and its history go in the body, and a
    conversation in a query string would be logged by every proxy between here and the browser
    -- with transcript quotes in it.
    """
    return EventSourceResponse(_stream(body))


@app.get("/voice")
async def voice_status() -> dict[str, Any]:
    """
    Whether speech is configured, so the console can hide the microphone rather than offer one
    that fails.
    """
    from assistant import voice

    return {
        "available": voice.available(),
        "voice": voice.voice_name() if voice.available() else None,
        "detail": (
            "speech in and out via Deepgram"
            if voice.available()
            else "DEEPGRAM_API_KEY is not set; the assistant is text-only"
        ),
    }


@app.post("/transcribe")
async def transcribe(request: Request) -> dict[str, Any]:
    """
    One recorded question to text.

    Takes the raw body with the browser's own Content-Type rather than multipart: MediaRecorder
    hands over a Blob whose type is platform-specific -- webm/opus on Chrome, mp4 on Safari --
    and Deepgram sniffs the container, so forwarding it unchanged is what makes both work.

    A failure returns a message rather than a status code alone. This is triggered by someone
    holding a microphone button, and "transcription failed" with no reason is indistinguishable
    from a broken microphone.
    """
    from assistant import voice

    body = await request.body()
    try:
        text = await voice.transcribe(body, request.headers.get("content-type", "audio/webm"))
    except voice.VoiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"transcription failed: {type(exc).__name__}: {exc}"
        ) from exc
    return {"text": text, "empty": not text}


class Speak(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


@app.post("/speak")
async def speak(body: Speak) -> Response:
    """
    One answer to one audio file.

    Returns the audio directly rather than a URL, so nothing has to be stored: an answer about a
    candidate synthesised to a file on disk is a second copy of interview content with its own
    retention question, for no benefit over streaming the bytes once.
    """
    from assistant import voice

    try:
        audio, content_type = await voice.speak(body.text)
    except voice.VoiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"synthesis failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not audio:
        raise HTTPException(status_code=400, detail="nothing to say")
    # `no-store`: this is a recruiter listening to an assessment, and a proxy holding it is a
    # copy of candidate information nobody asked to make.
    return Response(
        content=audio, media_type=content_type, headers={"Cache-Control": "no-store"}
    )
