"""
Resolve a stored Agent into the things a live session actually needs.

**The gap this closes.** Until now the console could create agents, knowledge bases and
lexicons, and the running conversation consulted none of them — it took every setting from an
environment variable. A console that configures things the runtime ignores is a demo of a
console, not a product. `AVATAR_AGENT=<id>` is what joins them.

**Why one env var still, rather than a per-session choice.** The candidate picks nothing; an
interview is opened *for* them. Selecting the agent per socket is the right end state and needs
a session token the console mints, which does not exist yet. Reading one id from the environment
is the smallest step that makes stored configuration real, and it moves to a token without
changing anything below this module.

**Failure is loud, deliberately.** A missing agent, knowledge base or lexicon raises at session
construction rather than degrading. The alternative — falling back to defaults — produces an
interviewer that behaves plausibly while ignoring the configuration someone carefully wrote,
which is the same failure class as the empty transcript that took a day to find: nothing errors,
the output is merely wrong.

Nothing here imports a renderer, torch, or a web framework, so it stays outside the
orchestration boundary that `tests/test_boundaries.py` enforces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from avatar.api.guardrails import Policy
from avatar.contracts import SentenceStream
from avatar.knowledge import build_retriever
from avatar.knowledge.contracts import Retriever
from avatar.knowledge.keyword import NullRetriever
from avatar.plan import (
    DEFAULT_MAX_TURNS,
    DEFAULT_MIN_SIGNALS,
    Competency,
    InterviewPlan,
    slug,
)
from avatar.store import NotFound, Store, store
from avatar.tools import Tool, tools_from_records

AGENT_ENV = "AVATAR_AGENT"
COLLECTION = "agents"


class AgentNotConfigured(RuntimeError):
    """
    A referenced resource is missing.

    Its own type rather than a bare RuntimeError so a caller can distinguish "misconfiguration,
    tell the operator" from "the model provider is down, retry".
    """


@dataclass
class ResolvedAgent:
    """
    Everything a session needs from stored configuration, already loaded.

    Resolution happens once at session start, not per turn. The retriever in particular indexes
    its whole corpus here, so the per-turn cost is a scored lookup rather than a reload — and
    re-reading documents on every turn would put file I/O inside the latency budget for no
    benefit.
    """

    agent_id: str | None = None
    name: str = ""
    system_prompt: str = ""
    retriever: Retriever = field(default_factory=NullRetriever)
    lexicon: list[tuple[str, str]] = field(default_factory=list)
    guardrail: Policy | None = None
    tools: list[Tool] = field(default_factory=list)
    plan: InterviewPlan = field(default_factory=InterviewPlan)
    face_reference: str | None = None
    """
    The path the renderer should build its identity from, resolved from `face_id`.

    Separate from `face_id` because the runtime needs the *file*, and only this module knows how
    a face record maps to one. Without it the whole faces resource was decorative: `face_id` was
    loaded here faithfully and then read by nothing, while `server.py` handed the renderer an
    `AVATAR_REFERENCE` environment variable. An operator could attach a face in the console,
    watch it prepare successfully, start a session, and get whatever the env var pointed at --
    with no error anywhere. That is the same failure class as a knowledge base the interviewer
    never consulted, and it is the last one of those.
    """
    llm_model: str = ""
    voice_id: str = ""
    face_id: str | None = None
    turn_taking: dict[str, Any] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        """Whether a stored agent was used at all, for `/config` to report honestly."""
        return self.agent_id is not None


def resolve_for_session(session_id: str | None, *, data: Store | None = None) -> ResolvedAgent:
    """
    Resolve configuration for a socket, preferring the session record over the environment.

    This is what makes the candidate's link load-bearing. The link carries a session id; the
    record names an agent; the agent names a knowledge base, a lexicon, a guardrail and a face.
    Nothing about the conversation then depends on how the server process was started.

    `AVATAR_AGENT` stays as a fallback, in that order, for two cases that both matter: running
    the prototype against no console data at all — which the README promises works — and pinning
    one agent for a scripted demo without minting a session first.

    An unknown session id falls through to the environment rather than raising. The runtime
    should not refuse to talk to a candidate because a record was deleted; it should hold the
    conversation on whatever default is configured, and the missing record is visible in the
    console rather than as a failed connection.
    """
    data = data or store
    if session_id:
        try:
            record = data.get("sessions", session_id)
        except NotFound:
            record = {}
        agent_id = record.get("agent_id")
        if agent_id:
            return resolve_agent(str(agent_id), data=data)
    return resolve_agent(data=data)


def resolve_agent(agent_id: str | None = None, *, data: Store | None = None) -> ResolvedAgent:
    """
    Load an agent and everything it references. Returns defaults when none is selected.

    Defaults rather than raising when `AVATAR_AGENT` is unset, because a clean clone with no
    stored data must still run — the README promises that, and every other boundary here keeps
    the same promise.
    """
    data = data or store
    chosen = agent_id or os.environ.get(AGENT_ENV) or ""
    if not chosen:
        return ResolvedAgent()

    try:
        record = data.get(COLLECTION, chosen)
    except NotFound as exc:
        raise AgentNotConfigured(
            f"{AGENT_ENV}={chosen!r} does not exist. Create it in the console, or unset "
            f"{AGENT_ENV} to run on defaults."
        ) from exc

    return ResolvedAgent(
        agent_id=chosen,
        name=str(record.get("name", "")),
        system_prompt=str(record.get("system_prompt") or ""),
        retriever=_load_knowledge(data, record.get("knowledge_base_ids") or [], chosen),
        lexicon=_load_lexicon(data, record.get("pronunciation_id"), chosen),
        guardrail=_load_guardrail(data, record.get("guardrail_id"), chosen),
        tools=_load_tools(data, record.get("tool_ids") or [], chosen),
        plan=_load_plan(data, record.get("rubric_id"), chosen),
        face_reference=_load_face(data, record.get("face_id"), chosen),
        llm_model=str(record.get("llm_model") or ""),
        voice_id=str(record.get("voice_id") or ""),
        face_id=record.get("face_id"),
        turn_taking=dict(record.get("turn_taking") or {}),
    )


def _load_knowledge(data: Store, kb_ids: list[str], agent_id: str) -> Retriever:
    """
    Index every attached knowledge base into one retriever.

    One retriever across all of them rather than one each: retrieval has to rank a job
    description's paragraphs against a rubric's on the same scale, and per-base retrievers would
    return the top hit from each regardless of whether the second was relevant at all.

    Document ids are namespaced by base, because `index` replaces by document id and two bases
    can legitimately hold a document with the same id.
    """
    if not kb_ids:
        return NullRetriever()

    retriever = build_retriever()
    indexed = 0
    for kb_id in kb_ids:
        try:
            base = data.get("knowledge", kb_id)
        except NotFound as exc:
            raise AgentNotConfigured(
                f"agent {agent_id!r} references knowledge base {kb_id!r}, which does not "
                "exist. An interviewer silently running without its context is worse than "
                "one that refuses to start."
            ) from exc
        for document in base.get("documents") or []:
            text = str(document.get("text") or "")
            if not text.strip():
                continue
            indexed += retriever.index(
                f"{kb_id}:{document.get('id', '')}",
                text,
                source=str(document.get("filename") or ""),
            )

    if not indexed:
        # Attached but empty. Not an error — an operator may have created the base before
        # uploading to it — but the caller reports it so it is visible rather than puzzling.
        return retriever
    return retriever


def _load_plan(data: Store, rubric_id: str | None, agent_id: str) -> InterviewPlan:
    """
    Load the rubric an agent interviews against, as an immutable plan.

    Ids are read from the stored record rather than re-derived from the name. The API stamps
    them at write time precisely so this cannot drift: re-slugging here would work until an
    operator renamed a competency, at which point past sessions' coverage would key off an id
    that no longer exists and their reports would silently lose an area. The fallback covers
    records written before ids were stamped.
    """
    if not rubric_id:
        return InterviewPlan()
    try:
        record = data.get("rubrics", rubric_id)
    except NotFound as exc:
        raise AgentNotConfigured(
            f"agent {agent_id!r} references rubric {rubric_id!r}, which does not exist. An "
            "interview running without the plan someone wrote would look fine and cover "
            "nothing in particular."
        ) from exc

    competencies = []
    for entry in record.get("competencies") or []:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        competencies.append(
            Competency(
                id=str(entry.get("id") or slug(name)),
                name=name,
                probe=str(entry.get("probe") or ""),
                signals=tuple(
                    str(signal).strip()
                    for signal in entry.get("signals") or []
                    if str(signal).strip()
                ),
                max_turns=int(entry.get("max_turns") or DEFAULT_MAX_TURNS),
                min_signals=int(entry.get("min_signals") or DEFAULT_MIN_SIGNALS),
                weight=float(entry.get("weight") or 1.0),
            )
        )
    return InterviewPlan(name=str(record.get("name") or ""), competencies=tuple(competencies))


def _load_face(data: Store, face_id: str | None, agent_id: str) -> str | None:
    """
    The reference path for an attached face, or `None` to fall back to the environment.

    Fatal when the face is missing, like every other reference: an interview that silently wears
    a different face than the one configured is worse than one that refuses to start, and the
    operator can see the reason.

    A face that has not been prepared is *not* fatal. `prepare_identity` runs at session start
    regardless -- preparation is a cache, not a gate -- so a `pending` face costs the first
    session the preprocessing time and works. Refusing here would turn a slow first session into
    a broken one. A `failed` face is different: whatever the renderer could not read then it
    will not read now, so that is worth stopping for.
    """
    if not face_id:
        return None
    try:
        record = data.get("faces", face_id)
    except NotFound as exc:
        raise AgentNotConfigured(
            f"agent {agent_id!r} references face {face_id!r}, which does not exist. An "
            "interview wearing a different face than the one configured is worse than one "
            "that refuses to start."
        ) from exc

    if str(record.get("status")) == "failed":
        raise AgentNotConfigured(
            f"agent {agent_id!r} references face {face_id!r}, whose preparation failed: "
            f"{record.get('failure_reason') or 'no reason recorded'}. Re-prepare it or attach "
            "another face."
        )
    reference = str(record.get("reference_path") or "").strip()
    return reference or None


def _load_lexicon(data: Store, lex_id: str | None, agent_id: str) -> list[tuple[str, str]]:
    if not lex_id:
        return []
    try:
        record = data.get("pronunciations", lex_id)
    except NotFound as exc:
        raise AgentNotConfigured(
            f"agent {agent_id!r} references pronunciation lexicon {lex_id!r}, which does "
            "not exist."
        ) from exc
    return [
        (str(entry.get("term", "")), str(entry.get("say", "")))
        for entry in record.get("entries") or []
        if str(entry.get("term", "")).strip()
    ]


def _load_guardrail(data: Store, guard_id: str | None, agent_id: str) -> Policy | None:
    """
    Load the policy an agent references, validated through the same model the API writes.

    Validating on read rather than trusting the file: a policy edited by hand on disk, or
    written by an older build, would otherwise reach enforcement with a missing field and fail
    mid-conversation. Better to refuse at session start, where an operator sees it.
    """
    if not guard_id:
        return None
    try:
        record = data.get("guardrails", guard_id)
    except NotFound as exc:
        raise AgentNotConfigured(
            f"agent {agent_id!r} references guardrail {guard_id!r}, which does not exist. "
            "An interview running without the policy someone wrote is worse than one that "
            "refuses to start."
        ) from exc
    # Annotated rather than returned bare: with pydantic absent under the CI install
    # `model_validate` is `Any`, and strict mypy rejects returning it as `Policy`.
    policy: Policy = Policy.model_validate(record)
    return policy


def _load_tools(data: Store, tool_ids: list[str], agent_id: str) -> list[Tool]:
    """
    Load the tools an agent may call. A missing one is fatal at session start.

    Fatal rather than skipped, because a tool the model is told about and cannot reach is worse
    than one it was never offered: it will try, get an error, and spend a round trip inside the
    turn discovering what the operator could have been told here.
    """
    if not tool_ids:
        return []
    records = []
    for tool_id in tool_ids:
        try:
            records.append(data.get("tools", tool_id))
        except NotFound as exc:
            raise AgentNotConfigured(
                f"agent {agent_id!r} references tool {tool_id!r}, which does not exist."
            ) from exc
    return tools_from_records(records)


def build_llm_with_tools(agent: ResolvedAgent) -> SentenceStream:
    """
    Build the interviewer, handing it an executor when the agent has tools.

    Here rather than in `server.py` so the server does not have to know that only one adapter
    supports tool calling. Anthropic's does not yet, and offering tools to an adapter that
    ignores them would be worse than not offering them -- the model would never call, and nobody
    would know why.
    """
    from avatar.llm_anthropic import build_llm
    from avatar.tools import ToolExecutor

    name = os.environ.get("AVATAR_LLM", "scripted")
    if name != "openai" or not agent.tools:
        return build_llm(name)

    from avatar.llm_openai import OpenAIInterviewer

    return OpenAIInterviewer(
        system=agent.system_prompt or OpenAIInterviewer().system,
        executor=ToolExecutor(agent.tools),
    )
