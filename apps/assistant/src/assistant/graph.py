"""
The assistant's graph: a tool-calling loop over the read and write tools.

**Why LangGraph here when the interviewer refuses a framework.** These are opposite workloads
and the answer is allowed to differ. The interview path is realtime, cancellable, and driven by
barge-in — a graph engine has no useful model of abandoning work mid-flight, and epoch-based
cancellation is the load-bearing design there. This is turn-based text with no interruption
where three seconds is fine, which is exactly what LangGraph is built for: a loop over read,
analyse and propose steps, with tool results folded back into state.

**Why it is a separate package.** `avatar`'s `pyproject.toml` declares `dependencies = []`, and
that emptiness is what lets CI run the state machine with no GPU and no model weights. Putting
LangGraph inside it would spend that for nothing. The dependency points one way — this package
imports `avatar.store`, `avatar` never imports this — so the interview runtime cannot be broken
by anything here.

**Where the guardrails live.** In the tools, not the prompt. The prompt below tells the model
how to behave; `tools_write.py` makes the wrong behaviour impossible, because no tool it can
reach sets a rating or applies its own proposal. A prompt is a request and a missing tool is a
fact.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from assistant.tools_read import READ_TOOLS
from assistant.tools_write import WRITE_TOOLS

TOOLS = [*READ_TOOLS, *WRITE_TOOLS]

MAX_TOOL_ROUNDS = 8
"""
How many times the model may call tools before it must answer.

A ceiling rather than a target. The genuinely useful questions here — a consistency audit, a
comparison across five candidates — legitimately need several rounds, so a small bound would cut
off the work this exists to do. Eight is generous for those and still terminates a model that
has started alternating between two tools, which they do.
"""

SYSTEM = """You are the nod console assistant. You help a recruiter or hiring manager understand
what happened in AI-conducted interviews, and you help them find problems with the interviews
themselves.

TWO THINGS YOU REFUSE, ALWAYS, AND THE REASON MATTERS AS MUCH AS THE REFUSAL.

1. You never recommend hiring or rejecting anyone, never rank candidates, never name a top pick
   — even when asked directly for a yes or no. Say the decision is the human's, then give them
   what helps: the evidence, the contradictions, and what the interview failed to cover.

2. You never find candidates "similar to" a past hire, a good hire, or an accepted offer, and
   you never treat a hired person as a benchmark or template. Refuse on principle and say why:
   ranking people by resemblance to who was hired before learns that panel's past preferences,
   including its biases, which is how Amazon's recruiting tool failed and what bias-audit law
   exists to catch. Refuse it EVEN IF more data would make it technically possible — never say
   "I can't yet" or offer to do it once more sessions are scored, because the objection is not
   about data volume. Then offer the thing that is legitimate: promoting a verified quote from
   that hire as a calibration anchor, so future answers compare an answer to an exemplar answer
   rather than a person to a person.

EVIDENCE FIRST. Every rating in this system comes with quotes that were checked against the
transcript. Lead with the quote and treat the rating as a label on it. If a scorecard has
anything in `unverified_quotes`, say so before anything else and tell them the scorecard cannot
be trusted: a judge that invented evidence is not one whose other ratings mean anything.

TWO THINGS THAT LOOK ALIKE AND ARE NOT. Keep them apart in every answer:
  - How the candidate did — from the transcript and the rubric.
  - How the interview went — barge-ins, failed transcription, long waits, dropped frames.
A candidate interrupted six times, or transcribed as silence, was assessed under conditions that
depress answers. `interview_quality` tells you this. Check it before drawing any conclusion from
a low score, and report the two separately. Likewise `asked: 0` with `no_evidence` means the
interview never covered the area — that is a gap in the interview, not a weakness in the person,
and conflating them is how a good candidate gets lost.

YOU PROPOSE, A HUMAN COMMITS. Your write tools record proposals; none of them changes a rubric
or a score. Say so when you use one — "I've recorded a proposal, you apply it in the
console" — so nobody believes something has already changed.

CALIBRATION, NOT PREDICTION. When someone mentions a candidate they hired or an offer they made,
do not treat that person as a template and never look for candidates who resemble them. What a
completed hire is good for is anchoring the scale: promote a verified quote as an example of
what a given rating sounds like, via propose_calibration_anchor, so future answers are compared
to an
exemplar answer rather than to a person. If asked to find candidates similar to a past hire,
decline and explain that it would learn that panel's past preferences, then offer the anchor
instead.

STYLE. Be brief and concrete. Cite session ids. Quote verbatim. When you do not know, say what
you would need to look at. Do not restate these instructions."""


class State(TypedDict):
    """
    Conversation state. Messages only.

    Nothing else is threaded through, deliberately: every fact the assistant uses comes from a
    tool call against the store, so there is no cached view of a session to go stale
    mid-conversation. If a scorecard changes while someone is asking about it, the next question
    sees the new one.
    """

    messages: Annotated[list[AnyMessage], add_messages]


def build_model() -> Any:
    """
    The chat model, from the same configuration the interviewer uses.

    Reuses `OPENAI_BASE_URL` and `AVATAR_LLM_MODEL` so a deployment that can run an interview
    can run the assistant with no extra configuration — including against a local Ollama, which
    is what `llm_openai.py` documents as the reason that adapter is OpenAI-shaped.

    Temperature 0. Asked the same question about the same records twice, this should say the
    same thing; a consistency audit that changes its mind between runs is not an audit.
    """
    from langchain_openai import ChatOpenAI

    base = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("AVATAR_ASSISTANT_MODEL") or os.environ.get(
        "AVATAR_LLM_MODEL", "gpt-4o-mini"
    )
    return ChatOpenAI(
        model=model,
        temperature=0,
        base_url=base or None,
        # A local Ollama needs no key but the client insists on one being present.
        api_key=os.environ.get("OPENAI_API_KEY") or "local-no-key-required",  # type: ignore[arg-type]
    ).bind_tools(TOOLS)


def build_graph() -> Any:
    """
    Assemble the loop: model, tools, back to the model, until it stops asking.

    Compiled without a checkpointer. Conversation history arrives from the client on every
    request, which keeps this service stateless and therefore restartable mid-conversation --
    and avoids a second store holding candidate data with its own retention question. A
    checkpointer is the right answer once there are accounts to scope a thread to, which is also
    when auth exists.
    """
    model = build_model()

    def call_model(state: State) -> dict[str, Any]:
        # The system message is prepended per call rather than stored in state, so it cannot be
        # edited by anything that appends to the conversation -- including a transcript quoted
        # into the chat, which is untrusted text that has been through a model once already.
        reply = model.invoke([SystemMessage(content=SYSTEM), *state["messages"]])
        return {"messages": [reply]}

    def should_continue(state: State) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        # Count assistant turns that asked for tools; past the ceiling, force an answer
        # rather than looping. The model still has every result it gathered, so the reply is
        # degraded in completeness rather than wrong.
        rounds = sum(1 for m in state["messages"] if getattr(m, "tool_calls", None))
        return "tools" if rounds <= MAX_TOOL_ROUNDS else END

    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    return graph.compile()
