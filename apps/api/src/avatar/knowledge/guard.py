"""
Guardrail enforcement, as one more decorator on the `SentenceStream` boundary.

**Both directions live here, and that is the point.** The input check runs on the candidate's
transcript before it reaches the model; the output check runs on each generated sentence before
it reaches TTS. Wrapping the LLM boundary is what lets one decorator hold both, because that
boundary is the only place where the input has already been assembled and the output has not yet
been spoken. The orchestrator is unchanged again.

**The policy is evaluated by `avatar.api.guardrails.evaluate`, not reimplemented.** A second
copy would drift from the one the console's `/check` endpoint exercises, and then the button an
operator uses to test a policy would stop describing what the interview actually enforces —
which is worse than having no test button at all.

**Nothing here does I/O.** The output check sits between the model and the synthesiser, inside a
turn already measuring 2.7-5.8s against a sub-second target, so a network round trip at that
point would be self-defeating twice over: it delays the speech it is inspecting, and it fails
open or closed at exactly the wrong moment. Regex and string comparisons only.

**Output enforcement is per sentence, not per turn.** Sentences stream to TTS as they arrive —
that overlap is why sub-second turnaround is arguable at all — so waiting for a whole turn to
check it would mean either buffering the turn, which destroys the overlap, or checking after the
first sentence has already been spoken, which is not enforcement.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence

from avatar.api.guardrails import Policy, evaluate
from avatar.contracts import Message
from avatar.knowledge.augment import SentenceStreamLike, latest_candidate_text


def with_guardrail(llm: SentenceStreamLike, policy: Policy | None) -> SentenceStreamLike:
    """
    Wrap an LLM so both directions are policed.

    A `None` policy returns the stream untouched rather than a wrapper that always allows, so
    the unguarded path adds no call at all.

    On an input violation the model is never invoked. That saves the tokens, but the real reason
    is that a banned topic must not enter the conversation history: history is what every later
    turn is conditioned on, so admitting it once admits it for the rest of the interview.
    """
    if policy is None:
        return llm

    def guarded(history: Sequence[Message]) -> AsyncGenerator[str, None]:
        candidate = latest_candidate_text(history)
        inbound = evaluate(policy, candidate, "input") if candidate else None

        if inbound is not None and not inbound.allowed:
            return _refuse(policy, inbound.violations)

        # Redaction applies even when the turn is allowed: PII is not a violation to refuse
        # over, it is something that must not be sent to a third-party model or written into a
        # transcript. So the model sees the redacted text and history keeps the original, which
        # is the split a reviewer needs.
        outbound_history = list(history)
        if inbound is not None and inbound.redacted_text != candidate:
            outbound_history = _with_redacted_answer(history, inbound.redacted_text)

        return _policed(llm(outbound_history), policy)

    return guarded


async def _refuse(policy: Policy, violations: list[str]) -> AsyncGenerator[str, None]:
    """
    Speak the refusal instead of the model's answer.

    `on_violation` distinguishes what happens next, and the difference is a product decision
    rather than a technical one: `refuse` says no and waits, `redirect` says no and asks
    something else, `end_session` stops the interview. Only the first two can be expressed as
    text, so the third is spoken the same way and its enforcement belongs to the state machine —
    which does not have it yet. Saying so here rather than pretending the setting is honoured.
    """
    yield policy.refusal_message
    if policy.on_violation == "redirect":
        yield " Let us come back to the technical side — tell me about a system you have run."
    _ = violations  # recorded by the caller's telemetry, not needed for the text


async def _policed(
    stream: AsyncGenerator[str, None], policy: Policy
) -> AsyncGenerator[str, None]:
    """
    Pass sentences through, substituting the refusal on the first one that violates.

    Closing the wrapped stream on violation is deliberate and load-bearing: `aclose` is what
    aborts the provider's HTTP request, so a blocked turn stops being generated and stops being
    billed. Without it the model keeps producing text nobody will ever hear — the same leak the
    orchestrator's `aclosing` calls exist to prevent, one level out.
    """
    try:
        async for sentence in stream:
            verdict = evaluate(policy, sentence, "output")
            if not verdict.allowed:
                yield policy.refusal_message
                return
            # Redacted rather than original: this string is what reaches the synthesiser, and
            # an email read aloud is disclosed whether or not it was a refusable violation.
            yield verdict.redacted_text
    finally:
        await stream.aclose()


def _with_redacted_answer(history: Sequence[Message], redacted: str) -> list[Message]:
    """
    Copy the history with the last candidate turn's text replaced.

    A copy, not a mutation: the orchestrator owns that list and truncates it against audio the
    client acknowledged playing. Editing it here would mean the transcript a reviewer reads had
    been quietly rewritten by a policy, and the two records would disagree about what was said.
    """
    out = list(history)
    for index in range(len(out) - 1, -1, -1):
        if out[index].get("role") == "user":
            out[index] = {"role": "user", "content": redacted}
            break
    return out
