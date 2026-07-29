"""
Guardrail enforcement at the LLM boundary.

The tests that matter are the ones about *what the model is spared*. Refusing after the model
has already generated a banned answer is not enforcement, and neither is refusing after the
first sentence has been spoken — so the assertions are about whether the model was invoked and
whether its stream was closed, not only about the text that came back.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence

import pytest

pytest.importorskip("fastapi", reason="Policy lives beside the console router")

from avatar.api.guardrails import Policy
from avatar.contracts import Message
from avatar.knowledge.guard import with_guardrail

BANNED = "salary negotiation"


def policy(**overrides: object) -> Policy:
    base: dict[str, object] = {
        "name": "Interview policy",
        "banned_topics": [BANNED],
        "pii_redaction": True,
        "max_answer_chars": 200,
        "refusal_message": "Let us keep to the technical discussion.",
        "on_violation": "refuse",
    }
    base.update(overrides)
    return Policy.model_validate(base)


class SpyLlm:
    """Records whether it was called, with what, and whether its stream was closed."""

    def __init__(self, *sentences: str) -> None:
        self.sentences = sentences or ("A perfectly ordinary question.",)
        self.calls: list[Sequence[Message]] = []
        self.closed = False

    def __call__(self, history: Sequence[Message]) -> AsyncGenerator[str, None]:
        self.calls.append(list(history))
        spy = self

        async def stream() -> AsyncGenerator[str, None]:
            try:
                for sentence in spy.sentences:
                    yield sentence
            finally:
                spy.closed = True

        return stream()


async def drain(stream: AsyncGenerator[str, None]) -> str:
    return "".join([chunk async for chunk in stream])


def said(text: str) -> list[Message]:
    return [{"role": "user", "content": text}]


# -- no policy -------------------------------------------------------------


def test_no_policy_returns_the_stream_itself() -> None:
    """The unguarded path must add not even a wrapper call."""
    llm = SpyLlm()

    assert with_guardrail(llm, None) is llm


# -- input side ------------------------------------------------------------


async def test_a_banned_topic_never_reaches_the_model() -> None:
    """
    The assertion is `calls == []`, not the returned text.

    A banned topic must not enter conversation history, because history conditions every later
    turn — admitting it once admits it for the rest of the interview. Refusing after generation
    would leave it there.
    """
    llm = SpyLlm()
    guarded = with_guardrail(llm, policy())

    out = await drain(guarded(said(f"I want to discuss {BANNED}")))

    assert llm.calls == []
    assert out == "Let us keep to the technical discussion."


async def test_redirect_adds_a_follow_up_question() -> None:
    """`refuse` stops; `redirect` says no and keeps the interview moving. A product
    distinction, so it has to be visible in the output."""
    guarded = with_guardrail(SpyLlm(), policy(on_violation="redirect"))

    out = await drain(guarded(said(f"tell me about {BANNED}")))

    assert out.startswith("Let us keep to the technical discussion.")
    assert len(out) > len("Let us keep to the technical discussion.")


async def test_an_allowed_answer_reaches_the_model_unchanged() -> None:
    llm = SpyLlm("What did the incident cost you?")
    guarded = with_guardrail(llm, policy())

    out = await drain(guarded(said("we lost data for six hours")))

    assert llm.calls[0] == said("we lost data for six hours")
    assert out == "What did the incident cost you?"


async def test_pii_is_redacted_before_the_model_sees_it_but_is_not_refused() -> None:
    """
    PII is not a refusable offence — it is something that must not be sent to a third-party
    model. So the turn proceeds and the model receives the placeholder.
    """
    llm = SpyLlm()
    guarded = with_guardrail(llm, policy())

    await drain(guarded(said("reach me at someone@example.com")))

    handed = str(llm.calls[0][-1]["content"])
    assert "someone@example.com" not in handed
    assert "[redacted-email]" in handed


async def test_the_original_history_is_not_mutated_by_redaction() -> None:
    """
    The orchestrator owns that list and truncates it against acknowledged audio. Rewriting it
    here would mean the transcript a reviewer reads had been quietly edited by a policy, and the
    two records would disagree about what was said.
    """
    history = said("reach me at someone@example.com")
    guarded = with_guardrail(SpyLlm(), policy())

    await drain(guarded(history))

    assert history[0]["content"] == "reach me at someone@example.com"


async def test_a_first_turn_with_no_candidate_text_is_not_checked() -> None:
    """Nothing has been said, so there is nothing to police — and an empty-string check would
    match a banned topic against nothing and waste the pass."""
    llm = SpyLlm()
    guarded = with_guardrail(llm, policy())

    await drain(guarded([]))

    assert llm.calls == [[]]


# -- output side -----------------------------------------------------------


async def test_a_banned_sentence_is_replaced_and_the_stream_is_closed() -> None:
    """
    Closing is the load-bearing half. `aclose` aborts the provider's HTTP request, so a blocked
    turn stops being generated and stops being billed; without it the model keeps producing text
    nobody will hear.
    """
    llm = SpyLlm("Fine so far.", f"Now about {BANNED}.", "And another thing.")
    guarded = with_guardrail(llm, policy())

    out = await drain(guarded(said("an ordinary answer")))

    assert out == "Fine so far.Let us keep to the technical discussion."
    assert llm.closed, "the upstream generator must be closed, not abandoned"


async def test_an_over_long_answer_is_refused() -> None:
    """`max_answer_chars` is an output-side rule: it bounds what the avatar says, never what the
    candidate may say."""
    llm = SpyLlm("x" * 500)
    guarded = with_guardrail(llm, policy(max_answer_chars=100))

    assert await drain(guarded(said("ok"))) == "Let us keep to the technical discussion."


async def test_a_long_candidate_answer_is_never_refused_for_length() -> None:
    """The thing this product exists to collect is a long answer from the candidate."""
    llm = SpyLlm("A question.")
    guarded = with_guardrail(llm, policy(max_answer_chars=50))

    assert await drain(guarded(said("y" * 2000))) == "A question."


async def test_pii_in_the_models_output_is_redacted_before_it_is_spoken() -> None:
    """An email read aloud is disclosed whether or not it was a refusable violation."""
    llm = SpyLlm("Mail me at leak@example.com about it.")
    guarded = with_guardrail(llm, policy())

    out = await drain(guarded(said("ok")))

    assert "leak@example.com" not in out
    assert "[redacted-email]" in out


async def test_sentences_are_checked_individually_not_as_a_whole_turn() -> None:
    """
    Sentences stream to TTS as they arrive, and that overlap is why a sub-second turnaround is
    arguable at all. Buffering the turn to check it would destroy the overlap; checking after
    the fact would not be enforcement.
    """
    llm = SpyLlm("First.", "Second.", "Third.")
    guarded = with_guardrail(llm, policy())

    chunks = [chunk async for chunk in guarded(said("ok"))]

    assert chunks == ["First.", "Second.", "Third."], "each arrives separately"
