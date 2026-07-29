"""
A `Judge` backed by an OpenAI-compatible chat completion.

**Why not reuse `OpenAIInterviewer`.** That class is a `SentenceStream`: it streams token deltas
and splits them into sentences, because a conversation needs the first sentence before the last
one exists. A judge needs one complete JSON object and nothing else, so streaming it would mean
reassembling a document from sentence fragments — a decision the interviewer makes for latency
reasons that the scorer has no reason to inherit.

**Why it is a separate module rather than a method.** This is reached only from the scoring
path, which runs after a session. Keeping it out of `llm_openai.py` means the interview's hot
path does not grow a code path it never takes, and it keeps `scoring.py` provider-blind: that
module takes a callable and never learns who answers.

**Temperature 0.** The same transcript should produce the same scorecard. A hiring record that
changes when re-scored is not evidence of anything, and a reviewer comparing two candidates
needs the variance to be in the answers rather than in the sampler.

The client construction is `llm_openai`'s, deliberately reused: it already handles a local
Ollama base URL with no key, a hosted key, and the error message that names both. A second copy
would drift, and the first thing to break would be the credential-free path a clean clone runs
on.
"""

from __future__ import annotations

import os

DEFAULT_MAX_TOKENS = 600
"""
Enough for a rating, two sentences, and three short quotes.

Small on purpose: the reply is a fixed-shape object, and a generous ceiling only buys room for
the preamble the prompt asks it not to write. Too small is visible as truncated JSON, which
`_extract_json` fails on and `judge_competency` records as an unassessed competency -- a loud
failure rather than a silently shortened rationale.
"""


def judge_model() -> str:
    """
    Which model judges. Follows the interviewer's model unless told otherwise.

    Two variables rather than one because the trade-offs genuinely differ: the interviewer is
    picked for time-to-first-token, and a judge running after the session has no such
    constraint, so a slower and more careful model is the obvious choice. Defaulting to the same
    one keeps a single-credential setup working without configuration.
    """
    from avatar.llm_openai import DEFAULT_MODEL

    return (
        os.environ.get("AVATAR_JUDGE_MODEL")
        or os.environ.get("AVATAR_LLM_MODEL")
        or DEFAULT_MODEL
    )


def build_judge() -> tuple[object, str] | None:
    """
    A judge callable and the model name, or `None` when no model is configured.

    `None` rather than an exception, because "no model available" is an expected state on a
    clean clone and the caller records it as an unavailable scorecard. Raising would make an
    ordinary configuration into an error the operator has to interpret.

    The model name comes back alongside the callable so the scorecard can say which model
    produced it. A rating without its author is not reviewable a month later, when the default
    has moved.
    """
    if os.environ.get("AVATAR_LLM", "scripted") not in ("openai", "anthropic"):
        # Following the interviewer's configuration rather than probing for credentials: a
        # deployment running the scripted interviewer is not one that wants real model calls
        # happening in the background.
        return None

    try:
        from avatar.llm_openai import _build_client

        client = _build_client()
    except Exception:
        # A missing SDK or absent key. Reported by the caller as unavailable with its reason,
        # which is more useful than a traceback in a background task nobody is watching.
        return None

    model = judge_model()

    async def judge(prompt: str) -> str:
        response = await client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            max_completion_tokens=DEFAULT_MAX_TOKENS,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(response.choices[0].message.content or "")

    return judge, model
