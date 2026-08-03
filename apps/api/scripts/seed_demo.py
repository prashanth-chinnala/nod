#!/usr/bin/env python3
"""
Fill a clean install with realistic data for every console page.

**Why this posts to the API instead of writing the store.** Seeding the store directly would be
faster and would prove nothing. Every record here goes through the same routers, the same
pydantic models and the same foreign keys an operator's click goes through, so a seed that
succeeds is evidence the create path works — and a seed that fails has found a real bug. It also
means this script cannot invent a shape the API would reject, which a direct writer silently
can.

**Nothing here is hardcoded into the application.** This is data, posted over HTTP, into
whatever store `AVATAR_STORE` selected. The voices are verified against Deepgram's own model
list at run time rather than typed from memory, because a voice id that 404s at synthesis time
is a defect that only shows up mid-interview.

**What it does not fabricate.** Faces and voices are built from real media in the MuseTalk
checkout — two different people, and speech that is actually speech. Where a persona would need
media we do not have, this seeds fewer of them rather than duplicating one person under several
names. The scored session is generated from the real scorer, not from a written-out score.

    python scripts/seed_demo.py                 # add to whatever is there
    python scripts/seed_demo.py --reset         # delete every record first
    python scripts/seed_demo.py --dry-run       # print the plan, touch nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from avatar.config import load_env  # noqa: E402  (path is set above)

load_env()
"""
The same `.env` files the server reads, read the same way.

Not decoration. Without this the script sees whatever the invoking shell happened to export,
which on a normal run is nothing -- so `DEEPGRAM_API_KEY` was absent, the voice list came back
empty, and every agent silently got the default voice while the console showed five
identically-voiced personas. Reading the environment through the project's own loader is the
difference between "uses the real configuration" and "uses whatever was lying around".
"""

API = os.environ.get("AVATAR_API", "http://127.0.0.1:8000")

VENDOR = ROOT / "vendor" / "MuseTalk" / "data"
"""
Where the seed media comes from.

The MuseTalk checkout is already required to run the real renderer, and it ships two clips of
two different people plus three speech recordings. Using those means this script adds no media
of its own to a repository that deliberately gitignores every `.mp4` and `.wav`.
"""


# -- transport --------------------------------------------------------------------------------


class ApiError(RuntimeError):
    pass


def call(
    method: str, path: str, body: Any = None, *, files: dict[str, Any] | None = None
) -> Any:
    """One API call. Raises with the server's own message, which is usually the actual problem."""
    url = f"{API}{path}"
    headers: dict[str, str] = {}
    data: bytes | None = None

    if files is not None:
        boundary = "----nodseed7a1c9f"
        parts: list[bytes] = []
        for name, value in files.items():
            if isinstance(value, Path):
                parts.append(
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                    f"filename=\"{value.name}\"\r\n"
                    f"Content-Type: application/octet-stream\r\n\r\n".encode()
                    + value.read_bytes()
                    + b"\r\n"
                )
            else:
                parts.append(
                    f"--{boundary}\r\nContent-Disposition: form-data; "
                    f"name=\"{name}\"\r\n\r\n{value}\r\n".encode()
                )
        parts.append(f"--{boundary}--\r\n".encode())
        data = b"".join(parts)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:600]
        raise ApiError(f"{method} {path} -> HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise ApiError(
            f"cannot reach the API at {API}: {exc.reason}. Start it with "
            "'python -m uvicorn avatar.server:app --app-dir apps/api/src'."
        ) from None


def voices_available() -> list[str]:
    """
    Aura voice ids, from Deepgram's own model list.

    Asked rather than remembered. A voice id typed from memory that no longer exists fails at
    synthesis time, mid-interview, with the agent silently falling back or erroring — the worst
    possible moment to discover a typo. If the key is missing this returns empty and the caller
    seeds one agent per available voice, which is zero, and says so.
    """
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not key:
        return []
    request = urllib.request.Request(
        "https://api.deepgram.com/v1/models", headers={"Authorization": f"Token {key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:  # network, auth, shape -- all the same answer here
        print(f"   !! could not list Deepgram voices ({type(exc).__name__}); using the default")
        return []
    return sorted(
        model["canonical_name"]
        for model in payload.get("tts", [])
        if model.get("canonical_name", "").startswith("aura-2-")
        and model.get("canonical_name", "").endswith("-en")
    )


# -- content ----------------------------------------------------------------------------------

KNOWLEDGE = [
    {
        "name": "Platform engineering handbook",
        "description": "How the platform team works: on-call, deploys, incident review, and the "
        "boundaries between services. What an interviewer may cite as fact.",
        "documents": [
            (
                "oncall.md",
                "On-call rotation. One primary and one secondary, rotating weekly on Wednesdays "
                "at 10:00 UTC. The primary owns acknowledgement within five minutes for a "
                "page-level alert and thirty minutes for a ticket-level one. The secondary is "
                "not a second pair of hands during business hours; the secondary exists so that "
                "a primary can sleep. Escalation to the secondary is expected, not a failure, "
                "and any page unacknowledged for ten minutes escalates automatically.\n\n"
                "Handover happens in writing in the rotation channel and covers three things: "
                "what is currently degraded, what was silenced and when it will un-silence, and "
                "anything a runbook did not cover. A handover that says 'quiet week' and nothing "
                "else is not a handover.",
            ),
            (
                "deploys.md",
                "Deployment policy. Every change ships behind a flag, default off. A deploy and "
                "a release are separate events and the distinction is load-bearing: deploying is "
                "moving code to production, releasing is turning a flag on for someone. We "
                "deploy many times a day and release deliberately.\n\n"
                "Rollback is a flag flip, not a redeploy, and it is expected to take under a "
                "minute. If a change cannot be rolled back by flipping a flag — a schema "
                "migration, a data backfill — it goes through a written plan that names the "
                "forward-only step and who approved it. Friday afternoon deploys are allowed; "
                "Friday afternoon releases of anything without a flag are not.",
            ),
            (
                "incident-review.md",
                "Incident review. Blameless, written within three working days, and the author "
                "is whoever held the pager, not whoever wrote the code. The review answers four "
                "questions: what did a customer experience, what was the sequence, what made it "
                "hard to diagnose, and what specifically changes.\n\n"
                "'More monitoring' is not an action item. An action item names a signal, a "
                "threshold and an owner. We track the count of open incident actions as a "
                "first-class metric because an unclosed action is a decision to accept the risk "
                "again, and that decision should be visible rather than implicit.",
            ),
            (
                "service-boundaries.md",
                "Service boundaries. A service owns its data and no other service reads its "
                "tables. Cross-service reads go through an interface the owner published and can "
                "version. This is enforced at review rather than by tooling, which means it "
                "erodes unless people argue about it.\n\n"
                "Shared libraries are permitted for pure functions and forbidden for anything "
                "holding state or performing I/O, because a shared client is a distributed "
                "deploy dependency wearing a library's clothes.",
            ),
        ],
    },
    {
        "name": "Data platform runbook",
        "description": "Pipelines, freshness contracts, backfills and the on-call surface for "
        "data. Cited when probing a data engineering candidate on specifics.",
        "documents": [
            (
                "freshness.md",
                "Freshness contracts. Every published dataset carries a freshness SLO and an "
                "owner, and both are in code beside the pipeline rather than in a spreadsheet. "
                "The default is four hours. A dataset without an SLO is not published; it is "
                "someone's scratch table that other people happen to read, which is the state we "
                "are trying to leave.\n\n"
                "A breach pages the owning team, not the data platform team. The platform team "
                "owns the machinery; the producing team owns the meaning. Conflating those is how "
                "a platform team ends up debugging business logic at 3am.",
            ),
            (
                "backfills.md",
                "Backfills. Any backfill touching more than one partition is written down before "
                "it runs, with the range, the expected row delta and the rollback. Backfills run "
                "at reduced concurrency against a separate pool so a backfill cannot starve "
                "production reads — this rule exists because it happened.\n\n"
                "Idempotency is a requirement, not an aspiration: a backfill that is not safe to "
                "re-run is a backfill that will be re-run by someone who does not know that.",
            ),
            (
                "late-arriving-data.md",
                "Late-arriving data. Events can arrive hours after their event time and the "
                "warehouse must not silently drop them. Windows are keyed on event time with a "
                "bounded allowed lateness, and anything past the bound goes to a quarantine "
                "table that is monitored rather than to /dev/null.\n\n"
                "The count of quarantined rows is a dashboard, because a silent drop looks "
                "exactly like correct behaviour until someone asks why a number moved.",
            ),
        ],
    },
    {
        "name": "Company and role context",
        "description": "What the company does, how the team is structured, and what this role is "
        "accountable for. Keeps an interviewer's framing answers consistent.",
        "documents": [
            (
                "about.md",
                "We build interview infrastructure: a real-time conversational avatar that "
                "conducts a structured technical interview and produces a defensible, evidence "
                "linked assessment. Our customers are engineering organisations that interview "
                "at volume and do not want a hiring decision resting on whoever happened to be "
                "free that afternoon.\n\n"
                "The product is judged on two things: whether the conversation feels like a "
                "conversation, and whether the resulting assessment survives being argued with.",
            ),
            (
                "team.md",
                "Engineering is organised in three groups. Platform owns the runtime, the media "
                "plane and deployment. Product owns the console, the interview room and the "
                "reporting surface. Applied ML owns the rendering pipeline, the scoring models "
                "and the evaluation harness.\n\n"
                "Groups are deliberately small and own their on-call. A group that cannot be "
                "paged for what it builds does not really own it.",
            ),
        ],
    },
]

RUBRICS = [
    {
        "name": "Senior backend engineer",
        "description": "Depth on distributed systems, judgment under production pressure, and "
        "whether they can be specific about work they actually did.",
        "competencies": [
            {
                "name": "Distributed systems depth",
                "probe": "Walk me through a system you built that had to stay correct while "
                "something downstream was failing. What did you assume, and which assumption "
                "turned out to be wrong?",
                "signals": [
                    "names a specific consistency or ordering guarantee and what relied on it",
                    "distinguishes retries from idempotency rather than treating them as one",
                    "describes a failure mode they did not anticipate, and how they found it",
                    "can say what they would do differently without being asked",
                ],
                "max_turns": 4,
                "min_signals": 2,
                "weight": 1.4,
            },
            {
                "name": "Production judgment",
                "probe": "Tell me about a time you had to choose between shipping and being "
                "confident. How did you decide, and what did the decision cost?",
                "signals": [
                    "names the risk they accepted, not just the risk they avoided",
                    "describes a rollback or a mitigation that actually existed beforehand",
                    "separates what they controlled from what they escalated",
                ],
                "max_turns": 3,
                "min_signals": 2,
                "weight": 1.2,
            },
            {
                "name": "Debugging under uncertainty",
                "probe": "Describe the hardest bug you have diagnosed. Not the most impactful — "
                "the hardest to see.",
                "signals": [
                    "describes the hypothesis they held and how it was disproved",
                    "used a measurement rather than reasoning alone to narrow it",
                    "can state why it was hard to see, not only what it was",
                ],
                "max_turns": 3,
                "min_signals": 2,
                "weight": 1.3,
            },
            {
                "name": "Communication and specificity",
                "probe": "Explain that same system to someone who will maintain it but was not "
                "there when it was built.",
                "signals": [
                    "chooses a level of detail suited to the listener",
                    "uses concrete numbers or names rather than 'high scale' and 'a lot'",
                    "flags what they are unsure about",
                ],
                "max_turns": 2,
                "min_signals": 1,
                "weight": 1.0,
            },
        ],
    },
    {
        "name": "Data engineer",
        "description": "Pipeline correctness, freshness thinking, and comfort with the parts of "
        "data work that are unglamorous and load-bearing.",
        "competencies": [
            {
                "name": "Pipeline correctness",
                "probe": "How do you know a pipeline you own is producing correct output today, "
                "not just that it ran?",
                "signals": [
                    "distinguishes 'the job succeeded' from 'the data is right'",
                    "names a specific check with a threshold and an owner",
                    "has an opinion on where to put the check and why",
                ],
                "max_turns": 3,
                "min_signals": 2,
                "weight": 1.4,
            },
            {
                "name": "Late and out-of-order data",
                "probe": "What happens in your pipeline when an event arrives six hours after "
                "its event time?",
                "signals": [
                    "separates event time from processing time without prompting",
                    "names a bounded lateness policy rather than 'we handle it'",
                    "says what happens to data past the bound",
                ],
                "max_turns": 3,
                "min_signals": 2,
                "weight": 1.3,
            },
            {
                "name": "Backfill discipline",
                "probe": "Talk me through the last large backfill you ran.",
                "signals": [
                    "idempotency was a property of the job, not a hope",
                    "isolated the backfill from production read capacity",
                    "had a way to verify the result other than 'no errors'",
                ],
                "max_turns": 3,
                "min_signals": 1,
                "weight": 1.1,
            },
        ],
    },
    {
        "name": "Engineering manager",
        "description": "Whether they own outcomes, give real feedback, and can describe a "
        "decision that was unpopular and right — or popular and wrong.",
        "competencies": [
            {
                "name": "Ownership of outcomes",
                "probe": "Tell me about something your team shipped that did not work. What was "
                "your part in that?",
                "signals": [
                    "describes their own decision, not only the team's execution",
                    "names what they would change in how they led, not just what was built",
                    "does not attribute the outcome entirely upward or downward",
                ],
                "max_turns": 4,
                "min_signals": 2,
                "weight": 1.5,
            },
            {
                "name": "Feedback and performance",
                "probe": "Describe the most difficult piece of feedback you have given. What did "
                "you say, in what words?",
                "signals": [
                    "can quote roughly what they actually said",
                    "describes the outcome, including if it did not land",
                    "separates behaviour from character in how they framed it",
                ],
                "max_turns": 3,
                "min_signals": 2,
                "weight": 1.3,
            },
            {
                "name": "Technical credibility",
                "probe": "What is a technical decision your team made that you disagreed with, "
                "and how did you handle it?",
                "signals": [
                    "engaged with the substance rather than only the process",
                    "can state the other side's strongest argument",
                    "was willing to be overruled and says so",
                ],
                "max_turns": 3,
                "min_signals": 1,
                "weight": 1.1,
            },
        ],
    },
    {
        "name": "Frontend engineer",
        "description": "State, rendering behaviour under real conditions, and whether "
        "accessibility is a habit or a checklist.",
        "competencies": [
            {
                "name": "State and data flow",
                "probe": "Describe a piece of UI where the state got away from you. What was the "
                "shape of the problem?",
                "signals": [
                    "identifies where state was duplicated or derived twice",
                    "distinguishes server state from UI state",
                    "names the fix in terms of ownership, not libraries",
                ],
                "max_turns": 3,
                "min_signals": 2,
                "weight": 1.3,
            },
            {
                "name": "Real-world rendering",
                "probe": "What did you do the last time an interface was fast on your machine and "
                "slow for users?",
                "signals": [
                    "measured on a representative device or network, not locally",
                    "names a specific cost — bundle, layout, re-render, image weight",
                    "knows what improved and by how much",
                ],
                "max_turns": 3,
                "min_signals": 2,
                "weight": 1.2,
            },
            {
                "name": "Accessibility as practice",
                "probe": "How do you know whether the thing you shipped last week is usable with "
                "a keyboard and a screen reader?",
                "signals": [
                    "describes something they actually do, routinely",
                    "distinguishes automated checks from real assistive-tech testing",
                    "treats it as part of done rather than a later pass",
                ],
                "max_turns": 2,
                "min_signals": 1,
                "weight": 1.0,
            },
        ],
    },
]

GUARDRAILS = [
    {
        "name": "Standard interview guardrail",
        "banned_topics": [
            "salary history",
            "current compensation",
            "age",
            "marital status",
            "pregnancy or family plans",
            "religion",
            "political affiliation",
            "national origin",
            "disability or health history",
            "sexual orientation",
            "criminal record",
            "trade secrets from a current employer",
        ],
        "pii_redaction": True,
        "max_answer_chars": 700,
        "refusal_message": "That is not something I should ask about, and it would not tell me "
        "anything about your engineering. Let us go back to the work itself.",
        "on_violation": "redirect",
    },
    {
        "name": "Strict — regulated hiring",
        "banned_topics": [
            "salary history",
            "current compensation",
            "age",
            "date of birth",
            "marital status",
            "pregnancy or family plans",
            "religion",
            "political affiliation",
            "national origin",
            "immigration or visa status",
            "disability or health history",
            "sexual orientation",
            "criminal record",
            "union membership",
            "military discharge status",
            "credit history",
        ],
        "pii_redaction": True,
        "max_answer_chars": 500,
        "refusal_message": "I am not able to discuss that. If you would like to raise it, please "
        "contact the hiring team directly rather than through this interview.",
        "on_violation": "refuse",
    },
    {
        "name": "Practice mode — permissive",
        "banned_topics": ["trade secrets from a current employer"],
        "pii_redaction": False,
        "max_answer_chars": 1200,
        "refusal_message": "Let us keep your current employer's confidential details out of this "
        "— tell me about your own reasoning instead.",
        "on_violation": "redirect",
    },
]

PRONUNCIATIONS = [
    {
        "name": "Engineering vocabulary",
        "entries": [
            {"term": "Kubernetes", "say": "koo-ber-NET-eez"},
            {"term": "kubectl", "say": "koob-CONTROL"},
            {"term": "PostgreSQL", "say": "POST-gres-cue-ell"},
            {"term": "psql", "say": "P-S-Q-L"},
            {"term": "Kafka", "say": "KAF-kuh"},
            {"term": "nginx", "say": "engine-X"},
            {"term": "Redis", "say": "RED-iss"},
            {"term": "SQLite", "say": "S-Q-L-ite"},
            {"term": "Ansible", "say": "AN-sih-bull"},
            {"term": "Grafana", "say": "gruh-FAH-nuh"},
            {"term": "Terraform", "say": "TERRA-form"},
            {"term": "Istio", "say": "ISS-tee-oh"},
            {"term": "Ceph", "say": "SEFF"},
            {"term": "YAML", "say": "YAM-ull"},
            {"term": "JWT", "say": "JOT"},
            {"term": "OAuth", "say": "OH-auth"},
            {"term": "gRPC", "say": "gee-R-P-C"},
            {"term": "GraphQL", "say": "graph-Q-L"},
            {"term": "WebRTC", "say": "web-R-T-C"},
            {"term": "SIGKILL", "say": "sig-KILL"},
        ],
    },
    {
        "name": "Data and ML vocabulary",
        "entries": [
            {"term": "PyTorch", "say": "pie-TORCH"},
            {"term": "NumPy", "say": "NUM-pie"},
            {"term": "SciPy", "say": "SIGH-pie"},
            {"term": "Pandas", "say": "PAN-dus"},
            {"term": "Parquet", "say": "par-KAY"},
            {"term": "Avro", "say": "AV-roh"},
            {"term": "dbt", "say": "D-B-T"},
            {"term": "Airflow", "say": "AIR-flow"},
            {"term": "Flink", "say": "FLINK"},
            {"term": "Presto", "say": "PRESS-toh"},
            {"term": "Trino", "say": "TREE-noh"},
            {"term": "ClickHouse", "say": "CLICK-house"},
            {"term": "Iceberg", "say": "ICE-berg"},
            {"term": "CUDA", "say": "KOO-duh"},
            {"term": "cuDNN", "say": "coo-D-N-N"},
            {"term": "ONNX", "say": "ON-icks"},
            {"term": "TensorRT", "say": "TENSOR-R-T"},
            {"term": "ROC AUC", "say": "rock A-U-C"},
        ],
    },
    {
        "name": "Product and company names",
        "entries": [
            {"term": "nod", "say": "NOD"},
            {"term": "Exterview", "say": "EX-ter-view"},
            {"term": "LiveKit", "say": "LIVE-kit"},
            {"term": "Deepgram", "say": "DEEP-gram"},
            {"term": "MuseTalk", "say": "MUSE-talk"},
            {"term": "LivePortrait", "say": "LIVE-portrait"},
            {"term": "Chroma", "say": "KROH-muh"},
            {"term": "uvicorn", "say": "OO-vee-corn"},
            {"term": "FastAPI", "say": "fast-A-P-I"},
            {"term": "pydantic", "say": "pie-DAN-tick"},
        ],
    },
]

TOOLS = [
    {
        "name": "record_competency_signal",
        "kind": "builtin",
        "description": "Record that the candidate demonstrated a specific signal for a "
        "competency, with the quote that evidences it. Call this the moment you hear the "
        "evidence, not at the end of the interview.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "competency": {
                    "type": "string",
                    "description": "The competency name exactly as it appears in the rubric",
                },
                "signal": {
                    "type": "string",
                    "description": "Which signal was demonstrated",
                },
                "quote": {
                    "type": "string",
                    "description": "The candidate's own words that evidence it, verbatim",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["weak", "clear", "strong"],
                    "description": "How directly the quote evidences the signal",
                },
            },
            "required": ["competency", "signal", "quote"],
        },
        "enabled": True,
    },
    {
        "name": "flag_for_human_review",
        "kind": "builtin",
        "description": "Flag this interview for a human to review before any decision is made. "
        "Use it when something happened that a score cannot represent — a claim you could not "
        "probe, a possible misunderstanding, or a candidate in obvious difficulty.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "What a reviewer needs to know, in one or two sentences",
                },
                "severity": {
                    "type": "string",
                    "enum": ["note", "concern", "blocking"],
                },
            },
            "required": ["reason"],
        },
        "enabled": True,
    },
    {
        "name": "note_followup_question",
        "kind": "builtin",
        "description": "Record a question worth asking in a later round that this interview did "
        "not have time for. Does not affect the score.",
        "parameters_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "why": {
                    "type": "string",
                    "description": "What answering it would tell a later interviewer",
                },
            },
            "required": ["question"],
        },
        "enabled": True,
    },
    {
        "name": "lookup_role_requirement",
        "kind": "http",
        "description": "Look up the published requirement for the role being interviewed for, so "
        "framing answers stay consistent with what was advertised.",
        "url": "http://127.0.0.1:8000/healthz",
        "timeout_ms": 1500,
        "parameters_schema": {
            "type": "object",
            "properties": {
                "role": {"type": "string"},
                "field": {
                    "type": "string",
                    "enum": ["level", "scope", "must_have", "nice_to_have"],
                },
            },
            "required": ["role"],
        },
        "enabled": False,
    },
]


def agent_specs(voices: list[str]) -> list[dict[str, Any]]:
    """
    The interviewer personas.

    Each gets a distinct voice, taken from the list Deepgram actually returned rather than typed
    from memory. If the list is short the extras fall back to the configured default, which is a
    visible, harmless degradation rather than a 404 at synthesis time.
    """

    def voice(name: str) -> str:
        return name if name in voices else (voices[0] if voices else "aura-2-thalia-en")

    return [
        {
            "name": "Senior backend interviewer",
            "voice_id": voice("aura-2-thalia-en"),
            "rubric": "Senior backend engineer",
            "guardrail": "Standard interview guardrail",
            "pronunciation": "Engineering vocabulary",
            "knowledge": ["Platform engineering handbook", "Company and role context"],
            "tools": [
                "record_competency_signal",
                "flag_for_human_review",
                "note_followup_question",
            ],
            "turn_taking": {"end_of_turn_silence_ms": 700},
            "system_prompt": "You are a senior engineer conducting a technical interview. You "
            "have done this often enough to be calm about it.\n\nAsk one question at a time and "
            "then stop talking. When an answer is general, ask for the specific instance — a "
            "system, a number, a date, a decision someone actually made. Do not accept 'we' when "
            "you need 'I'; ask what their own part was, once, without labouring it.\n\nYou are "
            "allowed to say you do not understand, and you should when it is true. If the "
            "candidate corrects you, take the correction. Never ask two questions in one turn, "
            "and never summarise their answer back to them at length — it wastes the time they "
            "could be using to talk.\n\nRecord each signal with the tool as you hear it, quoting "
            "their words rather than your paraphrase.",
        },
        {
            "name": "Data engineering interviewer",
            "voice_id": voice("aura-2-apollo-en"),
            "rubric": "Data engineer",
            "guardrail": "Standard interview guardrail",
            "pronunciation": "Data and ML vocabulary",
            "knowledge": ["Data platform runbook", "Company and role context"],
            "tools": ["record_competency_signal", "flag_for_human_review"],
            "turn_taking": {"end_of_turn_silence_ms": 800},
            "system_prompt": "You interview data engineers. You care about correctness more than "
            "about tool names, and you have seen enough pipelines to know that the interesting "
            "part is what happens when the data is wrong.\n\nPush past the happy path. If someone "
            "describes a pipeline, ask what it does when an event arrives late, when a schema "
            "changes without warning, or when the same batch is delivered twice. Ask how they "
            "would know it was wrong, not whether they think it is right.\n\nGive people room to "
            "think — a longer silence is fine and you should not fill it. Ask one question, then "
            "wait.",
        },
        {
            "name": "Engineering manager interviewer",
            "voice_id": voice("aura-2-athena-en"),
            "rubric": "Engineering manager",
            "guardrail": "Strict — regulated hiring",
            "pronunciation": "Product and company names",
            "knowledge": ["Company and role context", "Platform engineering handbook"],
            "tools": [
                "record_competency_signal",
                "flag_for_human_review",
                "note_followup_question",
            ],
            "turn_taking": {"end_of_turn_silence_ms": 900},
            "system_prompt": "You interview engineering managers. Your job is to find out what "
            "this person actually did, as distinct from what happened around them while they were "
            "present.\n\nAsk for specific episodes, not philosophy. When someone answers with a "
            "principle, ask for the time they applied it and what it cost. When someone describes "
            "a success, ask what they would change. When someone describes a failure, listen for "
            "whether they place themselves inside it.\n\nBe warm and unhurried — people give "
            "worse answers when they feel examined. Allow longer silences than feel comfortable; "
            "managers often say the true thing on the second attempt.",
        },
        {
            "name": "Frontend interviewer",
            "voice_id": voice("aura-2-andromeda-en"),
            "rubric": "Frontend engineer",
            "guardrail": "Standard interview guardrail",
            "pronunciation": "Engineering vocabulary",
            "knowledge": ["Company and role context"],
            "tools": ["record_competency_signal", "note_followup_question"],
            "turn_taking": {"end_of_turn_silence_ms": 650},
            "system_prompt": "You interview frontend engineers. You are interested in what users "
            "experienced, not in which framework was used.\n\nWhen someone names a library, ask "
            "what problem it solved and what it cost. When someone says an interface was fast, "
            "ask how they know and on what device. Ask about keyboard and screen-reader use as a "
            "normal engineering question, in the same tone as any other — not as a "
            "compliance check.\n\nOne question per turn. Keep your own turns short; you are here "
            "to listen.",
        },
        {
            "name": "Practice run — friendly",
            "voice_id": voice("aura-2-aries-en"),
            "rubric": "Senior backend engineer",
            "guardrail": "Practice mode — permissive",
            "pronunciation": "Engineering vocabulary",
            "knowledge": ["Company and role context"],
            "tools": ["note_followup_question"],
            "turn_taking": {"end_of_turn_silence_ms": 1000},
            "system_prompt": "This is a practice interview and the candidate knows it. Your job "
            "is to help them get better at being interviewed, not to assess them.\n\nAsk real "
            "questions, but when an answer is weak say so kindly and specifically, and give them "
            "another go at it. Tell them what a stronger version of their answer would have "
            "included. Be generous with silence — this is the one place where waiting too long "
            "costs nothing.",
        },
    ]


FACES = [
    ("Interviewer — studio video A", VENDOR / "video" / "sun.mp4"),
    ("Interviewer — studio video B", VENDOR / "video" / "yongen.mp4"),
]

VOICE_SAMPLES = [
    ("Reference recording — speaker A", VENDOR / "audio" / "sun.wav"),
    ("Reference recording — speaker B", VENDOR / "audio" / "yongen.wav"),
    ("Reference recording — English sample", VENDOR / "audio" / "eng.wav"),
]

CANDIDATES = [
    {
        "name": "Aparna Rao",
        "email": "aparna.rao@example.com",
        "role": "Senior Backend Engineer",
        "notes": "Referred by the payments team. Probe the ledger and ordering work.",
        "agent": "Senior backend interviewer",
        "resume": "aparna-rao.md",
        "resume_text": """# Aparna Rao
Senior Backend Engineer — Bengaluru

## Experience

**Staff Engineer, Ledger Platform (2022–present)**
Owned the double-entry ledger behind payouts. Moved settlement from a nightly batch to streaming
reconciliation, cutting the window from 14 hours to under 4 minutes. Introduced a per-account
sequence number so correctness stopped depending on Kafka partition ordering, after a rebalance
caused stale balances to overwrite fresh ones.

**Senior Engineer, Payments (2019–2022)**
Built idempotent payment capture across three providers. Designed the retry and dedupe layer and
led the payments on-call rotation. Duplicate-charge incidents went from roughly two a month to
zero over two quarters.

## Skills
Go, Python, PostgreSQL, Kafka, Kubernetes, Terraform, gRPC
""",
    },
    {
        "name": "Daniel Okonkwo",
        "email": "d.okonkwo@example.com",
        "role": "Data Engineer",
        "notes": "Strong on batch, unproven on streaming. Push on late-arriving events.",
        "agent": "Data engineering interviewer",
        "resume": "daniel-okonkwo.md",
        "resume_text": """# Daniel Okonkwo
Data Engineer — Lagos

## Experience

**Data Engineer, Analytics Platform (2021–present)**
Own 40+ dbt models feeding revenue reporting. Cut the nightly run from 6 hours to 90 minutes by
replacing full refreshes with incremental models. Introduced freshness SLOs per dataset with the
producing team as owner rather than the platform team.

**Analyst turned Engineer (2018–2021)**
Started in analytics, moved into engineering after automating a reporting pack that took three
days a month. Built the first Airflow deployment and the on-call runbook for it.

## Skills
Python, SQL, dbt, Airflow, Snowflake, Spark, Kafka Connect
""",
    },
    {
        "name": "Mei Lin Chen",
        "email": "meilin.chen@example.com",
        "role": "Engineering Manager",
        "notes": "Second-time manager. Ask about the reorg she described in the screen.",
        "agent": "Engineering manager interviewer",
        "resume": "mei-lin-chen.md",
        "resume_text": """# Mei Lin Chen
Engineering Manager — Singapore

## Experience

**Engineering Manager, Platform (2022–present)**
Two teams, 11 engineers. Took over a group that had missed three consecutive quarterly commitments
and reset scope with the product lead rather than adding people. Introduced written incident
reviews and moved on-call ownership from a central team to the producing teams.

**Tech Lead, Infrastructure (2019–2022)**
Led the migration off a single shared Postgres to per-service databases. Wrote the cutover plan and
the rollback; ran it over five months with no customer-visible incident.

## Skills
Team building, incident practice, platform strategy. Still writes Go on Fridays.
""",
    },
    {
        "name": "Tom Whitfield",
        "email": "tom.whitfield@example.com",
        "role": "Frontend Engineer",
        "notes": "No resume on file yet — invited before it arrived.",
        "agent": "Frontend interviewer",
        "resume": None,
        "resume_text": None,
    },
]

TRANSCRIPT = [
    (
        "We had a queue-backed ingest that assumed events for one account arrived in order. It "
        "did for about eighteen months, then a partition rebalance broke it and we were writing "
        "stale balances over fresh ones.",
        "How did you find out the ordering assumption had broken?",
    ),
    (
        "A customer noticed before we did, which is the part I still think about. Our monitoring "
        "watched consumer lag and error rate, and both were healthy — the pipeline was happily "
        "processing the wrong thing in the wrong order.",
        "What did you change, and what did you deliberately not change?",
    ),
    (
        "We added a per-account sequence number and made the writer reject anything older than "
        "what it had already applied, so correctness stopped depending on delivery order. We did "
        "not move to a single partition per account — that would have fixed it too and capped our "
        "throughput at one consumer per account.",
        "You said a customer found it first. What did you do about the monitoring?",
    ),
    (
        "We added a check on the data rather than on the machinery: a daily reconciliation "
        "against the source of truth, alerting on any account whose balance disagreed. It runs at "
        "four in the morning and it has paged us twice since, both times for real problems.",
        "If you were doing it again, what would you do differently?",
    ),
    (
        "I would have written down the ordering assumption when we made it. It was in someone's "
        "head for eighteen months. The sequence number is two hours of work — the expensive part "
        "was nobody knowing the assumption existed to question it.",
        "That is a good place to stop. Thank you for your time.",
    ),
]


# -- seeding ----------------------------------------------------------------------------------


def reset() -> None:
    """
    Delete every record, in dependency order so a foreign key never blocks a delete.

    **Sessions go through the store, not the API, and that is the one deliberate exception to
    this script's rule.** `/sessions` has no DELETE on purpose: immutability is what makes an
    interview record evidence, and `test_there_is_no_way_to_edit_or_delete_a_session` asserts
    the absence of the route so it reads as a decision rather than an oversight. Resetting a
    development install is a maintenance operation, not a product one, so it reaches past the
    API here — visibly, in one place, in a script named for seeding a demo — instead of arguing
    for an endpoint that would weaken the product to serve a developer.
    """
    print("-- resetting")

    from avatar.store import store as data

    sessions = data.list("sessions")
    for record in sessions:
        data.delete("sessions", record["id"])
    print(f"   {len(sessions)} sessions deleted (via the store: the API has no DELETE, by design)")

    for collection, label in (
        ("candidates", "candidates"),
        ("agents", "agents"),
        ("faces", "faces"),
        ("voices", "voices"),
        ("knowledge", "knowledge bases"),
        ("rubrics", "rubrics"),
        ("guardrails", "guardrails"),
        ("pronunciations", "pronunciations"),
        ("tools", "tools"),
    ):
        records = call("GET", f"/{collection}") or []
        removed = 0
        for record in records:
            try:
                call("DELETE", f"/{collection}/{record['id']}")
                removed += 1
            except ApiError as exc:
                print(f"   !! could not delete {collection}/{record['id']}: {exc}")
        print(f"   {removed} {label} deleted")


def seed(dry_run: bool = False) -> dict[str, Any]:
    made: dict[str, Any] = {}

    voices = voices_available()
    print(f"-- {len(voices)} English Aura voices available from Deepgram")

    if dry_run:
        print("\n-- dry run: nothing was written")
        print(f"   {len(KNOWLEDGE)} knowledge bases, "
              f"{sum(len(k['documents']) for k in KNOWLEDGE)} documents")
        print(f"   {len(RUBRICS)} rubrics, {len(GUARDRAILS)} guardrails, "
              f"{len(PRONUNCIATIONS)} lexicons, {len(TOOLS)} tools")
        print(f"   {len(FACES)} faces, {len(VOICE_SAMPLES)} voice references, "
              f"{len(agent_specs(voices))} agents")
        return made

    print("-- knowledge bases")
    made["knowledge"] = {}
    for spec in KNOWLEDGE:
        kb = call("POST", "/knowledge",
                  {"name": spec["name"], "description": spec["description"]})
        for filename, body in spec["documents"]:
            call("POST", f"/knowledge/{kb['id']}/documents",
                 {"filename": filename, "text": body})
        # Embedding is a separate call on purpose: it is the slow, failable part, and a
        # knowledge base with documents but no index is a real state the console has to show.
        try:
            report = call("POST", f"/knowledge/{kb['id']}/embed", {})
            chunks = (report or {}).get("chunks", "?")
        except ApiError as exc:
            chunks = f"embed failed: {exc}"
        made["knowledge"][spec["name"]] = kb["id"]
        print(f"   {spec['name']}: {len(spec['documents'])} docs, {chunks} chunks")

    print("-- rubrics")
    made["rubrics"] = {}
    for spec in RUBRICS:
        rubric = call("POST", "/rubrics", spec)
        made["rubrics"][spec["name"]] = rubric["id"]
        print(f"   {spec['name']}: {len(spec['competencies'])} competencies")

    print("-- guardrails")
    made["guardrails"] = {}
    for spec in GUARDRAILS:
        guardrail = call("POST", "/guardrails", spec)
        made["guardrails"][spec["name"]] = guardrail["id"]
        print(f"   {spec['name']}: {len(spec['banned_topics'])} banned topics, "
              f"on_violation={spec['on_violation']}")

    print("-- pronunciations")
    made["pronunciations"] = {}
    for spec in PRONUNCIATIONS:
        lexicon = call("POST", "/pronunciations", spec)
        made["pronunciations"][spec["name"]] = lexicon["id"]
        print(f"   {spec['name']}: {len(spec['entries'])} entries")

    print("-- tools")
    made["tools"] = {}
    for spec in TOOLS:
        tool = call("POST", "/tools", spec)
        made["tools"][spec["name"]] = tool["id"]
        state = "enabled" if spec.get("enabled", True) else "disabled"
        print(f"   {spec['name']} ({spec['kind']}, {state})")

    print("-- faces")
    made["faces"] = {}
    for name, path in FACES:
        if not path.is_file():
            print(f"   !! skipping {name}: no media at {path}")
            continue
        face = call("POST", "/faces/upload", files={"name": name, "file": path})
        made["faces"][name] = face["id"]
        print(f"   {name}: {face.get('source_kind')} "
              f"{face.get('width')}x{face.get('height')}, "
              f"{face.get('duration_seconds')}s, status={face.get('status')}")

    print("-- voice references")
    made["voices"] = {}
    for name, path in VOICE_SAMPLES:
        if not path.is_file():
            print(f"   !! skipping {name}: no media at {path}")
            continue
        try:
            voice = call("POST", "/voices/upload", files={"name": name, "file": path})
        except ApiError as exc:
            print(f"   !! {name}: {exc}")
            continue
        made["voices"][name] = voice["id"]
        print(f"   {name}: {voice.get('duration_seconds')}s, status={voice.get('status')}")

    print("-- agents")
    made["agents"] = {}
    faces = list(made["faces"].values())
    for index, spec in enumerate(agent_specs(voices)):
        body: dict[str, Any] = {
            "name": spec["name"],
            "system_prompt": spec["system_prompt"],
            "voice_id": spec["voice_id"],
            "turn_taking": spec["turn_taking"],
            "rubric_id": made["rubrics"][spec["rubric"]],
            "guardrail_id": made["guardrails"][spec["guardrail"]],
            "pronunciation_id": made["pronunciations"][spec["pronunciation"]],
            "knowledge_base_ids": [made["knowledge"][n] for n in spec["knowledge"]],
            "tool_ids": [made["tools"][n] for n in spec["tools"]],
        }
        # Faces are shared across agents rather than duplicated: we have two real people's clips
        # and inventing five personas from two faces would be a lie the console would then show.
        if faces:
            body["face_id"] = faces[index % len(faces)]
        agent = call("POST", "/agents", body)
        made["agents"][spec["name"]] = agent["id"]
        print(f"   {spec['name']}: voice={spec['voice_id']}, "
              f"{len(body['knowledge_base_ids'])} kb, {len(body['tool_ids'])} tools")

    print("-- candidates")
    made["candidates"] = {}
    resumes = ROOT / "data" / "seed-resumes"
    for spec in CANDIDATES:
        body = {
            "name": spec["name"],
            "email": spec["email"],
            "role": spec["role"],
            "notes": spec["notes"],
            "agent_id": made["agents"][spec["agent"]],
        }
        candidate = call("POST", "/candidates", body)
        made["candidates"][spec["name"]] = candidate["id"]
        detail = "no resume"
        if spec["resume_text"]:
            # Written to a temporary file and uploaded, rather than posted as a text field: the
            # upload path is the one an operator uses, and seeding around it would leave the
            # extractor untested by this script. The file lives under `data/`, which is gitignored,
            # so no invented personal document enters the repository.
            resumes.mkdir(parents=True, exist_ok=True)
            path = resumes / str(spec["resume"])
            path.write_text(str(spec["resume_text"]))
            uploaded = call(
                "POST", f"/candidates/{candidate['id']}/resume", files={"file": path}
            )
            detail = (
                f"{uploaded.get('resume_chars')} chars"
                if not uploaded.get("resume_error")
                else f"extract failed: {uploaded['resume_error']}"
            )
        print(f"   {spec['name']} ({spec['role']}): {detail}")

    print("-- interviews for three of them")
    for name in ("Aparna Rao", "Daniel Okonkwo", "Tom Whitfield"):
        invite = call("POST", f"/candidates/{made['candidates'][name]}/interview", {})
        # An attestation, so the report shows the identity block rather than "not recorded". Tom's
        # deliberately differs from the name on file: one report should show the mismatch warning,
        # because that is the case a reviewer most needs to have seen before it matters.
        typed = "Thomas Whitfield" if name == "Tom Whitfield" else name
        call(
            "POST",
            f"/sessions/{invite['session_id']}/attendance",
            {
                "confirmed_name": typed,
                "consented_to_recording": True,
                "user_agent": "Mozilla/5.0 (seed)",
                "timezone": "Asia/Kolkata",
            },
        )
        made.setdefault("invites", {})[name] = invite["session_id"]
        flag = "  <- name differs, on purpose" if typed != name else ""
        print(f"   {name}: {invite['session_id']} attested as {typed!r}{flag}")

    print("-- a completed, scored session")
    made["sessions"] = {}
    agent_id = made["agents"]["Senior backend interviewer"]
    session = call(
        "POST",
        "/sessions",
        {"agent_id": agent_id, "candidate_id": made["candidates"]["Aparna Rao"]},
    )
    call(
        "POST",
        f"/sessions/{session['id']}/attendance",
        {
            "confirmed_name": "Aparna Rao",
            "consented_to_recording": True,
            "user_agent": "Mozilla/5.0 (seed)",
            "timezone": "Asia/Kolkata",
        },
    )
    for index, (heard, said) in enumerate(TRANSCRIPT, start=1):
        call("POST", f"/sessions/{session['id']}/turns", {
            "epoch": index,
            "heard": heard,
            "said": said,
            "transcribed": True,
            "llm_ttft_ms": 1900 + index * 210,
            "tts_first_audio_ms": 280 + index * 9,
            "first_frame_ms": 2400 + index * 190,
            "perceived_total_ms": 2450 + index * 195,
            "interrupted": index == 3,
            "silent": False,
        })
    call("POST", f"/sessions/{session['id']}/end", {})
    made["sessions"]["scored"] = session["id"]
    print(f"   {session['id']}: {len(TRANSCRIPT)} turns recorded, session ended")

    # The score comes from the real scorer against the real rubric. It takes seconds of model
    # time, so it is requested and then polled rather than assumed.
    try:
        call("POST", f"/sessions/{session['id']}/score", {})
        for _ in range(40):
            record = call("GET", f"/sessions/{session['id']}")
            scoring = record.get("scoring") or {}
            if scoring.get("status") in ("ready", "failed"):
                print(f"   scoring: {scoring.get('status')}")
                break
            time.sleep(3)
        else:
            print("   scoring: still running — the report page will fill in when it lands")
    except ApiError as exc:
        print(f"   !! scoring could not be requested: {exc}")

    # A second session left open, so the interview room has something to join.
    live = call("POST", "/sessions", {"agent_id": made["agents"]["Practice run — friendly"]})
    made["sessions"]["open"] = live["id"]
    print(f"   {live['id']}: left open for the interview room")

    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="delete every record first")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, write nothing")
    args = parser.parse_args()

    config = call("GET", "/config")
    print(f"-- {API}: store={config.get('store')} renderer={config.get('renderer')} "
          f"tts={config.get('tts')} stt={config.get('stt')} llm={config.get('llm')}")
    problems = config.get("schema_problems") or []
    for problem in problems:
        print(f"   !! schema: {problem}")
    if problems:
        return 1

    if args.reset and not args.dry_run:
        reset()

    made = seed(dry_run=args.dry_run)
    if args.dry_run:
        return 0

    print("\n-- what exists now")
    for collection in ("candidates", "agents", "faces", "voices", "knowledge", "rubrics",
                       "guardrails", "pronunciations", "tools", "sessions"):
        records = call("GET", f"/{collection}") or []
        print(f"   {collection:16} {len(records)}")

    if made.get("sessions", {}).get("open"):
        print(f"\n   interview room: http://localhost:3000/interview/"
              f"{made['sessions']['open']}")
    if made.get("sessions", {}).get("scored"):
        print(f"   report:         http://localhost:3000/sessions/"
              f"{made['sessions']['scored']}")
    print("   console:        http://localhost:3000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
