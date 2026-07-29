"""
CRUD for guardrails — the input and output policy attached to a session.

**Why every check here is a local string/regex pass and never a model call.** The
output-side check runs on generated text *after* the LLM and *before* TTS, which puts it
inside a latency budget with nothing to spare: a turn already measures 2.7-5.8s end to end
(PROCESS.md §3.4), and the text path is the part the candidate experiences as silence. A
network round trip at that point would be self-defeating twice over — it adds its own
latency to the exact segment that cannot absorb any, and it adds a dependency that can time
out mid-sentence, at which point the enforcement either fails open (a guardrail that is
trusted and does nothing — worse than none) or stalls the turn. So `evaluate()` is
deliberately dumb: literal topic terms and two PII patterns, in-process, no I/O.

The trade-off is stated rather than hidden. This catches what it is told to catch and
cannot generalise: a banned topic reached by paraphrase walks straight through. A semantic
classifier is the right tool for that and belongs on the *input* side, off the critical
path, where a few hundred milliseconds are affordable. It is not implemented here.

Two policy decisions worth reading before changing anything:

  * **Redaction is a repair, not a refusal.** A PII hit rewrites the text and reports the
    violation, but leaves `allowed` true. Refusing a turn because the candidate said their
    own email address would throw away their answer to protect them from themselves.
  * **`allowed=False` means "the caller must apply `on_violation`".** This endpoint never
    decides between refuse/redirect/end_session; that belongs to whoever owns the turn.
    `redacted_text` is always the safest shippable rendering of the input, so a caller that
    proceeds anyway still ships something compliant.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from avatar.store import NotFound, Store, store

COLLECTION = "guardrails"
ID_PREFIX = "guard"

DEFAULT_MAX_ANSWER_CHARS = 600
"""
Roughly four spoken sentences at Aura's cadence.

The cap exists because answer length *is* latency here: every character is synthesised and
then rendered frame by frame, so an over-long answer is not verbose, it is a candidate
waiting. The default is a starting point for an interviewer persona, not a measurement.
"""

MAX_ANSWER_CHARS_CEILING = 20_000
"""
The outer wall, not a recommendation.

A value large enough that truncation can never fire makes the field look configured while
enforcing nothing; rejecting the absurd end keeps "600" and "20000" as choices someone made
rather than a number nobody read.
"""

DEFAULT_REFUSAL_MESSAGE = "I'd rather not go into that. Tell me about your own work instead."

MIN_PII_DIGITS = 7
"""
How many digits make a run phone- or card-like.

Seven is the shortest local phone number; cards are 13-19. Below seven, digit runs are
years, prices, counts, and version numbers — redacting those would mangle ordinary answers,
and a redactor that mangles ordinary answers gets switched off, which protects nobody.
"""

EMAIL_PLACEHOLDER = "[redacted-email]"
NUMBER_PLACEHOLDER = "[redacted-number]"

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")

_DIGIT_RUN_RE = re.compile(r"\+?\d[\d\s().\-]{5,}\d")
"""
A *candidate* digit run: separators allowed inside, because that is how humans say a phone
number out loud and how STT writes it down ("555 867 5309", "4111-1111-1111-1111"). Matching
bare `\\d{7,}` would miss every spaced form, which is most of them. The digit count is then
checked in `_redact_numbers` — the pattern is intentionally loose and the counter is the
actual rule, since a length bound inside the regex cannot tell digits from hyphens.
"""


def get_store() -> Store:
    """
    Indirection so tests can bind a `Store(tmp_path)` via `dependency_overrides`.

    Without it the routes close over the process-wide default and a test run writes into
    whatever `AVATAR_DATA_DIR` points at — i.e. someone's real console data.
    """
    return store


StoreDep = Annotated[Store, Depends(get_store)]

OnViolation = Literal["refuse", "redirect", "end_session"]
Direction = Literal["input", "output"]


def _clean_name(value: str) -> str:
    """
    Reject a blank name rather than storing it.

    A guardrail is picked from a dropdown; a whitespace-only name renders as an empty row,
    which is indistinguishable from the next one and gets attached to the wrong session.
    """
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("name must not be blank")
    return cleaned


def _clean_topics(values: list[str]) -> list[str]:
    """
    Normalise topics to lowercase with inner whitespace collapsed, and reject blanks.

    Blank entries are the important part: an empty term compiles to a pattern that matches
    every string, so one stray comma in the form field would produce a policy that refuses
    every single turn — a total outage that passes validation. Lowercasing is safe because
    matching is case-insensitive anyway, and it makes duplicates ("Salary", "salary")
    detectable so the check does not report the same violation twice.
    """
    seen: dict[str, None] = {}
    for raw in values:
        topic = " ".join(raw.split()).lower()
        if not topic:
            raise ValueError("banned_topics must not contain blank entries")
        seen.setdefault(topic, None)
    return list(seen)


class Policy(BaseModel):
    """
    The policy fields, shared by the create body, the patch merge, and the stored record.

    No `extra="forbid"` here: this is also used to read records back off disk, where a field
    added in a later version must not make every existing guardrail unreadable.
    """

    name: str
    banned_topics: list[str] = Field(default_factory=list)
    pii_redaction: bool = False
    max_answer_chars: int = Field(
        default=DEFAULT_MAX_ANSWER_CHARS, ge=1, le=MAX_ANSWER_CHARS_CEILING
    )
    refusal_message: str = DEFAULT_REFUSAL_MESSAGE
    on_violation: OnViolation = "refuse"

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _clean_name(value)

    @field_validator("banned_topics")
    @classmethod
    def _topics(cls, value: list[str]) -> list[str]:
        return _clean_topics(value)

    @field_validator("refusal_message")
    @classmethod
    def _refusal_not_blank(cls, value: str) -> str:
        """
        A blank refusal message is silence on the wire.

        Every `on_violation` mode can reach this string — `refuse` speaks it, `redirect`
        leads with it, `end_session` signs off with it — so there is no configuration in
        which an empty one is meaningful. To the candidate, silence after their question is
        indistinguishable from a crashed avatar, which is the worst possible reading of a
        guardrail working exactly as configured.
        """
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("refusal_message must not be blank")
        return cleaned


class GuardrailCreate(Policy):
    """
    A new guardrail.

    `extra="forbid"` on purpose: a misspelled field name would otherwise be dropped in
    silence, leaving a policy that reads as configured and enforces the defaults.
    """

    model_config = ConfigDict(extra="forbid")


class GuardrailPatch(BaseModel):
    """
    A partial update. Omitted fields are left alone.

    Note what this cannot express: clearing a field back to null. `Store.update` drops
    `None` values, so `{"refusal_message": null}` is a no-op rather than a deletion — the
    safer default, since an accidental null must not strip the refusal text off a live
    policy. Clearing `banned_topics` is still possible with `[]`, which is a real value.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    banned_topics: list[str] | None = None
    pii_redaction: bool | None = None
    max_answer_chars: int | None = Field(default=None, ge=1, le=MAX_ANSWER_CHARS_CEILING)
    refusal_message: str | None = None
    on_violation: OnViolation | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _clean_name(value)

    @field_validator("banned_topics")
    @classmethod
    def _topics(cls, value: list[str]) -> list[str]:
        return _clean_topics(value)


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    direction: Direction


class CheckResult(BaseModel):
    """
    What the caller acts on.

    `violations` are stable machine-readable codes (`banned_topic:<term>`, `pii:email`,
    `pii:number`, `max_answer_chars:<len>>{limit}`) rather than prose, because they end up
    in telemetry and in the console's checker panel, and a message someone reworded should
    not break a counter.
    """

    allowed: bool
    violations: list[str]
    redacted_text: str


@lru_cache(maxsize=512)
def _topic_pattern(topic: str) -> re.Pattern[str]:
    """
    Match a topic as a whole term, case-insensitively.

    `(?<!\\w)` / `(?!\\w)` rather than `\\b` because a topic may begin or end with a
    non-word character ("c++", ".net"), where `\\b` asserts the opposite of what is wanted
    and silently never matches. The lookarounds are what stop a ban on "ai" from firing on
    "said" and "maintain" — plain substring matching produces exactly that class of false
    positive, and a checker that cries wolf is one an operator disables wholesale.

    Cached because the output side re-derives these on every generated sentence.
    """
    return re.compile(rf"(?<!\w){re.escape(topic)}(?!\w)", re.IGNORECASE)


def _redact_numbers(text: str) -> str:
    """Blank out phone/card-like digit runs, leaving years and prices alone."""

    def replace(match: re.Match[str]) -> str:
        found = match.group(0)
        digits = sum(char.isdigit() for char in found)
        # Below the threshold this was a year or a quantity, not an identifier. Returning
        # the match unchanged is how the loose pattern above is narrowed without a regex
        # that would need to count digits it cannot distinguish from separators.
        return NUMBER_PLACEHOLDER if digits >= MIN_PII_DIGITS else found

    return _DIGIT_RUN_RE.sub(replace, text)


def _truncate(text: str, limit: int) -> str:
    """
    Cut to `limit` characters on the last word boundary that fits.

    Slicing mid-word produces a fragment TTS pronounces as a nonsense syllable, which reads
    as a broken renderer rather than as a policy doing its job.
    """
    if len(text) <= limit:
        return text
    cut = text[:limit]
    head, sep, _ = cut.rpartition(" ")
    return (head if sep and head else cut).rstrip()


def evaluate(policy: Policy, text: str, direction: Direction) -> CheckResult:
    """
    Apply the policy locally. No I/O of any kind — see the module docstring for why.

    Order matters in two places:

      * Banned topics are matched against the **original** text, not the redacted one, so a
        placeholder can never swallow a banned term and report the turn as clean.
      * The length check runs on the **redacted** text, because that is the string that
        would actually reach TTS, and it is the only length the budget cares about.

    `max_answer_chars` is an *output*-side rule only. It bounds what the avatar says; a long
    answer from the candidate is the thing this product exists to collect.
    """
    violations: list[str] = []
    redacted = text

    if policy.pii_redaction:
        without_email = _EMAIL_RE.sub(EMAIL_PLACEHOLDER, redacted)
        if without_email != redacted:
            violations.append("pii:email")
        # Emails are redacted first so digits inside a mailbox local part are already gone
        # and cannot trip the number pass a second time.
        without_numbers = _redact_numbers(without_email)
        if without_numbers != without_email:
            violations.append("pii:number")
        redacted = without_numbers

    for topic in policy.banned_topics:
        if _topic_pattern(topic).search(text):
            violations.append(f"banned_topic:{topic}")

    over_length = direction == "output" and len(redacted) > policy.max_answer_chars
    if over_length:
        violations.append(f"max_answer_chars:{len(redacted)}>{policy.max_answer_chars}")
        redacted = _truncate(redacted, policy.max_answer_chars)

    blocked = over_length or any(v.startswith("banned_topic:") for v in violations)
    return CheckResult(allowed=not blocked, violations=violations, redacted_text=redacted)


router = APIRouter(prefix="/guardrails", tags=["guardrails"])


def _load(data: Store, guardrail_id: str) -> dict[str, Any]:
    try:
        return data.get(COLLECTION, guardrail_id)
    except NotFound:
        # 404 rather than a bare KeyError traceback: an unknown id is a routine client
        # mistake (stale tab, deleted record), not a server fault.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no guardrail {guardrail_id!r}"
        ) from None


@router.get("")
def list_guardrails(data: StoreDep) -> list[dict[str, Any]]:
    """Newest first, per `Store.list`. The console's table renders this order verbatim."""
    return data.list(COLLECTION)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_guardrail(body: GuardrailCreate, data: StoreDep) -> dict[str, Any]:
    return data.create(COLLECTION, ID_PREFIX, body.model_dump())


@router.get("/{guardrail_id}")
def get_guardrail(guardrail_id: str, data: StoreDep) -> dict[str, Any]:
    return _load(data, guardrail_id)


@router.patch("/{guardrail_id}")
def update_guardrail(guardrail_id: str, body: GuardrailPatch, data: StoreDep) -> dict[str, Any]:
    """
    Merge a patch, then re-validate the **merged** policy before writing.

    Validating the patch in isolation is not enough: the invariants belong to the stored
    record, and a field-by-field-valid patch can still land a record that `evaluate()` would
    read as a policy nobody intended. Re-validating the merge is what stops a write that
    passes at request time and misbehaves at turn time.
    """
    current = _load(data, guardrail_id)
    patch = body.model_dump(exclude_unset=True, exclude_none=True)
    try:
        Policy.model_validate({**current, **patch})
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=[
                {"field": ".".join(str(part) for part in err["loc"]), "message": err["msg"]}
                for err in exc.errors()
            ],
        ) from exc
    return data.update(COLLECTION, guardrail_id, patch)


@router.delete("/{guardrail_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_guardrail(guardrail_id: str, data: StoreDep) -> None:
    try:
        data.delete(COLLECTION, guardrail_id)
    except NotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no guardrail {guardrail_id!r}"
        ) from None


@router.post("/{guardrail_id}/check")
def check_guardrail(guardrail_id: str, body: CheckRequest, data: StoreDep) -> CheckResult:
    """
    Run the policy against a piece of text. Local only; no model call, no network.

    Exists as an endpoint mostly so the console can offer a live checker. A guardrail nobody
    can test is a guardrail nobody trusts — an operator who cannot see "salary" trip on a
    sample answer has no way to tell a working policy from a typo in the topic list.

    The record is read through `Policy` rather than used as a raw dict so a guardrail stored
    before a field existed evaluates with that field's default instead of raising a KeyError
    inside the turn.
    """
    return evaluate(Policy.model_validate(_load(data, guardrail_id)), body.text, body.direction)
