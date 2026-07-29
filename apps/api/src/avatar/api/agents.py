"""
CRUD for the Agent — the object every other console resource hangs off.

**Why the turn-taking numbers are validated here and not only in `TurnDetector`.** The
detector already refuses an inverted hysteresis pair at construction, which is the right
place for the invariant and the wrong place to *discover* it: construction happens when a
candidate connects, so a bad pair stored through this API surfaces as a session that dies
at start rather than as a form that would not save. Rejecting the same shape at write time
puts the failure in front of the operator who caused it, while there is still a form open.

**Why the defaults are imported rather than typed out.** They are the runtime's own
constants. A literal `0.6` here would drift from `audio.turn_detection` the first time a
threshold is tuned, and the console would then be confidently describing a policy the
server does not run.

**Why records go back to the client exactly as the store wrote them.** A response model
would filter every read through this file's idea of an Agent, so a field written by a
newer build would vanish from the console while still sitting on disk — a partial deploy
would look like data loss. The store is the authority on what an agent is; this module is
the authority on what may be written.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from avatar.audio.turn_detection import (
    END_OF_TURN_SILENCE_MS,
    MIN_SPEECH_MS,
    ONSET_FRAMES,
    ONSET_PROBABILITY,
    RELEASE_PROBABILITY,
)
from avatar.store import NotFound, store

COLLECTION = "agents"
ID_PREFIX = "agent"

router = APIRouter(prefix="/agents", tags=["agents"])

LlmProvider = Literal["openai", "anthropic", "scripted"]
VoiceProvider = Literal["deepgram", "tone"]


def _stripped_name(value: str) -> str:
    """
    Trim, and refuse what is left if it is empty.

    `min_length=1` alone accepts a single space, which is worse than an empty name: the row
    renders blank and cannot be found by anyone searching the list for it.
    """
    stripped = value.strip()
    if not stripped:
        raise ValueError("name must not be blank")
    return stripped


AgentName = Annotated[str, Field(min_length=1), AfterValidator(_stripped_name)]
"""Shared by create and update so the two paths cannot disagree about a legal name."""


class TurnTaking(BaseModel):
    """
    The turn-taking policy for one agent, mirroring `TurnDetector`'s parameters.

    Exposed as configuration rather than hidden behind server defaults because
    `end_of_turn_silence_ms` is the largest single term in the measured latency budget and
    is a conversational judgment, not a technical one — no hardware makes it smaller, so
    the person tuning the interview has to be able to see and move it.
    """

    # A typo'd key is worse than a rejection: it would be stored, never read, and the
    # operator would believe they had changed a threshold they had not.
    model_config = ConfigDict(extra="forbid")

    onset_probability: float = Field(default=ONSET_PROBABILITY, ge=0.0, le=1.0)
    release_probability: float = Field(default=RELEASE_PROBABILITY, ge=0.0, le=1.0)
    # `ge=1` mirrors the detector exactly: it raises on anything smaller, because zero
    # consecutive frames means every frame of room noise is an interruption.
    onset_frames: int = Field(default=ONSET_FRAMES, ge=1)
    min_speech_ms: int = Field(default=MIN_SPEECH_MS, ge=0)
    end_of_turn_silence_ms: int = Field(default=END_OF_TURN_SILENCE_MS, ge=0)

    @model_validator(mode="after")
    def _hysteresis_must_not_invert(self) -> TurnTaking:
        """
        Release must sit strictly below onset. The gap *is* the hysteresis.

        Equal thresholds are rejected as well as inverted ones, which is stricter than the
        detector: with no gap, the probability dip inside an ordinary word drops below the
        release bar the moment it stops clearing the onset bar, so the turn ends mid-word.
        A config with no hysteresis at all has no reason to reach disk.
        """
        if self.release_probability >= self.onset_probability:
            raise ValueError(
                f"release_probability ({self.release_probability}) must be below "
                f"onset_probability ({self.onset_probability}); the gap between them is "
                "the hysteresis that stops a dip mid-word from ending the turn"
            )
        return self


class AgentCreate(BaseModel):
    """What a client may send to create an agent."""

    model_config = ConfigDict(extra="forbid")

    name: AgentName
    system_prompt: str = ""
    # Defaults are the credential-free pair on purpose, matching the runtime's own
    # defaults: a fresh agent has to be usable on a clean clone with no keys, or the first
    # thing an operator does is debug a 401.
    llm_provider: LlmProvider = "scripted"
    llm_model: str = ""
    voice_provider: VoiceProvider = "tone"
    voice_id: str = ""
    face_id: str | None = None
    knowledge_base_ids: list[str] = Field(default_factory=list)
    tool_ids: list[str] = Field(default_factory=list)
    guardrail_id: str | None = None
    pronunciation_id: str | None = None
    turn_taking: TurnTaking = Field(default_factory=TurnTaking)


class AgentUpdate(BaseModel):
    """
    A partial update. Only the keys present in the request body are touched.

    `turn_taking` is replaced whole rather than merged field-by-field, because the
    hysteresis invariant spans two fields: merging `release_probability` alone against a
    stored `onset_probability` could only be checked after the merge, which would put the
    rule in a second place and give it a second chance to be forgotten. Sending the whole
    object keeps `TurnTaking` the single authority on whether a pair is legal.
    """

    model_config = ConfigDict(extra="forbid")

    name: AgentName | None = None
    system_prompt: str | None = None
    llm_provider: LlmProvider | None = None
    llm_model: str | None = None
    voice_provider: VoiceProvider | None = None
    voice_id: str | None = None
    face_id: str | None = None
    knowledge_base_ids: list[str] | None = None
    tool_ids: list[str] | None = None
    guardrail_id: str | None = None
    pronunciation_id: str | None = None
    turn_taking: TurnTaking | None = None


def _not_found(agent_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=f"no agent with id {agent_id!r}"
    )


@router.get("")
async def list_agents() -> list[dict[str, Any]]:
    """Newest first, as the store orders them — the list view relies on that order."""
    return store.list(COLLECTION)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_agent(body: AgentCreate) -> dict[str, Any]:
    return store.create(COLLECTION, ID_PREFIX, body.model_dump())


@router.get("/{agent_id}")
async def get_agent(agent_id: str) -> dict[str, Any]:
    try:
        return store.get(COLLECTION, agent_id)
    except NotFound as exc:
        raise _not_found(agent_id) from exc


@router.patch("/{agent_id}")
async def update_agent(agent_id: str, body: AgentUpdate) -> dict[str, Any]:
    """
    Merge the keys that were sent.

    `exclude_unset` is what makes this a patch rather than a replace: without it, every
    field the client omitted would arrive as its default and quietly overwrite a tuned
    value. Note the one thing this cannot express — the store's merge drops `None`, so a
    nullable field such as `face_id` cannot be cleared back to null through here. That is
    a limitation of the store's merge, not a decision made here, and inventing a sentinel
    to work around it would put two ideas of "empty" into the data.
    """
    try:
        return store.update(COLLECTION, agent_id, body.model_dump(exclude_unset=True))
    except NotFound as exc:
        raise _not_found(agent_id) from exc


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(agent_id: str) -> None:
    """
    Hard delete, and deliberately not cascading.

    Sessions reference an agent id; erasing those references to keep the data tidy would
    rewrite history, and a transcript that no longer says which agent produced it is worth
    less than a dangling id.
    """
    try:
        store.delete(COLLECTION, agent_id)
    except NotFound as exc:
        raise _not_found(agent_id) from exc
