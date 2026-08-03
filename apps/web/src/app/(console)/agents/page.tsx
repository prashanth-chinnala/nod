"use client";

/**
 * Agents — the object every other resource attaches to.
 *
 * The screen is built around `end_of_turn_silence_ms` rather than around the model picker,
 * because that is where the product decision actually is. It is the largest single term in
 * the measured latency budget and the only one no hardware can shrink: whatever is set here
 * appears in the candidate's perceived wait unchanged. Burying it behind an "advanced"
 * disclosure would hide the one field on this page that decides how the interview feels.
 *
 * Turn-taking is therefore in the create form, not only in an editor, and it has a column in
 * the table. What the table does *not* show is a measured per-agent turnaround: no session
 * has been run per agent, so there is no number, and a plausible one here would be worse
 * than the gap.
 *
 * The knowledge-base, tool, guardrail and pronunciation attachments are omitted from the
 * form on purpose. They are references to resources that must already exist, and a free-text
 * id box invites a typo that validates here and fails at session start — they belong to a
 * picker built once those collections are populated.
 */

import { useCallback, useEffect, useState } from "react";

import { Attachments } from "@/components/attachments";

import {
  Button,
  Card,
  CardHeader,
  Cell,
  Chip,
  Empty,
  Combo,
  Field,
  Input,
  Page,
  Row,
  Select,
  Table,
  Textarea,
} from "@/components/ui";

const API = "http://127.0.0.1:8000/agents";

/**
 * The runtime's own turn-detection defaults, mirrored so the form opens on the values the
 * server would use anyway. Duplicated across a process boundary, so they are named once
 * here: a scattered `0.6` would silently disagree with `audio/turn_detection.py` the first
 * time a threshold is tuned.
 */
const DEFAULTS = {
  onset_probability: 0.6,
  release_probability: 0.35,
  onset_frames: 3,
  min_speech_ms: 200,
  end_of_turn_silence_ms: 700,
} as const;

/** The measured full-turn range from PROCESS.md §3.4. Quoted, never recomputed here. */
const TURN_BUDGET = "2.7–5.8s";

type LlmProvider = "openai" | "anthropic" | "scripted";
type VoiceProvider = "deepgram" | "tone" | "clone";

/**
 * Model-name suggestions, per provider.
 *
 * Suggestions, emphatically not a valid set — the field they feed is a `Combo`, which still
 * accepts anything typed. The distinction is the point: this console cannot know what a provider
 * offers, and a closed list would make a model released tomorrow unselectable.
 *
 * `scripted` has no entry because it has no model. It answers from a fixed script, which is what
 * makes it the credential-free default.
 *
 * The runtime's own currently-configured model is prepended at render time from `/config`. That
 * one is worth more than anything static here: it is the value known to work on this host, keys
 * and base URL included.
 */
const MODEL_SUGGESTIONS: Record<LlmProvider, readonly string[]> = {
  scripted: [],
  anthropic: ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
  // Deliberately empty. `AVATAR_LLM=openai` here points at an OpenAI-compatible base URL —
  // Ollama, LM Studio, vLLM — so the model names depend on which one is running and what is
  // pulled locally. Guessing a list would be worse than offering none; `/config` supplies the
  // one name that is actually serving.
  openai: [],
};

/**
 * Voice suggestions, per provider. Same reasoning as the models above.
 *
 * Deepgram's list is only the adapter's own `DEFAULT_VOICE` from `audio/tts_deepgram.py` — a
 * value this repo can point at. The rest of the Aura catalogue is not enumerated here because
 * nothing in this repo knows it, and a misremembered voice id fails at session start with a
 * provider error rather than at the form.
 */
const VOICE_SUGGESTIONS: Record<VoiceProvider, readonly string[]> = {
  tone: [],
  deepgram: ["aura-2-thalia-en"],
  // A cloned voice has no catalogue to suggest from: the identity comes from an uploaded
  // recording attached below, not from a name typed here.
  clone: [],
};

/** A face, as much of one as this form needs to offer it. */
type FaceOption = { id: string; name: string; status: string };

/**
 * The subset of `/config` this form uses: what the runtime is actually running right now.
 *
 * The provider fields matter as much as the names. `/config` reports one model and one voice —
 * whichever provider the server is configured for — so suggesting `gpt-oss:20b` to someone who
 * just selected `anthropic` would be offering a name that cannot work. The suggestion is only
 * made when the provider matches.
 */
type RuntimeConfig = { llm?: string; llm_model?: string; tts?: string; tts_voice?: string };

type TurnTaking = {
  onset_probability: number;
  release_probability: number;
  onset_frames: number;
  min_speech_ms: number;
  end_of_turn_silence_ms: number;
};

type Agent = {
  id: string;
  name: string;
  system_prompt: string;
  llm_provider: LlmProvider;
  llm_model: string;
  voice_provider: VoiceProvider;
  voice_id: string;
  voice_ref_id: string | null;
  face_id: string | null;
  knowledge_base_ids: string[];
  tool_ids: string[];
  guardrail_id: string | null;
  pronunciation_id: string | null;
  rubric_id: string | null;
  turn_taking: TurnTaking;
  created_at: string;
  updated_at: string;
};

const HEAD = ["name", "model", "voice", "attached", "end of turn", "updated", ""] as const;

/** An unset optional reference reads as a dash, not as an empty cell that looks broken. */
const UNSET = "—";

/**
 * Which kinds of resource an agent references, as a short list.
 *
 * Names the kinds rather than counting them: "rubric, guardrail" answers the question an operator
 * is actually asking, where "2 attached" would send them to the editor to find out which two.
 */
function attachedSummary(agent: Agent): string {
  const parts: string[] = [];
  if (agent.rubric_id) parts.push("rubric");
  if (agent.face_id) parts.push("face");
  if (agent.voice_ref_id) parts.push("voice");
  if (agent.guardrail_id) parts.push("guardrail");
  if (agent.pronunciation_id) parts.push("lexicon");
  if (agent.knowledge_base_ids.length) parts.push(`${agent.knowledge_base_ids.length} kb`);
  if (agent.tool_ids.length) parts.push(`${agent.tool_ids.length} tools`);
  return parts.join(", ");
}

/**
 * Flagged above the runtime's default, and deliberately not against a measured target.
 *
 * There is no per-agent measurement to compare with, so this says only what it can defend:
 * this agent waits longer than the runtime's own 700ms before answering, and that extra
 * silence lands in the candidate's wait as-is. Going the other way is not "better" — it
 * trades latency for answering into a thinking pause — so below the default is left neutral.
 */
function silenceStatus(ms: number): "warn" | "neutral" {
  return ms > DEFAULTS.end_of_turn_silence_ms ? "warn" : "neutral";
}

/**
 * FastAPI reports a 422 as a list of per-field errors and everything else as a string.
 * Flattening both is what puts the hysteresis rejection — which names both thresholds — in
 * front of the operator instead of a generic "the runtime rejected this".
 */
function problemFrom(payload: unknown, status: number): string {
  const detail = (payload as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((entry) => {
        const { loc, msg } = entry as { loc?: unknown; msg?: unknown };
        const field = Array.isArray(loc) ? loc.slice(1).join(".") : "";
        return typeof msg === "string" ? (field ? `${field}: ${msg}` : msg) : null;
      })
      .filter((line): line is string => line !== null);
    if (messages.length > 0) return messages.join(" · ");
  }
  return `The runtime rejected this agent (${status}).`;
}

export default function AgentsPage() {
  const [agents, setAgents] = useState<Agent[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [editing, setEditing] = useState<Agent | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);

  /*
    A promise chain rather than async/await, and not as a style preference. React 19's
    `set-state-in-effect` rule cannot see through an async function reference, so calling one
    from an effect body is reported as a synchronous setState; the shape it does accept is
    this one, where state is written from a callback the fetch invokes later. Wrapping the
    same async function in an IIFE would silence the rule without changing what runs, which
    is worse than writing the form it asks for.

    Nothing is cleared before the request resolves, so a retry keeps the current rows and the
    current message on screen instead of flashing an empty table on the way to the answer.
  */
  const load = useCallback(() => {
    fetch(API, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`the runtime answered ${response.status}`);
        return (await response.json()) as Agent[];
      })
      .then((listed) => {
        setAgents(listed);
        setError(null);
      })
      .catch((cause: unknown) => {
        // The runtime is a separate process on :8000. A bare "Failed to fetch" reads as a
        // bug in this page, so the message has to name the thing that is actually down.
        setAgents(null);
        setError(cause instanceof Error ? cause.message : "the runtime is unreachable");
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function remove(id: string) {
    setRemoving(id);
    try {
      const response = await fetch(`${API}/${id}`, { method: "DELETE" });
      // 409 is the store refusing because something still references this agent. Surfaced as-is:
      // the runtime's message names what, which is more useful than "could not delete".
      if (!response.ok && response.status !== 404) {
        const payload: unknown = await response.json().catch(() => null);
        setError(problemFrom(payload, response.status));
        return;
      }
      if (editing?.id === id) setEditing(null);
      load();
    } catch {
      setError("Could not reach the runtime to delete that agent.");
    } finally {
      setRemoving(null);
    }
  }


  return (
    <Page
      title="Agents"
      lede={`An agent is the interviewer: a prompt, a model, a voice, a face, and a turn-taking policy. End-of-turn silence is the largest single term in a turn that already measures ${TURN_BUDGET}, so it is configured here rather than hidden in the server.`}
      action={
        <Button variant="primary" onClick={() => setShowForm((open) => !open)}>
          {showForm ? "Close" : "New agent"}
        </Button>
      }
    >
      {editing ? (
        <CreateForm
          agent={editing}
          onCancel={() => setEditing(null)}
          onCreated={() => {
            setEditing(null);
            load();
          }}
        />
      ) : showForm ? (
        <CreateForm
          onCreated={() => {
            setShowForm(false);
            load();
          }}
        />
      ) : null}

      <Card>
        <CardHeader
          title="Configured agents"
          hint="End of turn is how long the detector waits in silence before answering — a conversational judgement, not a technical one, and it is added to the candidate's wait unchanged. Measured turnaround per agent is not shown: no agent has run a session yet, so there is no number to report."
        />

        {error ? (
          <Empty
            title="Could not reach the runtime"
            action={<Button onClick={load}>Retry</Button>}
          >
            {error}. The console reads agents from the runtime on 127.0.0.1:8000 — start it
            with <code className="font-mono text-ink">uvicorn avatar.server:app</code> and
            retry.
          </Empty>
        ) : agents === null ? (
          <div className="px-5 py-14 text-center text-[12.5px] text-ink-mid">
            Loading agents…
          </div>
        ) : agents.length === 0 ? (
          <Empty
            title="No agents yet"
            action={
              <Button variant="primary" onClick={() => setShowForm(true)}>
                New agent
              </Button>
            }
          >
            An agent is the interviewer a candidate actually meets — what it asks, which model
            answers, whose voice and face it wears, and how patiently it waits before
            replying. Everything else in this console attaches to one. Create the first and
            the defaults will be the credential-free pair, so it runs with no keys set.
          </Empty>
        ) : (
          <Table head={HEAD}>
            {agents.map((agent) => (
              <Row key={agent.id}>
                <Cell>{agent.name}</Cell>
                <Cell dim mono>
                  {agent.llm_provider}
                  {agent.llm_model ? ` · ${agent.llm_model}` : ""}
                </Cell>
                <Cell dim mono>
                  {agent.voice_provider}
                  {agent.voice_id ? ` · ${agent.voice_id}` : ""}
                </Cell>
                <Cell dim>
                  {/* What is attached, by kind rather than by id. An operator scanning this list
                      wants to know whether an agent has a rubric at all; the ids are in the
                      editor below, where they can be changed. */}
                  {attachedSummary(agent) || UNSET}
                </Cell>
                <Cell>
                  {/* The column that decides how the interview feels, so it gets the chip. */}
                  <Chip status={silenceStatus(agent.turn_taking.end_of_turn_silence_ms)}>
                    {agent.turn_taking.end_of_turn_silence_ms}ms
                  </Chip>
                </Cell>
                <Cell dim mono>
                  {agent.updated_at}
                </Cell>
                <Cell>
                  {/* Edit and delete, which this screen went without -- and a comment in this very
                      table used to point at "the editor below", describing something that did not
                      exist. An agent is the object here most worth tuning: the prompt is written
                      by iteration, and the end-of-turn window is a number nobody guesses right
                      the first time. Create-only meant delete-and-retype, which loses the id every
                      recorded session references. */}
                  <span className="flex gap-1.5">
                    <Button
                      onClick={() => {
                        setShowForm(false);
                        setEditing(agent);
                      }}
                    >
                      Edit
                    </Button>
                    <Button
                      variant="danger"
                      disabled={removing === agent.id}
                      onClick={() => void remove(agent.id)}
                    >
                      {removing === agent.id ? "…" : "Delete"}
                    </Button>
                  </span>
                </Cell>
              </Row>
            ))}
          </Table>
        )}
      </Card>

      {/* Below the table rather than inside a row: the reference pickers are six controls and a
          two-line explanation each, which would not survive being squeezed into a table cell.
          It also keeps the table scannable, which is what a list view is for. */}
      <Attachments agents={agents ?? []} onChanged={load} />
    </Page>
  );
}

/**
 * Inline create form, toggled rather than shown in a modal.
 *
 * A modal would put the turn-taking numbers on top of the table they have to be judged
 * against, and it would have nowhere to grow when the attachment pickers arrive.
 */
/**
 * The agent form, for a new agent or an existing one.
 *
 * **One component for both, because the alternative was what shipped: nothing.** This screen let
 * you create an agent and never change it — and a comment in the table above pointed at "the
 * editor below, where they can be changed", describing something that did not exist. An agent is
 * the object in this product most worth tuning: the prompt is written by iteration, and the
 * end-of-turn window is a number nobody guesses correctly the first time. Create-only meant
 * delete-and-retype, which loses the id every attached session references.
 *
 * A second component would have duplicated the provider suggestions, the hysteresis check and the
 * turn-taking fields, and the two would have drifted. So the same form PATCHes when it is given an
 * agent and POSTs when it is not.
 */
function CreateForm({
  agent,
  onCreated,
  onCancel,
}: {
  agent?: Agent | null;
  onCreated: () => void;
  onCancel?: () => void;
}) {
  const editing = Boolean(agent);
  const [name, setName] = useState(agent?.name ?? "");
  const [systemPrompt, setSystemPrompt] = useState(agent?.system_prompt ?? "");
  const [llmProvider, setLlmProvider] = useState<LlmProvider>(
    (agent?.llm_provider as LlmProvider) ?? "scripted",
  );
  const [llmModel, setLlmModel] = useState(agent?.llm_model ?? "");
  const [voiceProvider, setVoiceProvider] = useState<VoiceProvider>(
    (agent?.voice_provider as VoiceProvider) ?? "tone",
  );
  const [voiceId, setVoiceId] = useState(agent?.voice_id ?? "");
  const [faceId, setFaceId] = useState(agent?.face_id ?? "");
  const [faces, setFaces] = useState<FaceOption[] | null>(null);
  const [runtime, setRuntime] = useState<RuntimeConfig | null>(null);
  const turn = agent?.turn_taking;
  const [onsetProbability, setOnsetProbability] = useState(
    String(turn?.onset_probability ?? DEFAULTS.onset_probability),
  );
  const [releaseProbability, setReleaseProbability] = useState(
    String(turn?.release_probability ?? DEFAULTS.release_probability),
  );
  const [onsetFrames, setOnsetFrames] = useState(
    String(turn?.onset_frames ?? DEFAULTS.onset_frames),
  );
  const [minSpeechMs, setMinSpeechMs] = useState(
    String(turn?.min_speech_ms ?? DEFAULTS.min_speech_ms),
  );
  const [endOfTurnMs, setEndOfTurnMs] = useState(
    String(turn?.end_of_turn_silence_ms ?? DEFAULTS.end_of_turn_silence_ms),
  );
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  /*
    The two lists this form offers. Promise chains rather than async/await in the effect body,
    for the reason given on `load` above.

    Neither failure is worth surfacing as an error. A face list that does not arrive leaves the
    picker disabled and the agent creatable without a face — which is a supported outcome, since
    faces attach after the fact — and a `/config` that does not answer only costs a suggestion.
    The list view above already reports an unreachable runtime once, loudly; saying it again
    beside two optional fields would be noise.
  */
  useEffect(() => {
    fetch("http://127.0.0.1:8000/faces", { cache: "no-store" })
      .then(async (response) => (response.ok ? ((await response.json()) as FaceOption[]) : []))
      .then(setFaces)
      .catch(() => setFaces([]));

    fetch("http://127.0.0.1:8000/config", { cache: "no-store" })
      .then(async (response) =>
        response.ok ? ((await response.json()) as RuntimeConfig) : null,
      )
      .then(setRuntime)
      .catch(() => setRuntime(null));
  }, []);

  /*
    The runtime's live value first, then the static ones, deduplicated.

    It goes first because it is the strongest suggestion available: whatever `/config` reports is
    serving on this host right now, with its keys and base URL already resolved. A static name
    below it may or may not be reachable from this machine.
  */
  const suggest = (live: string | undefined, fallback: readonly string[]): readonly string[] => [
    ...new Set([live, ...fallback].filter((value): value is string => Boolean(value))),
  ];

  // The API rejects this too, and it is the API's rule. Checking it here as well only saves
  // a round trip and puts the explanation next to the field that caused it.
  const hysteresisInverted = Number(releaseProbability) >= Number(onsetProbability);

  async function submit() {
    setBusy(true);
    setProblem(null);
    try {
      const response = await fetch(editing ? `${API}/${agent!.id}` : API, {
        method: editing ? "PATCH" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          system_prompt: systemPrompt,
          llm_provider: llmProvider,
          llm_model: llmModel,
          voice_provider: voiceProvider,
          voice_id: voiceId,
          // An empty box is "no face", not a face whose id is the empty string — the second
          // would validate here and fail when the renderer went looking for it.
          face_id: faceId.trim() || null,
          turn_taking: {
            onset_probability: Number(onsetProbability),
            release_probability: Number(releaseProbability),
            onset_frames: Number(onsetFrames),
            min_speech_ms: Number(minSpeechMs),
            end_of_turn_silence_ms: Number(endOfTurnMs),
          },
        }),
      });

      if (!response.ok) {
        const payload: unknown = await response.json().catch(() => null);
        throw new Error(problemFrom(payload, response.status));
      }
      onCreated();
    } catch (cause) {
      setProblem(
        cause instanceof Error
          ? cause.message
          : `Could not ${editing ? "save" : "create"} the agent.`,
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader
        title={editing ? `Editing ${agent!.name}` : "New agent"}
        hint={
          editing
            ? "Changes apply to interviews created from now on. Sessions already recorded keep the configuration they ran under, which is why a report stays readable after an agent is retuned."
            : "Provider defaults are the credential-free pair, so a new agent runs on a clean clone with no keys. Knowledge, tools, guardrails and pronunciations attach after the agent exists — they are references, and typing an id that does not exist yet fails at session start rather than here."
        }
        action={onCancel ? <Button onClick={onCancel}>Cancel</Button> : undefined}
      />

      <div className="grid gap-4 px-5 py-5 sm:grid-cols-2">
        <Field label="Name" hint="How this interviewer is identified everywhere else">
          <Input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Backend screening interviewer"
          />
        </Field>

        {/* A closed dropdown, unlike the model and voice fields below: a face is a row in this
            product's own database, so the full valid set is known here and anything outside it is
            a typo that would validate now and fail at session start. */}
        <Field
          label="Face"
          hint={
            faces === null
              ? "Reading the faces the runtime knows about…"
              : faces.length === 0
                ? "No faces yet. An agent without one runs on the placeholder renderer; create a face on the Faces screen and attach it below."
                : "Or none, which runs the placeholder renderer"
          }
        >
          <Select
            value={faceId}
            disabled={faces === null || faces.length === 0}
            onChange={(event) => setFaceId(event.target.value)}
          >
            <option value="">
              {faces && faces.length === 0 ? "No faces yet" : "None — placeholder renderer"}
            </option>
            {(faces ?? []).map((face) => (
              <option key={face.id} value={face.id}>
                {/* Status is on the option because it changes what happens at session start: an
                    unprepared face pays its enrollment cost on the first turn instead of having
                    paid it offline, which is the whole reason Prepare exists. */}
                {face.name}
                {face.status === "ready" ? "" : ` — ${face.status}, not prepared`}
              </option>
            ))}
          </Select>
        </Field>

        <Field label="LLM provider" hint="scripted needs no key and no network">
          <Select
            value={llmProvider}
            onChange={(event) => setLlmProvider(event.target.value as LlmProvider)}
          >
            <option value="scripted">scripted</option>
            <option value="anthropic">anthropic</option>
            <option value="openai">openai</option>
          </Select>
        </Field>

        <Field
          label="LLM model"
          hint={
            llmProvider === "scripted"
              ? "scripted answers from a fixed script and has no model to choose"
              : "Suggestions, not the valid set — the provider owns that list. Empty means the adapter's own default."
          }
        >
          <Combo
            id="llm-model"
            value={llmModel}
            suggestions={suggest(
              runtime?.llm === llmProvider ? runtime?.llm_model : undefined,
              MODEL_SUGGESTIONS[llmProvider],
            )}
            disabled={llmProvider === "scripted"}
            onChange={(event) => setLlmModel(event.target.value)}
            placeholder={
              llmProvider === "scripted"
                ? "not used"
                : (runtime?.llm === llmProvider ? runtime?.llm_model : "") || "adapter default"
            }
          />
        </Field>

        <Field label="Voice provider" hint="tone is the credential-free placeholder voice">
          <Select
            value={voiceProvider}
            onChange={(event) => setVoiceProvider(event.target.value as VoiceProvider)}
          >
            <option value="tone">tone</option>
            <option value="deepgram">deepgram</option>
            <option value="clone">clone (a voice you uploaded)</option>
          </Select>
        </Field>

        <Field
          label="Voice"
          hint={
            voiceProvider === "tone"
              ? "tone emits a sine wave at the right length, not speech — it has no voices"
              : voiceProvider === "clone"
              ? "Cloned voices come from the Voices screen — attach one below, not here"
              : "Any Aura voice. Suggestions only; Deepgram owns the list. Empty means the adapter's default."
          }
        >
          <Combo
            id="voice-id"
            value={voiceId}
            suggestions={suggest(
              runtime?.tts === voiceProvider ? runtime?.tts_voice : undefined,
              VOICE_SUGGESTIONS[voiceProvider],
            )}
            disabled={voiceProvider === "tone"}
            onChange={(event) => setVoiceId(event.target.value)}
            placeholder={
              voiceProvider === "tone"
                ? "not used"
                : (runtime?.tts === voiceProvider ? runtime?.tts_voice : "") || "adapter default"
            }
          />
        </Field>

        <div className="sm:col-span-2">
          <Field label="System prompt" hint="What the interviewer is told before the first turn">
            <Textarea
              value={systemPrompt}
              onChange={(event) => setSystemPrompt(event.target.value)}
              placeholder="You are interviewing a backend engineer. Ask about failure modes, and follow up on anything vague."
            />
          </Field>
        </div>

        {/*
          Turn-taking is a section rather than an "advanced" disclosure. Three of these five
          numbers decide whether the avatar interrupts a cough or talks over a pause, and the
          fifth is most of the latency budget — none of it is incidental.
        */}
        <div className="border-t border-hair pt-4 sm:col-span-2">
          <p className="text-[12px] font-medium tracking-wide text-ink">Turn taking</p>
          <p className="mt-1 max-w-2xl text-[11.5px] leading-relaxed text-ink-mid">
            When the candidate is considered to have started talking, and when they are
            considered to have finished. Onset must be certain, because acting on it
            interrupts the avatar; end of turn must be patient, because answering into a
            thinking pause is worse than waiting.
          </p>
        </div>

        <Field
          label="End-of-turn silence (ms)"
          hint={`The largest single term in a turn that already measures ${TURN_BUDGET}, and pure configuration — no faster GPU reduces it. Raising it stops the avatar answering into a thinking pause; every millisecond added is added to the candidate's wait.`}
        >
          <Input
            type="number"
            min={0}
            step={50}
            value={endOfTurnMs}
            onChange={(event) => setEndOfTurnMs(event.target.value)}
          />
        </Field>

        <Field
          label="Minimum speech (ms)"
          hint={`Below this the turn is retracted rather than answered. Default ${DEFAULTS.min_speech_ms}ms — shorter than a syllable is not an utterance.`}
        >
          <Input
            type="number"
            min={0}
            step={50}
            value={minSpeechMs}
            onChange={(event) => setMinSpeechMs(event.target.value)}
          />
        </Field>

        <Field
          label="Onset probability"
          hint={`How confident before it counts as speech. High on purpose: a false positive cuts the avatar off mid-sentence, a false negative costs one frame. Default ${DEFAULTS.onset_probability}.`}
        >
          <Input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={onsetProbability}
            onChange={(event) => setOnsetProbability(event.target.value)}
          />
        </Field>

        <Field
          label="Release probability"
          hint={`How confident to stay in speech. Must sit below onset — the gap is the hysteresis that stops the dip inside a word from ending the turn. Default ${DEFAULTS.release_probability}.`}
        >
          <Input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={releaseProbability}
            onChange={(event) => setReleaseProbability(event.target.value)}
          />
        </Field>

        <Field
          label="Onset frames"
          hint={`Consecutive frames above the onset bar before speech is declared, at ~32ms each. Default ${DEFAULTS.onset_frames} — the cheapest defence against a cough or a door.`}
        >
          <Input
            type="number"
            min={1}
            step={1}
            value={onsetFrames}
            onChange={(event) => setOnsetFrames(event.target.value)}
          />
        </Field>

        {hysteresisInverted ? (
          <p className="text-[12.5px] leading-relaxed text-bad sm:col-span-2">
            Release probability ({releaseProbability}) must be below onset probability (
            {onsetProbability}). With no gap between them, the probability dip inside an
            ordinary word both fails to sustain speech and ends the turn, so the avatar
            answers mid-sentence.
          </p>
        ) : null}

        {problem ? (
          <p className="text-[12.5px] leading-relaxed text-bad sm:col-span-2">{problem}</p>
        ) : null}

        <div className="flex justify-end sm:col-span-2">
          <Button
            variant="primary"
            disabled={busy || !name.trim() || hysteresisInverted}
            onClick={() => void submit()}
          >
            {busy ? (editing ? "Saving…" : "Creating…") : editing ? "Save changes" : "Create agent"}
          </Button>
        </div>
      </div>
    </Card>
  );
}
