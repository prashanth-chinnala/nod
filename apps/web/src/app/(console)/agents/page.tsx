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

import {
  Button,
  Card,
  CardHeader,
  Cell,
  Chip,
  Empty,
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
type VoiceProvider = "deepgram" | "tone";

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
  face_id: string | null;
  knowledge_base_ids: string[];
  tool_ids: string[];
  guardrail_id: string | null;
  pronunciation_id: string | null;
  turn_taking: TurnTaking;
  created_at: string;
  updated_at: string;
};

const HEAD = ["name", "model", "voice", "face", "end of turn", "updated"] as const;

/** An unset optional reference reads as a dash, not as an empty cell that looks broken. */
const UNSET = "—";

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
      {showForm ? (
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
                <Cell dim mono>
                  {agent.face_id ?? UNSET}
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
              </Row>
            ))}
          </Table>
        )}
      </Card>
    </Page>
  );
}

/**
 * Inline create form, toggled rather than shown in a modal.
 *
 * A modal would put the turn-taking numbers on top of the table they have to be judged
 * against, and it would have nowhere to grow when the attachment pickers arrive.
 */
function CreateForm({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [llmProvider, setLlmProvider] = useState<LlmProvider>("scripted");
  const [llmModel, setLlmModel] = useState("");
  const [voiceProvider, setVoiceProvider] = useState<VoiceProvider>("tone");
  const [voiceId, setVoiceId] = useState("");
  const [faceId, setFaceId] = useState("");
  const [onsetProbability, setOnsetProbability] = useState(String(DEFAULTS.onset_probability));
  const [releaseProbability, setReleaseProbability] = useState(
    String(DEFAULTS.release_probability),
  );
  const [onsetFrames, setOnsetFrames] = useState(String(DEFAULTS.onset_frames));
  const [minSpeechMs, setMinSpeechMs] = useState(String(DEFAULTS.min_speech_ms));
  const [endOfTurnMs, setEndOfTurnMs] = useState(String(DEFAULTS.end_of_turn_silence_ms));
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  // The API rejects this too, and it is the API's rule. Checking it here as well only saves
  // a round trip and puts the explanation next to the field that caused it.
  const hysteresisInverted = Number(releaseProbability) >= Number(onsetProbability);

  async function submit() {
    setBusy(true);
    setProblem(null);
    try {
      const response = await fetch(API, {
        method: "POST",
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
      setProblem(cause instanceof Error ? cause.message : "Could not create the agent.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader
        title="New agent"
        hint="Provider defaults are the credential-free pair, so a new agent runs on a clean clone with no keys. Knowledge, tools, guardrails and pronunciations attach after the agent exists — they are references, and typing an id that does not exist yet fails at session start rather than here."
      />

      <div className="grid gap-4 px-5 py-5 sm:grid-cols-2">
        <Field label="Name" hint="How this interviewer is identified everywhere else">
          <Input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Backend screening interviewer"
          />
        </Field>

        <Field label="Face" hint="A face id, or leave empty for the placeholder renderer">
          <Input
            value={faceId}
            onChange={(event) => setFaceId(event.target.value)}
            placeholder="face_…"
            spellCheck={false}
          />
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

        <Field label="LLM model" hint="Empty means the adapter's own default">
          <Input
            value={llmModel}
            onChange={(event) => setLlmModel(event.target.value)}
            placeholder="claude-sonnet-4-5"
            spellCheck={false}
          />
        </Field>

        <Field label="Voice provider" hint="tone is the credential-free placeholder voice">
          <Select
            value={voiceProvider}
            onChange={(event) => setVoiceProvider(event.target.value as VoiceProvider)}
          >
            <option value="tone">tone</option>
            <option value="deepgram">deepgram</option>
          </Select>
        </Field>

        <Field label="Voice" hint="Provider voice id; empty means the adapter's default">
          <Input
            value={voiceId}
            onChange={(event) => setVoiceId(event.target.value)}
            placeholder="aura-asteria-en"
            spellCheck={false}
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
            {busy ? "Creating…" : "Create agent"}
          </Button>
        </div>
      </div>
    </Card>
  );
}
