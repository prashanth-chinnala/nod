"use client";

/**
 * Guardrails — the input and output policy attached to a session.
 *
 * The checker at the bottom is the point of this screen, not an extra. A guardrail is a rule
 * written in a text box and enforced somewhere the operator never sees; the failure mode is
 * not "it blocks too much", it is "the topic list has a typo and nobody finds out until an
 * interview goes wrong". So the page pairs every policy with a way to run real text through
 * it and see the verdict — a guardrail nobody can test is a guardrail nobody trusts.
 *
 * The checker calls the runtime's `/check`, which is a local string and regex pass, so what
 * is shown here is exactly what the turn will do. It is not an approximation reimplemented
 * in TypeScript: a second copy of the matching rules would drift, and the drift would appear
 * as a policy that tested clean and refused in production.
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

const API = "http://127.0.0.1:8000/guardrails";

/** Mirrors the router's defaults and bounds, so hint text cannot drift from enforcement. */
const DEFAULT_MAX_ANSWER_CHARS = 600;
const MAX_ANSWER_CHARS_CEILING = 20000;

type OnViolation = "refuse" | "redirect" | "end_session";
type Direction = "input" | "output";

type Guardrail = {
  id: string;
  name: string;
  banned_topics: string[];
  pii_redaction: boolean;
  max_answer_chars: number;
  refusal_message: string;
  on_violation: OnViolation;
  created_at: string;
  updated_at: string;
};

type CheckResult = {
  allowed: boolean;
  violations: string[];
  redacted_text: string;
};

const HEAD = ["name", "banned topics", "pii", "max answer", "on violation", ""] as const;

/**
 * Turn a 422 body into one sentence an operator can act on.
 *
 * FastAPI's validation errors and this router's merged-patch errors have different shapes,
 * and neither is presentable. Falling back to the status code is deliberate: a vague message
 * naming the field is still better than a rendered `[object Object]`.
 */
function explain(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((entry) => {
        const item = entry as { field?: unknown; loc?: unknown; msg?: unknown; message?: unknown };
        const where = Array.isArray(item.loc)
          ? item.loc.filter((p) => p !== "body").join(".")
          : typeof item.field === "string"
            ? item.field
            : "";
        const what = typeof item.msg === "string" ? item.msg : String(item.message ?? "");
        return where ? `${where}: ${what}` : what;
      })
      .filter(Boolean);
    if (parts.length > 0) return parts.join("; ");
  }
  return `The runtime rejected this (${status}).`;
}

/**
 * Split a free-text topic box into terms.
 *
 * Empty fragments are dropped here rather than posted: a trailing comma is the most common
 * thing a form submits, and the router rejects blank terms outright — an empty term would
 * match every string, so it would be a policy that refuses every turn. Dropping the fragment
 * is what the operator meant; showing them a 422 for a comma is not.
 */
function parseTopics(raw: string): string[] {
  return raw
    .split(/[,\n]/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

/** PII is repaired and the turn continues; a topic or length hit stops it. Tone says which. */
function violationTone(code: string): "warn" | "bad" {
  return code.startsWith("pii:") ? "warn" : "bad";
}

export default function GuardrailsPage() {
  const [items, setItems] = useState<Guardrail[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  // A promise chain rather than async/await, and not because it reads better: React's
  // `set-state-in-effect` rule cannot see past an awaited memoized callback and treats
  // `void load()` in an effect as a synchronous setState. Resolving the state updates inside
  // `.then`/`.catch` is the shape the rule endorses — state arriving from an external system
  // in a callback — and it keeps one loader shared by the effect, the retry button, and the
  // reload after a create or delete.
  const load = useCallback(() => {
    fetch(API, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`the runtime answered ${response.status}`);
        return (await response.json()) as Guardrail[];
      })
      .then((loaded) => {
        setItems(loaded);
        setError(null);
      })
      .catch((cause: unknown) => {
        // The runtime is a separate process on :8000. "Failed to fetch" alone reads as a bug
        // in this page, so the message has to name the thing that is actually down.
        setItems(null);
        setError(cause instanceof Error ? cause.message : "the runtime is unreachable");
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <Page
      title="Guardrails"
      lede="Input and output policy for a session. Checks are local string and regex passes, never a model call: the output side runs between the LLM and TTS, where a network round trip would be silence the candidate hears."
      action={
        <Button variant="primary" onClick={() => setShowForm((open) => !open)}>
          {showForm ? "Close" : "New guardrail"}
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
          title="Configured guardrails"
          hint="on violation is handed to whoever owns the turn; this resource detects and reports, it does not decide. PII hits are redacted and the turn continues — a candidate must not lose their answer for mentioning their own email."
        />

        {error ? (
          <Empty
            title="Could not reach the runtime"
            action={<Button onClick={load}>Retry</Button>}
          >
            {error}. The console reads guardrails from the runtime on 127.0.0.1:8000 — start it
            with <code className="font-mono text-ink">uvicorn avatar.server:app</code> and retry.
          </Empty>
        ) : items === null ? (
          <div className="px-5 py-14 text-center text-[12.5px] text-ink-mid">
            Loading guardrails…
          </div>
        ) : items.length === 0 ? (
          <Empty
            title="No guardrails yet"
            action={
              <Button variant="primary" onClick={() => setShowForm(true)}>
                New guardrail
              </Button>
            }
          >
            Without one, every session runs unbounded: no banned topics, no PII redaction, and
            no ceiling on answer length — and answer length is latency, since every character
            is synthesised and then rendered. Create the first policy, then test it in the
            checker below before attaching it to an agent.
          </Empty>
        ) : (
          <Table head={HEAD}>
            {items.map((item) => (
              <PolicyRow key={item.id} item={item} onChanged={load} />
            ))}
          </Table>
        )}
      </Card>

      <Checker items={items ?? []} />
    </Page>
  );
}

/**
 * One row, with a two-step delete.
 *
 * Arming before deleting because this list is scanned quickly and a policy is referenced by
 * live sessions: a single misplaced click must not silently remove the only thing bounding
 * what the avatar will say.
 */
function PolicyRow({ item, onChanged }: { item: Guardrail; onChanged: () => void }) {
  const [armed, setArmed] = useState(false);

  async function remove() {
    await fetch(`${API}/${item.id}`, { method: "DELETE" });
    setArmed(false);
    onChanged();
  }

  return (
    <Row>
      <Cell>{item.name}</Cell>
      <Cell dim>
        {item.banned_topics.length === 0 ? "none" : item.banned_topics.join(", ")}
      </Cell>
      <Cell>
        <Chip status={item.pii_redaction ? "ok" : "neutral"}>
          {item.pii_redaction ? "redacting" : "off"}
        </Chip>
      </Cell>
      <Cell mono dim>
        {item.max_answer_chars} chars
      </Cell>
      <Cell dim>{item.on_violation}</Cell>
      <Cell right>
        {armed ? (
          <span className="flex justify-end gap-2">
            <Button variant="danger" onClick={() => void remove()}>
              Confirm delete
            </Button>
            <Button onClick={() => setArmed(false)}>Cancel</Button>
          </span>
        ) : (
          <Button onClick={() => setArmed(true)}>Delete</Button>
        )}
      </Cell>
    </Row>
  );
}

/**
 * Create form.
 *
 * Surfaces the router's 422 verbatim rather than validating in parallel here. Two copies of
 * the rules would drift, and the copy that matters is the one the store enforces.
 */
function CreateForm({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [topics, setTopics] = useState("");
  const [pii, setPii] = useState("on");
  const [maxChars, setMaxChars] = useState(String(DEFAULT_MAX_ANSWER_CHARS));
  const [onViolation, setOnViolation] = useState<OnViolation>("refuse");
  const [refusal, setRefusal] = useState("I'd rather not go into that. Tell me about your own work instead.");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setProblem(null);
    try {
      const response = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          banned_topics: parseTopics(topics),
          pii_redaction: pii === "on",
          max_answer_chars: Number(maxChars),
          refusal_message: refusal,
          on_violation: onViolation,
        }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as unknown;
        throw new Error(explain(body, response.status));
      }
      onCreated();
    } catch (cause) {
      setProblem(cause instanceof Error ? cause.message : "Could not create the guardrail.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader
        title="New guardrail"
        hint="Topic matching is whole-term and case-insensitive, so a ban on “ai” does not fire on “said”. It is literal: a topic reached by paraphrase walks through, and no setting here changes that."
      />
      <div className="grid gap-4 px-5 py-5 sm:grid-cols-2">
        <Field label="Name" hint="How this policy is identified when attached to an agent">
          <Input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Standard interview policy"
          />
        </Field>

        <Field
          label="Max answer characters"
          hint={`Output side only — it bounds what the avatar says, never what the candidate says. Every character is synthesised and rendered, so this is a latency setting. Max ${MAX_ANSWER_CHARS_CEILING}.`}
        >
          <Input
            type="number"
            min={1}
            max={MAX_ANSWER_CHARS_CEILING}
            value={maxChars}
            onChange={(event) => setMaxChars(event.target.value)}
          />
        </Field>

        <Field label="PII redaction" hint="Redacts email addresses and phone/card-like digit runs">
          <Select value={pii} onChange={(event) => setPii(event.target.value)}>
            <option value="on">on</option>
            <option value="off">off</option>
          </Select>
        </Field>

        <Field label="On violation" hint="What the turn owner does when a check comes back blocked">
          <Select
            value={onViolation}
            onChange={(event) => setOnViolation(event.target.value as OnViolation)}
          >
            <option value="refuse">refuse</option>
            <option value="redirect">redirect</option>
            <option value="end_session">end_session</option>
          </Select>
        </Field>

        <div className="sm:col-span-2">
          <Field label="Banned topics" hint="One per line, or comma separated">
            <Textarea
              value={topics}
              onChange={(event) => setTopics(event.target.value)}
              placeholder={"salary\nvisa status"}
              spellCheck={false}
            />
          </Field>
        </div>

        <div className="sm:col-span-2">
          <Field
            label="Refusal message"
            hint="Spoken when a check blocks a turn. It cannot be blank: silence after a question reads as a crashed avatar, not as a policy."
          >
            <Textarea value={refusal} onChange={(event) => setRefusal(event.target.value)} />
          </Field>
        </div>

        {problem ? (
          <p className="text-[12.5px] leading-relaxed text-bad sm:col-span-2">{problem}</p>
        ) : null}

        <div className="flex justify-end sm:col-span-2">
          <Button variant="primary" disabled={busy || !name.trim()} onClick={() => void submit()}>
            {busy ? "Creating…" : "Create guardrail"}
          </Button>
        </div>
      </div>
    </Card>
  );
}

/**
 * The live checker.
 *
 * Runs against the same endpoint the turn uses, so the verdict here is the verdict there. It
 * shows the redacted text as well as the verdict, because that string — not the original — is
 * what reaches TTS, and an operator who has not seen it does not know what the avatar will
 * actually say.
 */
function Checker({ items }: { items: Guardrail[] }) {
  const [chosen, setChosen] = useState("");
  const [direction, setDirection] = useState<Direction>("output");
  const [text, setText] = useState("");
  const [result, setResult] = useState<CheckResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  // Derived rather than synced in an effect: the list arrives after first paint, and an
  // effect that copies it into state would leave a stale id behind after a delete.
  const activeId = items.some((item) => item.id === chosen) ? chosen : (items[0]?.id ?? "");

  async function run() {
    setBusy(true);
    setProblem(null);
    try {
      const response = await fetch(`${API}/${activeId}/check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, direction }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as unknown;
        throw new Error(explain(body, response.status));
      }
      setResult((await response.json()) as CheckResult);
    } catch (cause) {
      setResult(null);
      setProblem(cause instanceof Error ? cause.message : "The check could not run.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader
        title="Check text against a policy"
        hint="Same local pass the runtime performs on a turn — no model call, so this costs microseconds and matches production exactly."
      />

      {items.length === 0 ? (
        <Empty title="Nothing to test yet">
          The checker runs a policy that exists. Create a guardrail above, then paste a sample
          answer here to see which terms trip and what the avatar would be left saying.
        </Empty>
      ) : (
        <div className="grid gap-4 px-5 py-5 sm:grid-cols-2">
          <Field label="Guardrail" hint="The policy to evaluate against">
            <Select value={activeId} onChange={(event) => setChosen(event.target.value)}>
              {items.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.name}
                </option>
              ))}
            </Select>
          </Field>

          <Field
            label="Direction"
            hint="input is the candidate's transcript; output is generated text on its way to TTS, where the length cap applies"
          >
            <Select
              value={direction}
              onChange={(event) => setDirection(event.target.value as Direction)}
            >
              <option value="output">output</option>
              <option value="input">input</option>
            </Select>
          </Field>

          <div className="sm:col-span-2">
            <Field label="Text" hint="Paste a real sample. A policy tested on nothing proves nothing.">
              <Textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="My salary expectation is 120000 — mail me at ada@example.com."
              />
            </Field>
          </div>

          <div className="flex items-center justify-end gap-3 sm:col-span-2">
            {problem ? <p className="flex-1 text-[12.5px] text-bad">{problem}</p> : null}
            <Button variant="primary" disabled={busy || !text.trim()} onClick={() => void run()}>
              {busy ? "Checking…" : "Run check"}
            </Button>
          </div>

          {result ? (
            <div className="space-y-3 border-t border-hair pt-4 sm:col-span-2">
              <div className="flex flex-wrap items-center gap-2">
                <Chip status={result.allowed ? "ok" : "bad"}>
                  {result.allowed ? "allowed" : "blocked"}
                </Chip>
                {result.violations.length === 0 ? (
                  <span className="text-[12.5px] text-ink-mid">no violations</span>
                ) : (
                  result.violations.map((code) => (
                    <Chip key={code} status={violationTone(code)}>
                      {code}
                    </Chip>
                  ))
                )}
              </div>

              <div>
                <p className="text-[11px] font-medium tracking-[0.06em] uppercase text-ink-low">
                  what would reach TTS
                </p>
                <p className="mt-1.5 font-mono text-[12px] leading-relaxed text-ink">
                  {result.redacted_text || "(nothing)"}
                </p>
                {result.redacted_text === text ? null : (
                  <p className="mt-1.5 text-[11.5px] text-ink-low">
                    Rewritten by the policy — redaction and truncation both change this string.
                  </p>
                )}
              </div>
            </div>
          ) : null}
        </div>
      )}
    </Card>
  );
}
