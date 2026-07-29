"use client";

/**
 * Tools — functions the agent may call mid-interview.
 *
 * The screen is organised around the one number that decides whether a tool is usable:
 * `timeout_ms`. A tool call is a round trip *inside* a conversational turn that already
 * measures 2.7–5.8s, so the timeout is not a technical detail buried in a detail pane — it
 * is the column an operator scans, and the field the create form leads with after identity.
 *
 * What this page deliberately does not show is measured p95 per tool, which is what would
 * actually tell you whether a configured deadline is achievable. Nothing has executed a
 * tool yet, so there is no number, and an invented one would be worse than the gap.
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

const API = "http://127.0.0.1:8000/tools";

/** Mirrors the router's bounds. Kept as constants so the hint text cannot drift from them. */
const TIMEOUT_MAX_MS = 5000;
const DEFAULT_TIMEOUT_MS = 1500;

/** The measured full-turn range from PROCESS.md §3.4. Quoted, never recomputed here. */
const TURN_BUDGET = "2.7–5.8s";

type Kind = "http" | "builtin";

type Tool = {
  id: string;
  name: string;
  description: string;
  parameters_schema: Record<string, unknown>;
  kind: Kind;
  url: string | null;
  timeout_ms: number;
  enabled: boolean;
  created_at: string;
  updated_at: string;
};

const HEAD = ["name", "kind", "timeout", "enabled", "updated"] as const;

/**
 * Timeout tone is a judgement, not decoration.
 *
 * Anything at or above half the fastest observed turn is flagged: a 3s tool on top of a
 * 2.7s turn is a different product, and the table should say so before someone ships it.
 */
function timeoutStatus(ms: number): "ok" | "warn" | "bad" {
  if (ms > 2500) return "bad";
  if (ms > 1500) return "warn";
  return "ok";
}

export default function ToolsPage() {
  const [tools, setTools] = useState<Tool[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  /*
    Every state write happens after the await, never synchronously on entry. Clearing the
    error up front would be a setState in the effect body, which React 19's compiler rules
    reject for a real reason: it schedules a second render before the first has committed.
  */
  const load = useCallback(async () => {
    try {
      const response = await fetch(API, { cache: "no-store" });
      if (!response.ok) throw new Error(`the runtime answered ${response.status}`);
      const rows = (await response.json()) as Tool[];
      setTools(rows);
      setError(null);
    } catch (cause) {
      // The runtime is a separate process on :8000. "Failed to fetch" on its own reads as a
      // bug in this page, so the message has to name the thing that is actually down.
      setTools(null);
      setError(cause instanceof Error ? cause.message : "the runtime is unreachable");
    }
  }, []);

  useEffect(() => {
    void (async () => {
      await load();
    })();
  }, [load]);

  return (
    <Page
      title="Tools"
      lede={`Functions the agent may call mid-interview. Each call is a round trip inside a turn that already measures ${TURN_BUDGET}, so the timeout is the field that decides whether a tool is usable at all.`}
      action={
        <Button variant="primary" onClick={() => setShowForm((open) => !open)}>
          {showForm ? "Close" : "New tool"}
        </Button>
      }
    >
      {showForm ? (
        <CreateForm
          onCreated={() => {
            setShowForm(false);
            void load();
          }}
        />
      ) : null}

      <Card>
        <CardHeader
          title="Configured tools"
          hint={`Timeout is a hard per-call deadline, capped at ${TIMEOUT_MAX_MS}ms. Measured p95 per tool is not shown: no tool has executed yet, so there is no number to report.`}
        />

        {error ? (
          <Empty title="Could not reach the runtime" action={<Button onClick={() => void load()}>Retry</Button>}>
            {error}. The console reads tools from the runtime on 127.0.0.1:8000 — start it
            with <code className="font-mono text-ink">uvicorn avatar.server:app</code> and
            retry.
          </Empty>
        ) : tools === null ? (
          <div className="px-5 py-14 text-center text-[12.5px] text-ink-mid">Loading tools…</div>
        ) : tools.length === 0 ? (
          <Empty
            title="No tools yet"
            action={<Button variant="primary" onClick={() => setShowForm(true)}>New tool</Button>}
          >
            A tool is a function the interviewer can call while it is talking — scoring an
            answer, looking up history, ending the session. Until one exists the agent can
            only converse. Add the first one and give it a deadline well inside the{" "}
            {TURN_BUDGET} turn.
          </Empty>
        ) : (
          <Table head={HEAD}>
            {tools.map((tool) => (
              <Row key={tool.id}>
                <Cell mono>{tool.name}</Cell>
                <Cell dim>{tool.kind}</Cell>
                <Cell>
                  {/* Prominent on purpose: this is the number that stalls an interview. */}
                  <Chip status={timeoutStatus(tool.timeout_ms)}>{tool.timeout_ms}ms</Chip>
                </Cell>
                <Cell>
                  <Chip status={tool.enabled ? "ok" : "neutral"}>
                    {tool.enabled ? "enabled" : "disabled"}
                  </Chip>
                </Cell>
                <Cell dim mono>
                  {tool.updated_at}
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
 * Create form.
 *
 * Submits to the same endpoint the table reads and surfaces the router's 422 verbatim
 * rather than validating in parallel here. Two copies of the name pattern and the timeout
 * bound would drift, and the copy that matters is the one the store enforces.
 */
function CreateForm({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [kind, setKind] = useState<Kind>("builtin");
  const [url, setUrl] = useState("");
  const [timeoutMs, setTimeoutMs] = useState(String(DEFAULT_TIMEOUT_MS));
  const [schema, setSchema] = useState("{}");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  async function submit() {
    setBusy(true);
    setProblem(null);
    try {
      let parameters_schema: unknown;
      try {
        parameters_schema = JSON.parse(schema || "{}");
      } catch {
        // Parsed here rather than posted as a string, because the failure is local and the
        // server's message for it ("input is not a valid dict") would not point at the box.
        throw new Error("Parameters schema is not valid JSON.");
      }

      const response = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          description,
          kind,
          url: kind === "http" ? url : null,
          timeout_ms: Number(timeoutMs),
          parameters_schema,
        }),
      });

      if (!response.ok) {
        const detail = (await response.json().catch(() => null)) as {
          detail?: unknown;
        } | null;
        throw new Error(
          typeof detail?.detail === "string"
            ? detail.detail
            : `The runtime rejected this tool (${response.status}). Check the name pattern, the url, and the timeout.`,
        );
      }
      onCreated();
    } catch (cause) {
      setProblem(cause instanceof Error ? cause.message : "Could not create the tool.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader
        title="New tool"
        hint="The name becomes the function name the model emits, so it must be lowercase snake_case. An http tool needs a url — one without an endpoint is registered with the model and silently never fires."
      />
      <div className="grid gap-4 px-5 py-5 sm:grid-cols-2">
        <Field label="Name" hint="lowercase, digits and underscores, e.g. score_answer">
          <Input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="score_answer"
            spellCheck={false}
          />
        </Field>

        <Field
          label="Timeout (ms)"
          hint={`Hard per-call deadline, max ${TIMEOUT_MAX_MS}. This sits inside a turn already measuring ${TURN_BUDGET} — every millisecond here is added to the candidate's wait.`}
        >
          <Input
            type="number"
            min={1}
            max={TIMEOUT_MAX_MS}
            value={timeoutMs}
            onChange={(event) => setTimeoutMs(event.target.value)}
          />
        </Field>

        <Field label="Kind" hint="builtin dispatches in-process; http calls out">
          <Select value={kind} onChange={(event) => setKind(event.target.value as Kind)}>
            <option value="builtin">builtin</option>
            <option value="http">http</option>
          </Select>
        </Field>

        <Field
          label="URL"
          hint={kind === "http" ? "Required for an http tool." : "Not used by a builtin."}
        >
          <Input
            value={url}
            onChange={(event) => setUrl(event.target.value)}
            disabled={kind !== "http"}
            placeholder="http://127.0.0.1:9001/score"
            spellCheck={false}
          />
        </Field>

        <div className="sm:col-span-2">
          <Field label="Description" hint="What the model reads to decide when to call this">
            <Textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="Score the candidate's last answer from 1 to 5."
            />
          </Field>
        </div>

        <div className="sm:col-span-2">
          <Field label="Parameters schema" hint="JSON Schema, passed to the model verbatim">
            <Textarea
              value={schema}
              onChange={(event) => setSchema(event.target.value)}
              spellCheck={false}
              className="font-mono text-[12px]"
            />
          </Field>
        </div>

        {problem ? (
          <p className="text-[12.5px] leading-relaxed text-bad sm:col-span-2">{problem}</p>
        ) : null}

        <div className="flex justify-end sm:col-span-2">
          <Button variant="primary" disabled={busy || !name.trim()} onClick={() => void submit()}>
            {busy ? "Creating…" : "Create tool"}
          </Button>
        </div>
      </div>
    </Card>
  );
}
