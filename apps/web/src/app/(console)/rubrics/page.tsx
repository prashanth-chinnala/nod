"use client";

/**
 * Rubrics — the competency plan an interview is conducted and scored against.
 *
 * **Why this screen is a list-of-lists and not a flat form.** A rubric is competencies, and a
 * competency is four decisions that only make sense together: what to probe, what counts as
 * evidence, how many turns it may consume, and how much it weighs. Splitting them across screens
 * would hide the two relationships an operator has to see — that `min_signals` must be reachable
 * from the signal list, and that declared order is the running order while weight is not.
 *
 * **Order is meaningful and the page says so.** The runtime probes the first competency that is
 * still open, in declared order, so this list is a priority list rather than a set. That is easy
 * to miss in a table, so the position is numbered and the reordering controls are on every row.
 *
 * **The two ways a rubric silently does nothing are surfaced, not validated away.** A competency
 * with no signals can never be evidenced — it gets probed `max_turns` times and reported as
 * exhausted — and a rubric with no competencies steers nothing at all. Both are legitimate drafts,
 * so the API saves them and returns warnings; this renders those warnings on the row that caused
 * them. The alternative, refusing to save, teaches an operator to invent signals to satisfy a
 * validator.
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
  Table,
  Textarea,
  num,
} from "@/components/ui";

const API = "http://127.0.0.1:8000/rubrics";

/** Mirrors `avatar.plan`'s defaults, so hint text cannot drift from what the runtime does. */
const DEFAULT_MAX_TURNS = 3;
const DEFAULT_MIN_SIGNALS = 1;
const DEFAULT_WEIGHT = 1;

type Competency = {
  id?: string;
  name: string;
  probe: string;
  signals: string[];
  max_turns: number;
  min_signals: number;
  weight: number;
};

type Rubric = {
  id: string;
  name: string;
  description: string;
  competencies: Competency[];
  warnings?: string[];
  created_at: string;
  updated_at: string;
};

/** The editor's own shape: signals stay a string while being typed. */
type Draft = Omit<Competency, "signals"> & { signals: string };

const HEAD = [
  "#",
  "competency",
  "signals",
  num("max turns"),
  num("min signals"),
  num("weight"),
  "",
] as const;

function blankDraft(): Draft {
  return {
    name: "",
    probe: "",
    signals: "",
    max_turns: DEFAULT_MAX_TURNS,
    min_signals: DEFAULT_MIN_SIGNALS,
    weight: DEFAULT_WEIGHT,
  };
}

/**
 * Turn a 422 body into one sentence an operator can act on.
 *
 * Same shape as the other resource pages: FastAPI's validation errors and this router's
 * model-validator errors have different shapes and neither is presentable. Naming the field beats
 * a rendered `[object Object]`, and falling back to the status code beats saying nothing.
 */
function explain(body: unknown, status: number): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((entry) => {
        const item = entry as { loc?: unknown; msg?: unknown };
        const where = Array.isArray(item.loc)
          ? item.loc.filter((part) => part !== "body").join(".")
          : "";
        const what = typeof item.msg === "string" ? item.msg : "";
        return where ? `${where}: ${what}` : what;
      })
      .filter(Boolean);
    if (parts.length > 0) return parts.join("; ");
  }
  return `The runtime rejected this (${status}).`;
}

/**
 * Split a signal box into terms.
 *
 * Newline or comma, and blanks dropped: a trailing comma is the most common thing a form submits,
 * and an empty signal would match nothing while still counting toward `min_signals` — so it would
 * quietly raise the bar the operator thought they were lowering.
 */
function parseSignals(raw: string): string[] {
  return raw
    .split(/[,\n]/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

/** The same slug the API stamps, so the preview matches the id a report will cite. */
function slug(name: string): string {
  return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
}

export default function RubricsPage() {
  const [items, setItems] = useState<Rubric[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false);

  // A promise chain rather than async/await: React's `set-state-in-effect` rule cannot see past
  // an awaited memoized callback and treats `void load()` in an effect as a synchronous setState.
  const load = useCallback(() => {
    fetch(API, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`the runtime answered ${response.status}`);
        return (await response.json()) as Rubric[];
      })
      .then((loaded) => {
        setItems(loaded);
        setError(null);
      })
      .catch((cause: unknown) => {
        setItems(null);
        setError(cause instanceof Error ? cause.message : "the runtime is unreachable");
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <Page
      title="Rubrics"
      lede="The competencies an interview works through, in priority order. The runtime probes the first one still open, credits evidence from any answer, and the scorer judges each one separately after the session ends."
      action={
        <Button variant="primary" onClick={() => setShowForm((open) => !open)}>
          {showForm ? "Close" : "New rubric"}
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
          title="Configured rubrics"
          hint="Declared order is the running order — the interviewer stays on a competency until it is evidenced or its max turns are spent, then moves down. Weight is separate on purpose: it only affects the scorecard, so you can ask about something first and count it least."
        />

        {error ? (
          <Empty
            title="Could not reach the runtime"
            action={<Button onClick={load}>Retry</Button>}
          >
            {error}. The console reads rubrics from the runtime on 127.0.0.1:8000 — start it with{" "}
            <code className="font-mono text-ink">uvicorn avatar.server:app</code> and retry.
          </Empty>
        ) : items === null ? (
          <div className="px-5 py-14 text-center text-[12.5px] text-ink-mid">
            Loading rubrics…
          </div>
        ) : items.length === 0 ? (
          <Empty
            title="No rubrics yet"
            action={
              <Button variant="primary" onClick={() => setShowForm(true)}>
                New rubric
              </Button>
            }
          >
            Without one an interview has no agenda: it will answer whatever the candidate raises
            and cannot tell that it spent eight turns on deployment and never asked about
            debugging. A rubric is also what the scorer judges against — no rubric means no
            scorecard, only a transcript.
          </Empty>
        ) : (
          items.map((item) => <RubricCard key={item.id} item={item} onChanged={load} />)
        )}
      </Card>
    </Page>
  );
}

/**
 * One rubric, with its competencies and its warnings.
 *
 * A card each rather than one flat table, because the competency list is the content — a table of
 * rubrics would show six names and hide everything that determines what the interview does.
 */
function RubricCard({ item, onChanged }: { item: Rubric; onChanged: () => void }) {
  const [armed, setArmed] = useState(false);

  async function remove() {
    await fetch(`${API}/${item.id}`, { method: "DELETE" });
    setArmed(false);
    onChanged();
  }

  const blind = new Set(
    (item.warnings ?? [])
      .filter((note) => note.startsWith("no signals for "))
      .flatMap((note) =>
        note
          .slice("no signals for ".length)
          .split(" — ")[0]
          .split(", ")
          .map((name) => name.trim()),
      ),
  );

  return (
    <div className="border-t border-hair first:border-t-0">
      <div className="flex flex-wrap items-start gap-3 px-5 py-4">
        <div className="min-w-0 flex-1">
          <p className="text-[13.5px] font-medium text-ink">{item.name}</p>
          {item.description ? (
            <p className="mt-0.5 text-[12px] text-ink-mid">{item.description}</p>
          ) : null}
          <p className="mt-1 font-mono text-[11px] text-ink-low">{item.id}</p>
        </div>
        <Chip status={item.competencies.length > 0 ? "ok" : "warn"}>
          {item.competencies.length} competenc{item.competencies.length === 1 ? "y" : "ies"}
        </Chip>
        {armed ? (
          <span className="flex gap-2">
            <Button variant="danger" onClick={() => void remove()}>
              Confirm delete
            </Button>
            <Button onClick={() => setArmed(false)}>Cancel</Button>
          </span>
        ) : (
          <Button onClick={() => setArmed(true)}>Delete</Button>
        )}
      </div>

      {/* Warnings on the rubric, above the rows, because they describe what it will *fail* to do
          — an operator scanning the numbers would not otherwise learn that a competency can
          never be evidenced. */}
      {(item.warnings ?? []).length > 0 ? (
        <div className="mx-5 mb-4 rounded-lg border border-warn/40 bg-warn/5 px-4 py-3">
          {(item.warnings ?? []).map((note) => (
            <p key={note} className="text-[12px] leading-relaxed text-warn">
              {note}
            </p>
          ))}
        </div>
      ) : null}

      {item.competencies.length > 0 ? (
        <div className="px-1 pb-3">
          <Table head={HEAD}>
            {item.competencies.map((competency, index) => (
              <Row key={competency.id ?? competency.name}>
                {/* Position, not an index: this is the order the interview runs in. */}
                <Cell mono dim>
                  {index + 1}
                </Cell>
                <Cell>
                  <span className="block text-ink">{competency.name}</span>
                  {competency.probe ? (
                    <span className="mt-0.5 block text-[11.5px] text-ink-mid">
                      {competency.probe}
                    </span>
                  ) : null}
                </Cell>
                <Cell dim>
                  {competency.signals.length === 0 ? (
                    <Chip status="warn">none — cannot be evidenced</Chip>
                  ) : (
                    competency.signals.join(", ")
                  )}
                </Cell>
                <Cell mono dim right>
                  {competency.max_turns}
                </Cell>
                <Cell mono dim right>
                  {competency.min_signals}
                </Cell>
                <Cell mono dim right>
                  {competency.weight}
                </Cell>
                <Cell right>
                  {blind.has(competency.name) ? <Chip status="warn">no signals</Chip> : null}
                </Cell>
              </Row>
            ))}
          </Table>
        </div>
      ) : null}
    </div>
  );
}

/**
 * Create form: a rubric and its competencies in one submit.
 *
 * One submit rather than create-then-add-rows, because the API validates across the list — two
 * competencies whose names reduce to the same id are rejected together, and `min_signals` is
 * checked against its own signal list. Adding rows one at a time would report those failures one
 * at a time, after the operator had already committed to a shape.
 */
function CreateForm({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [rows, setRows] = useState<Draft[]>([blankDraft()]);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  function update(index: number, patch: Partial<Draft>) {
    setRows((current) =>
      current.map((row, position) => (position === index ? { ...row, ...patch } : row)),
    );
  }

  function move(index: number, delta: number) {
    setRows((current) => {
      const next = [...current];
      const target = index + delta;
      if (target < 0 || target >= next.length) return current;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }

  // Duplicate ids are rejected by the API; showing them here as well is not a second copy of the
  // rule, it is the same rule made visible before submitting — the slug is deterministic.
  const slugs = rows.map((row) => slug(row.name)).filter(Boolean);
  const duplicated = new Set(slugs.filter((value, index) => slugs.indexOf(value) !== index));

  async function submit() {
    setBusy(true);
    setProblem(null);
    try {
      const response = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          description,
          competencies: rows
            .filter((row) => row.name.trim().length > 0)
            .map((row) => ({
              name: row.name,
              probe: row.probe,
              signals: parseSignals(row.signals),
              max_turns: Number(row.max_turns),
              min_signals: Number(row.min_signals),
              weight: Number(row.weight),
            })),
        }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as unknown;
        throw new Error(explain(body, response.status));
      }
      onCreated();
    } catch (cause) {
      setProblem(cause instanceof Error ? cause.message : "Could not create the rubric.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader
        title="New rubric"
        hint="Probe is what to explore, in your words — not a question to read out. The runtime hands it to the model and the model writes the sentence, which is what lets it follow up on an answer instead of reading from a script."
      />
      <div className="space-y-5 px-5 py-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Name" hint="Shown on the agent form and in every report">
            <Input value={name} onChange={(event) => setName(event.target.value)} />
          </Field>
          <Field label="Description" hint="Optional — what role this is for">
            <Input
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
          </Field>
        </div>

        <div className="space-y-4">
          {rows.map((row, index) => (
            <div
              key={index}
              className="rounded-lg border border-hair-strong bg-glass-raise px-4 py-4"
            >
              <div className="mb-3 flex items-center gap-2">
                <span className="font-mono text-[11px] text-ink-low">#{index + 1}</span>
                {row.name ? (
                  <span className="font-mono text-[11px] text-ink-low">{slug(row.name)}</span>
                ) : null}
                {duplicated.has(slug(row.name)) ? (
                  <Chip status="bad">duplicate id — rename one</Chip>
                ) : null}
                <span className="ml-auto flex gap-1.5">
                  <Button onClick={() => move(index, -1)}>↑</Button>
                  <Button onClick={() => move(index, 1)}>↓</Button>
                  <Button
                    variant="danger"
                    onClick={() =>
                      setRows((current) =>
                        current.length === 1
                          ? current
                          : current.filter((_, position) => position !== index),
                      )
                    }
                  >
                    Remove
                  </Button>
                </span>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <Field label="Competency" hint="e.g. Debugging under pressure">
                  <Input
                    value={row.name}
                    onChange={(event) => update(index, { name: event.target.value })}
                  />
                </Field>
                <Field
                  label="Signals"
                  hint="Comma or newline separated. Matched whole-word, so C++ and .NET work"
                >
                  <Input
                    value={row.signals}
                    onChange={(event) => update(index, { signals: event.target.value })}
                  />
                </Field>
              </div>

              <div className="mt-4">
                <Field label="Probe" hint="What to explore — not a question to read out">
                  <Textarea
                    rows={2}
                    value={row.probe}
                    onChange={(event) => update(index, { probe: event.target.value })}
                  />
                </Field>
              </div>

              <div className="mt-4 grid gap-4 sm:grid-cols-3">
                <Field label="Max turns" hint="Then the interview moves on">
                  <Input
                    type="number"
                    min={1}
                    max={20}
                    value={row.max_turns}
                    onChange={(event) =>
                      update(index, { max_turns: Number(event.target.value) })
                    }
                  />
                </Field>
                <Field label="Min signals" hint="Distinct hits before it counts as evidenced">
                  <Input
                    type="number"
                    min={1}
                    max={20}
                    value={row.min_signals}
                    onChange={(event) =>
                      update(index, { min_signals: Number(event.target.value) })
                    }
                  />
                </Field>
                <Field label="Weight" hint="Scorecard only, not the running order">
                  <Input
                    type="number"
                    min={0.1}
                    max={10}
                    step={0.1}
                    value={row.weight}
                    onChange={(event) => update(index, { weight: Number(event.target.value) })}
                  />
                </Field>
              </div>

              {/* The one cross-field rule worth pre-empting: a bar the signal list cannot clear
                  is not a strict competency, it is one that can never be evidenced. */}
              {parseSignals(row.signals).length > 0 &&
              row.min_signals > new Set(parseSignals(row.signals)).size ? (
                <p className="mt-3 text-[12px] text-bad">
                  Min signals ({row.min_signals}) is above the{" "}
                  {new Set(parseSignals(row.signals)).size} distinct signal
                  {new Set(parseSignals(row.signals)).size === 1 ? "" : "s"} listed — this
                  competency could never be evidenced.
                </p>
              ) : null}
            </div>
          ))}

          <Button onClick={() => setRows((current) => [...current, blankDraft()])}>
            Add competency
          </Button>
        </div>

        {problem ? <p className="text-[12.5px] text-bad">{problem}</p> : null}

        <div className="flex items-center gap-3">
          <Button variant="primary" disabled={busy || !name.trim()} onClick={() => void submit()}>
            {busy ? "Creating…" : "Create rubric"}
          </Button>
          <p className="text-[11.5px] text-ink-low">
            Competencies with no name are dropped rather than rejected.
          </p>
        </div>
      </div>
    </Card>
  );
}
