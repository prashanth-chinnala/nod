"use client";

/**
 * The agent attachment editor: which rubric, face, guardrail, lexicon, knowledge bases and tools
 * an agent references.
 *
 * **Why this exists as its own screen section rather than in the create form.** These are
 * references to resources that must already exist, so a create form would have to either offer
 * empty pickers on a fresh install or accept typed ids that validate and then fail at session
 * start. The agents page said so in a comment and deferred them "to a picker built once those
 * collections are populated" — they are, and until this existed every resource in the console
 * could be created and none could be attached without curl. A console that configures things the
 * runtime cannot be told about is a demo of a console.
 *
 * **Why each control saves on change rather than behind a Save button.** Every field here is a
 * discrete choice from a list, so there is no half-typed state to protect, and a batched save
 * would need its own dirty-tracking and a way to discard. Saving immediately also means the
 * failure lands on the control that caused it, which matters because these writes fail for a
 * reason the operator can act on — a resource deleted in another tab.
 *
 * **Why detaching is a real option and not just a cosmetic one.** "None" sends `null`, which the
 * store now applies rather than dropping. That was a bug worth fixing before building this: with
 * nulls discarded, this picker would have offered a "None" that appeared to work and silently
 * changed nothing, which is worse than not offering it.
 */

import { useCallback, useEffect, useState } from "react";

import { Button, Card, CardHeader, Chip, Field, Select } from "@/components/ui";

const API = "http://127.0.0.1:8000";

/** Every collection an agent can point at, with the field that holds the reference. */
const SINGLE = [
  {
    field: "rubric_id",
    collection: "rubrics",
    label: "Rubric",
    hint: "Competencies the interview works through and the scorer judges against",
  },
  {
    field: "face_id",
    collection: "faces",
    label: "Face",
    hint: "Reference persona. The stub renderer ignores it until a real one runs",
  },
  {
    field: "guardrail_id",
    collection: "guardrails",
    label: "Guardrail",
    hint: "Input and output policy, enforced on every turn",
  },
  {
    field: "pronunciation_id",
    collection: "pronunciations",
    label: "Pronunciation",
    hint: "Lexicon applied to text before synthesis, never to history",
  },
] as const;

const MULTI = [
  {
    field: "knowledge_base_ids",
    collection: "knowledge",
    label: "Knowledge bases",
    hint: "Indexed into one retriever at session start, so paragraphs rank on the same scale",
  },
  {
    field: "tool_ids",
    collection: "tools",
    label: "Tools",
    hint: "Offered to the model each turn. A missing one is fatal at session start, not here",
  },
] as const;

type Named = { id: string; name?: string };

type Agent = {
  id: string;
  name: string;
  rubric_id: string | null;
  face_id: string | null;
  guardrail_id: string | null;
  pronunciation_id: string | null;
  knowledge_base_ids: string[];
  tool_ids: string[];
};

type Catalog = Record<string, Named[]>;

const COLLECTIONS = [...SINGLE, ...MULTI].map((entry) => entry.collection);

/**
 * Load every referencable collection once.
 *
 * One pass rather than a fetch per picker: six sequential requests would make the panel appear
 * a control at a time, and an operator would start choosing before the list they need has
 * arrived. A collection that fails to load resolves to empty rather than failing the whole
 * catalogue, so one missing resource type does not hide the other five.
 */
function useCatalog(): { catalog: Catalog; reload: () => void } {
  const [catalog, setCatalog] = useState<Catalog>({});

  const reload = useCallback(() => {
    Promise.all(
      COLLECTIONS.map((collection) =>
        fetch(`${API}/${collection}`, { cache: "no-store" })
          .then((response) => (response.ok ? response.json() : []))
          .then((items) => [collection, items as Named[]] as const)
          .catch(() => [collection, [] as Named[]] as const),
      ),
    ).then((pairs) => setCatalog(Object.fromEntries(pairs)));
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  return { catalog, reload };
}

/** A resource's display name, falling back to its id so a nameless record is still selectable. */
function label(item: Named): string {
  return item.name?.trim() ? `${item.name} · ${item.id}` : item.id;
}

export function Attachments({
  agents,
  onChanged,
}: {
  agents: Agent[];
  onChanged: () => void;
}) {
  const { catalog, reload } = useCatalog();
  // `null` until the operator picks one, with the effective agent derived below. The obvious
  // shape -- state seeded from `agents[0]` in an effect -- is what React's compiler lint rejects,
  // and it is right to: the list reloads after every write, so a sync effect has to be written
  // carefully enough not to bounce the selection back to the top agent mid-edit. Deriving removes
  // the possibility rather than guarding against it.
  const [chosen, setChosen] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);

  // Falls back to the first agent, and also covers the case where the chosen one was deleted in
  // another tab: the panel lands on a real agent instead of rendering an empty editor.
  const agent = agents.find((candidate) => candidate.id === chosen) ?? agents[0] ?? null;

  const patch = useCallback(
    async (field: string, value: string | string[] | null) => {
      if (!agent) return;
      setBusy(field);
      setProblem(null);
      try {
        const response = await fetch(`${API}/agents/${agent.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ [field]: value }),
        });
        if (!response.ok) {
          const body = (await response.json().catch(() => null)) as { detail?: unknown } | null;
          const detail = typeof body?.detail === "string" ? body.detail : response.status;
          throw new Error(`the runtime rejected this (${detail})`);
        }
        setSaved(field);
        onChanged();
      } catch (cause) {
        setProblem(cause instanceof Error ? cause.message : "could not save");
      } finally {
        setBusy(null);
      }
    },
    [agent, onChanged],
  );

  // Clear the transient "saved" marker without leaving a timer running past unmount.
  useEffect(() => {
    if (!saved) return;
    const timer = setTimeout(() => setSaved(null), 1600);
    return () => clearTimeout(timer);
  }, [saved]);

  return (
    <Card>
      <CardHeader
        title="Attachments"
        hint="What this agent references. Saved on change, and a reference that no longer exists makes the session refuse to start rather than run without it — loud beats a plausible interview that quietly ignored its configuration."
        action={<Button onClick={reload}>Reload lists</Button>}
      />

      <div className="space-y-5 px-5 py-5">
        <div className="flex flex-wrap items-end gap-4">
          <div className="min-w-64 flex-1">
            <Field label="Agent" hint="Attachments below belong to this one">
              <Select
                value={agent?.id ?? ""}
                onChange={(event) => setChosen(event.target.value)}
              >
                {agents.length === 0 ? <option value="">no agents yet</option> : null}
                {agents.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} · {item.id}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          {saved ? <Chip status="ok">saved</Chip> : null}
        </div>

        {problem ? <p className="text-[12.5px] text-bad">{problem}</p> : null}

        {agent === null ? (
          <p className="text-[12.5px] text-ink-mid">
            Create an agent first — attachments hang off one.
          </p>
        ) : (
          <>
            <div className="grid gap-4 sm:grid-cols-2">
              {SINGLE.map((entry) => {
                const options = catalog[entry.collection] ?? [];
                const current = agent[entry.field];
                return (
                  <Field key={entry.field} label={entry.label} hint={entry.hint}>
                    <Select
                      value={current ?? ""}
                      disabled={busy === entry.field}
                      onChange={(event) =>
                        // Empty string is the "None" option, and it becomes a real null rather
                        // than an empty string: the runtime treats "" and null differently --
                        // an empty id would be looked up and reported as a missing resource.
                        void patch(entry.field, event.target.value || null)
                      }
                    >
                      <option value="">
                        {options.length === 0
                          ? `none — no ${entry.collection} exist yet`
                          : "none"}
                      </option>
                      {options.map((item) => (
                        <option key={item.id} value={item.id}>
                          {label(item)}
                        </option>
                      ))}
                      {/* A reference to something since deleted would otherwise vanish from the
                          control and read as "nothing attached", while the session still refuses
                          to start. Shown explicitly so the operator can see what to fix. */}
                      {current && !options.some((item) => item.id === current) ? (
                        <option value={current}>{current} — missing</option>
                      ) : null}
                    </Select>
                  </Field>
                );
              })}
            </div>

            {MULTI.map((entry) => {
              const options = catalog[entry.collection] ?? [];
              const current = agent[entry.field];
              return (
                <div key={entry.field}>
                  <p className="text-[11px] font-medium tracking-[0.07em] uppercase text-ink-low">
                    {entry.label}
                  </p>
                  <p className="mt-1 text-[11.5px] text-ink-low">{entry.hint}</p>
                  {options.length === 0 ? (
                    <p className="mt-2 text-[12.5px] text-ink-mid">
                      No {entry.collection} exist yet.
                    </p>
                  ) : (
                    <div className="mt-2.5 flex flex-wrap gap-2">
                      {options.map((item) => {
                        const on = current.includes(item.id);
                        return (
                          <button
                            key={item.id}
                            type="button"
                            aria-pressed={on}
                            disabled={busy === entry.field}
                            onClick={() =>
                              void patch(
                                entry.field,
                                on
                                  ? current.filter((id) => id !== item.id)
                                  : [...current, item.id],
                              )
                            }
                            className={[
                              "rounded-lg border px-3 py-1.5 text-[12.5px] transition-colors",
                              on
                                ? "border-accent/50 bg-accent/15 text-accent"
                                : "border-hair-strong bg-glass-raise text-ink-mid hover:border-ink-low hover:text-ink",
                            ].join(" ")}
                          >
                            {label(item)}
                          </button>
                        );
                      })}
                    </div>
                  )}
                  {/* Same missing-reference case as above, for the list fields. */}
                  {current
                    .filter((id) => !options.some((item) => item.id === id))
                    .map((id) => (
                      <p key={id} className="mt-2 text-[12px] text-bad">
                        {id} is attached but no longer exists — sessions for this agent will
                        refuse to start until it is removed.
                        <button
                          type="button"
                          className="ml-2 underline"
                          onClick={() =>
                            void patch(
                              entry.field,
                              current.filter((existing) => existing !== id),
                            )
                          }
                        >
                          Remove it
                        </button>
                      </p>
                    ))}
                </div>
              );
            })}
          </>
        )}
      </div>
    </Card>
  );
}
