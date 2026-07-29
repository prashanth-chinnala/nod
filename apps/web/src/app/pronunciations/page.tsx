"use client";

/**
 * Pronunciation lexicons.
 *
 * The screen is arranged around the one question an operator actually has here — "will it
 * say this word right?" — so the preview is a first-class panel rather than a detail buried
 * in the editor. It calls the server's `/apply` endpoint instead of reimplementing the
 * substitution in JavaScript: a browser-side copy would reassure the operator about
 * replacements the synthesiser will not make, and the two edge cases (whole-word only, no
 * re-substitution inside a replacement) are exactly where a second implementation drifts.
 *
 * Consequence, stated because it is visible: the preview reflects *saved* entries. Editing a
 * row does not move it until Save lands. That is the honest reading — it shows what TTS
 * would receive right now.
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
} from "@/components/ui";

const API = "http://127.0.0.1:8000";

const PREVIEW_DEBOUNCE_MS = 250;
/**
 * The preview is a round trip per keystroke without this.
 *
 * 250ms is short enough to feel live and long enough that a typed sentence is one request
 * rather than forty — and forty in-flight requests would also land out of order, so the box
 * would settle on whichever reply was slowest rather than the latest text.
 */

type Entry = { term: string; say: string };

type Lexicon = {
  id: string;
  name: string;
  entries: Entry[];
  created_at: string;
  updated_at: string;
};

function jsonBody(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  };
}

/**
 * One fetch wrapper, because every caller needs the same two things: a non-2xx turned into
 * a throw, and the API's `detail` surfaced rather than a bare status. A 422 from the lexicon
 * validators carries the reason the save was refused, and dropping it would leave the
 * operator with "failed" and no idea which row was wrong.
 */
async function call(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(`${API}${path}`, init);
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body: unknown = await response.json();
      if (body && typeof body === "object" && "detail" in body) {
        detail = JSON.stringify((body as { detail: unknown }).detail);
      }
    } catch {
      // A non-JSON error body is still an error; the status carries enough.
    }
    throw new Error(detail);
  }
  return response;
}

export default function PronunciationsPage() {
  const [lexicons, setLexicons] = useState<Lexicon[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [newName, setNewName] = useState("");

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [draft, setDraft] = useState<Entry[]>([]);

  const [tryText, setTryText] = useState("We run nginx in front of PostgreSQL.");
  const [preview, setPreview] = useState("");
  // Bumped on every successful save so the preview re-runs against the new stored entries.
  const [revision, setRevision] = useState(0);

  const selected = lexicons?.find((lexicon) => lexicon.id === selectedId) ?? null;

  const load = useCallback(async () => {
    try {
      const response = await call("/pronunciations");
      setLexicons((await response.json()) as Lexicon[]);
      setError(null);
    } catch (cause) {
      setLexicons([]);
      setError(cause instanceof Error ? cause.message : "unreachable");
    }
  }, []);

  useEffect(() => {
    // Wrapped in an async callback rather than called directly: React 19's compiler treats a
    // bare `load()` in an effect body as a synchronous setState and refuses it. The wrapper
    // is not decoration, it is what makes the state write happen in a callback.
    void (async () => {
      await load();
    })();
  }, [load]);

  useEffect(() => {
    // The empty cases are handled inside the timer rather than as an early return, so the
    // preview always settles through one code path. An early `setPreview("")` in the effect
    // body would also be a synchronous render-triggering write, which React 19's compiler
    // rejects outright.
    const timer = window.setTimeout(() => {
      void (async () => {
        if (selectedId === null || tryText === "") {
          setPreview("");
          return;
        }
        try {
          const response = await call(
            `/pronunciations/${selectedId}/apply`,
            jsonBody("POST", { text: tryText }),
          );
          setPreview(((await response.json()) as { text: string }).text);
        } catch (cause) {
          setPreview("");
          setError(cause instanceof Error ? cause.message : "preview failed");
        }
      })();
    }, PREVIEW_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [selectedId, tryText, revision]);

  /**
   * Selecting seeds the editor from the stored record — here rather than in an effect on
   * `lexicons`, so that a list refresh triggered by a save elsewhere on the page cannot
   * discard half-typed rows.
   */
  function select(lexicon: Lexicon) {
    setSelectedId(lexicon.id);
    setDraft(lexicon.entries.map((entry) => ({ ...entry })));
  }

  async function mutate(work: () => Promise<void>) {
    setBusy(true);
    try {
      await work();
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "request failed");
    } finally {
      setBusy(false);
    }
  }

  function create() {
    void mutate(async () => {
      const response = await call("/pronunciations", jsonBody("POST", { name: newName }));
      const created = (await response.json()) as Lexicon;
      setNewName("");
      await load();
      select(created);
    });
  }

  function save() {
    if (selectedId === null) return;
    void mutate(async () => {
      // Rows where both fields are still empty are abandoned "Add term" clicks, not
      // intent, so they are dropped. A half-filled row is *not* dropped — it goes to the
      // server and comes back a 422, because silently discarding it is how a term the
      // operator thought they added goes missing.
      const entries = draft.filter((entry) => entry.term !== "" || entry.say !== "");
      await call(`/pronunciations/${selectedId}`, jsonBody("PATCH", { entries }));
      await load();
      setRevision((n) => n + 1);
    });
  }

  function remove(id: string) {
    void mutate(async () => {
      await call(`/pronunciations/${id}`, { method: "DELETE" });
      if (selectedId === id) {
        setSelectedId(null);
        setDraft([]);
      }
      await load();
    });
  }

  function editEntry(index: number, patch: Partial<Entry>) {
    setDraft((rows) => rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  }

  return (
    <Page
      title="Pronunciations"
      lede="Per-term overrides applied to the interviewer's text before synthesis. Whichever voice AVATAR_TTS resolved to, these are what it receives — no model change and no re-render involved."
      action={
        error ? (
          <Chip status="bad">{error}</Chip>
        ) : lexicons === null ? (
          <Chip status="info">loading</Chip>
        ) : (
          <Chip status="ok">{lexicons.length} lexicons</Chip>
        )
      }
    >
      <Card>
        <CardHeader
          title="New lexicon"
          hint="A lexicon is a named set of overrides — one per interview track is usually the right grain, since the words that break are domain-specific."
        />
        <div className="flex items-end gap-3 px-5 py-4">
          <div className="min-w-0 flex-1">
            <Field label="Name">
              <Input
                value={newName}
                placeholder="Infrastructure terms"
                onChange={(event) => setNewName(event.target.value)}
              />
            </Field>
          </div>
          <Button variant="primary" disabled={busy || newName.trim() === ""} onClick={create}>
            Create
          </Button>
        </div>
      </Card>

      <Card>
        <CardHeader title="Lexicons" hint="Newest first. Select one to edit its terms." />
        {lexicons === null ? (
          <p className="px-5 py-10 text-center text-[12.5px] text-ink-mid">Loading…</p>
        ) : lexicons.length === 0 ? (
          <Empty title="No lexicons yet">
            {error
              ? "The console could not reach the runtime on :8000. Start it with `uvicorn avatar.server:app` and this table will fill in."
              : "Every TTS voice mispronounces nginx, PostgreSQL, and Kubernetes. Name a lexicon above, add those three terms, and the interviewer stops sounding like it has never worked here."}
          </Empty>
        ) : (
          <Table head={["Name", "Terms", "Updated", ""]}>
            {lexicons.map((lexicon) => (
              <Row key={lexicon.id}>
                <Cell>
                  <button
                    type="button"
                    onClick={() => select(lexicon)}
                    className={
                      lexicon.id === selectedId
                        ? "text-left font-medium text-accent"
                        : "text-left text-ink hover:text-accent"
                    }
                  >
                    {lexicon.name}
                  </button>
                  <span className="mt-0.5 block font-mono text-[11px] text-ink-low">
                    {lexicon.id}
                  </span>
                </Cell>
                <Cell dim>{lexicon.entries.length}</Cell>
                <Cell dim mono>
                  {lexicon.updated_at}
                </Cell>
                <Cell right>
                  <Button variant="danger" disabled={busy} onClick={() => remove(lexicon.id)}>
                    Delete
                  </Button>
                </Cell>
              </Row>
            ))}
          </Table>
        )}
      </Card>

      {selected ? (
        <>
          <Card>
            <CardHeader
              title={`Terms — ${selected.name}`}
              hint="Matching is case-insensitive and whole-word, so “Kafka” never touches “Kafkaesque”. Write the replacement the way you want it heard, not spelled."
              action={
                <div className="flex gap-2">
                  <Button
                    disabled={busy}
                    onClick={() => setDraft((rows) => [...rows, { term: "", say: "" }])}
                  >
                    Add term
                  </Button>
                  <Button variant="primary" disabled={busy} onClick={save}>
                    Save
                  </Button>
                </div>
              }
            />
            {draft.length === 0 ? (
              <Empty
                title="No terms in this lexicon"
                action={
                  <Button onClick={() => setDraft([{ term: "", say: "" }])}>Add term</Button>
                }
              >
                An empty lexicon is a no-op — text passes through untouched. Add the words
                this interview track keeps getting wrong.
              </Empty>
            ) : (
              <div className="space-y-3 px-5 py-4">
                {draft.map((entry, index) => (
                  <div key={index} className="flex items-end gap-3">
                    <div className="min-w-0 flex-1">
                      <Field label="Written">
                        <Input
                          value={entry.term}
                          placeholder="nginx"
                          onChange={(event) => editEntry(index, { term: event.target.value })}
                        />
                      </Field>
                    </div>
                    <div className="min-w-0 flex-1">
                      <Field label="Spoken">
                        <Input
                          value={entry.say}
                          placeholder="engine ex"
                          onChange={(event) => editEntry(index, { say: event.target.value })}
                        />
                      </Field>
                    </div>
                    <Button
                      variant="danger"
                      onClick={() =>
                        setDraft((rows) => rows.filter((_, i) => i !== index))
                      }
                    >
                      Remove
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card>
            <CardHeader
              title="Try it"
              hint="Runs the server's substitution, so this is literally what the synthesiser would receive. Reflects saved terms — press Save to see an edit here."
            />
            <div className="space-y-4 px-5 py-4">
              <Field label="Text" hint="Paste a sentence the avatar keeps getting wrong.">
                <Input
                  value={tryText}
                  onChange={(event) => setTryText(event.target.value)}
                />
              </Field>
              <div>
                <p className="mb-1.5 text-[12px] font-medium text-ink">Sent to TTS</p>
                <p className="rounded-lg border border-hair bg-glass-raise px-3 py-2 text-[13px] leading-relaxed text-ink">
                  {preview === "" ? (
                    <span className="text-ink-low">
                      {tryText === ""
                        ? "Type something above."
                        : "…"}
                    </span>
                  ) : (
                    preview
                  )}
                </p>
                {preview !== "" && preview === tryText ? (
                  <p className="mt-2 text-[11.5px] text-ink-low">
                    Unchanged — no term in this lexicon matched.
                  </p>
                ) : null}
              </div>
            </div>
          </Card>
        </>
      ) : (
        <Card>
          <Empty title="No lexicon selected">
            Select a lexicon above to edit its terms and preview what the synthesiser would
            receive.
          </Empty>
        </Card>
      )}
    </Page>
  );
}
