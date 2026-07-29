"use client";

/**
 * Knowledge bases, and a retrieval tester for them.
 *
 * The tester is the reason this page is worth building. A knowledge base is otherwise a
 * black box: when the interviewer asks a question that ignores the job description, there is
 * no way to tell whether retrieval returned the wrong chunks or the model ignored the right
 * ones — and those two bugs live in completely different places. Showing the chunks a query
 * would pull, with their scores, makes that one glance instead of a guess.
 *
 * Scores are shown as bare numbers with no colour threshold. BM25 scores are not calibrated
 * across corpora, so "0.8 is good" is not a true statement about any base, and a green chip
 * would assert a threshold this system does not have.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import {
  Button,
  Card,
  CardHeader,
  Cell,
  Chip,
  Empty,
  Field,
  Input,
  Metric,
  Page,
  Row,
  Select,
  Table,
  num,
} from "@/components/ui";

/*
  The runtime is a separate process on :8000, as the sidebar already says. Hardcoded rather
  than proxied through a Next route handler: a proxy would add a hop and a second place for
  the URL to be wrong, and this console is operated next to the server it configures.
*/
const API = "http://127.0.0.1:8000";

const TOP_K_CHOICES = [3, 5, 10] as const;

type KnowledgeDocument = {
  id: string;
  filename: string;
  chunk_count: number;
};

type KnowledgeBase = {
  id: string;
  name: string;
  description?: string;
  /* Optional on purpose: records are JSON files an operator can edit by hand, so the page
     must survive one that predates a field rather than crashing the whole route. */
  documents?: KnowledgeDocument[];
  chunk_count?: number;
  total_chars?: number;
  updated_at?: string;
};

type Hit = {
  text: string;
  score: number;
  document_id: string;
};

function describe(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export default function KnowledgePage() {
  const [bases, setBases] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState("");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState<number>(TOP_K_CHOICES[0]);
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  /*
    Every state update here happens after the first `await`, and `loading` starts out true
    rather than being set at the top of this function. Setting it synchronously would make
    the mount effect below update state during the effect body, which React flags as a
    cascading render — and the pending state is already correct on first paint anyway.
  */
  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const response = await fetch(`${API}/knowledge`, { signal, cache: "no-store" });
      if (!response.ok) {
        throw new Error(`${API}/knowledge responded ${response.status}`);
      }
      const rows = (await response.json()) as KnowledgeBase[];
      setBases(rows);
      /* Keep the operator's selection across a reload; fall back to the first base only if
         the one they had chosen is gone. Resetting unconditionally would yank the tester back
         to another base every time the list refreshed. */
      setSelectedId((current) =>
        rows.some((row) => row.id === current) ? current : (rows[0]?.id ?? ""),
      );
      setLoadError(null);
      setLoading(false);
    } catch (error) {
      /* An aborted request is a cleanup, not a failure. Reporting it would flash an error
         on every mount in React's development double-render. */
      if (signal?.aborted) return;
      setLoadError(describe(error));
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    /* Awaited from inside the effect rather than called from its body, so every state
       update lands in a later microtask. The abort in the cleanup is what stops a discarded
       first render — React mounts effects twice in development — from overwriting the state
       the second one already fetched. */
    const run = async () => {
      await load(controller.signal);
    };
    void run();
    return () => controller.abort();
  }, [load]);

  const reload = useCallback(() => {
    setLoading(true);
    void load();
  }, [load]);

  const selected = useMemo(
    () => bases.find((base) => base.id === selectedId) ?? null,
    [bases, selectedId],
  );

  /* Chunks carry a document id, not a filename — a hit an operator cannot trace back to a
     file tells them nothing they can act on, so resolve it here. */
  const filenames = useMemo(() => {
    const byId = new Map<string, string>();
    for (const doc of selected?.documents ?? []) {
      byId.set(doc.id, doc.filename);
    }
    return byId;
  }, [selected]);

  const runQuery = useCallback(async () => {
    const probe = query.trim();
    if (!selected || !probe) return;

    setRunning(true);
    setQueryError(null);
    try {
      const response = await fetch(`${API}/knowledge/${selected.id}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: probe, top_k: topK }),
      });
      if (!response.ok) {
        throw new Error(`query responded ${response.status}`);
      }
      setHits((await response.json()) as Hit[]);
    } catch (error) {
      setQueryError(describe(error));
      setHits(null);
    } finally {
      setRunning(false);
    }
  }, [query, selected, topK]);

  const corpusEmpty = (selected?.chunk_count ?? 0) === 0;

  return (
    <Page
      title="Knowledge"
      lede="Documents chunked on paragraph boundaries and retrieved by keyword score. No embedding model and no vector store: a retrieval hop inside a conversational turn has no latency budget to spare, and keyword matching is competitive over a handful of short documents."
      action={
        <Button onClick={reload} disabled={loading}>
          {loading ? "Loading…" : "Reload"}
        </Button>
      }
    >
      <Card>
        <CardHeader
          title="Knowledge bases"
          hint="Newest first. Chunk and character counts are derived from the stored corpus, not reported by the uploader."
          action={loadError ? <Chip status="bad">unreachable</Chip> : null}
        />

        {loading ? (
          <p className="px-5 py-10 text-center text-[12.5px] text-ink-mid">
            Loading knowledge bases…
          </p>
        ) : loadError ? (
          <Empty
            title="Could not reach the runtime"
            action={<Button onClick={reload}>Retry</Button>}
          >
            <span className="block">{loadError}</span>
            <span className="mt-2 block">
              The console reads from the API process on :8000. Start it with{" "}
              <span className="font-mono text-[11.5px] text-ink">pnpm api</span> and retry.
            </span>
          </Empty>
        ) : bases.length === 0 ? (
          <Empty title="No knowledge bases yet">
            A knowledge base is what makes the interviewer ask about <em>this</em> role rather
            than generically strong questions: paste in the job description or a rubric and
            its paragraphs become retrievable chunks. Create one with{" "}
            <span className="font-mono text-[11.5px] text-ink">
              curl -X POST localhost:8000/knowledge -H &apos;content-type:
              application/json&apos; -d &apos;{"{"}&quot;name&quot;:&quot;Role
              brief&quot;{"}"}&apos;
            </span>
            , then upload a document to{" "}
            <span className="font-mono text-[11.5px] text-ink">
              /knowledge/&lt;id&gt;/documents
            </span>
            .
          </Empty>
        ) : (
          <Table head={["Name", num("Documents"), num("Chunks"), num("Characters"), "Updated"]}>
            {bases.map((base) => (
              <Row key={base.id}>
                <Cell>
                  <span className="block font-medium text-ink">{base.name}</span>
                  {base.description ? (
                    <span className="mt-0.5 block text-[12px] text-ink-mid">
                      {base.description}
                    </span>
                  ) : null}
                  <span className="mt-0.5 block font-mono text-[11px] text-ink-low">
                    {base.id}
                  </span>
                </Cell>
                <Cell right mono dim>
                  {base.documents?.length ?? 0}
                </Cell>
                <Cell right mono dim>
                  {base.chunk_count ?? 0}
                </Cell>
                <Cell right mono dim>
                  {base.total_chars ?? 0}
                </Cell>
                <Cell mono dim>
                  {base.updated_at ?? "—"}
                </Cell>
              </Row>
            ))}
          </Table>
        )}
      </Card>

      {selected ? (
        <>
          <Card>
            <div className="grid grid-cols-3 divide-x divide-hair">
              <Metric label="Documents" value={selected.documents?.length ?? 0} />
              <Metric label="Chunks" value={selected.chunk_count ?? 0} />
              <Metric
                label="Retrievable characters"
                value={selected.total_chars ?? 0}
                target="counted over chunks, not raw uploads"
              />
            </div>
          </Card>

          <Card>
            <CardHeader
              title="Retrieval tester"
              hint="What this base would return for a query, with scores. The only thing that distinguishes bad retrieval from a bad answer."
            />

            <form
              className="grid gap-x-4 px-5 py-5 sm:grid-cols-[minmax(0,1fr)_minmax(0,2fr)_auto_auto] sm:grid-rows-[auto_auto_auto]"
              onSubmit={(event) => {
                event.preventDefault();
                void runQuery();
              }}
            >
              <Field row label="Knowledge base">
                <Select
                  value={selectedId}
                  onChange={(event) => {
                    setSelectedId(event.target.value);
                    /* Results belong to the base they were run against. Keeping them on
                       screen after a switch would attribute one base's chunks to another. */
                    setHits(null);
                    setQueryError(null);
                  }}
                >
                  {bases.map((base) => (
                    <option key={base.id} value={base.id}>
                      {base.name}
                    </option>
                  ))}
                </Select>
              </Field>

              <Field row label="Query" hint="Scored by term overlap, so wording matters.">
                <Input
                  value={query}
                  placeholder="e.g. what should I ask about queues?"
                  onChange={(event) => setQuery(event.target.value)}
                />
              </Field>

              <Field row label="Top k">
                <Select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
                  {TOP_K_CHOICES.map((choice) => (
                    <option key={choice} value={choice}>
                      {choice}
                    </option>
                  ))}
                </Select>
              </Field>

              {/* Row 2 explicitly: the button belongs beside the controls, not aligned
                  against the full height of the fields next to it. */}
              <div className="row-start-2 self-start">
                <Button type="submit" variant="primary" disabled={running || !query.trim()}>
                  {running ? "Retrieving…" : "Retrieve"}
                </Button>
              </div>
            </form>

            {queryError ? (
              <div className="border-t border-hair px-5 py-4">
                <Chip status="bad">query failed</Chip>
                <p className="mt-2 text-[12.5px] text-ink-mid">{queryError}</p>
              </div>
            ) : corpusEmpty ? (
              <div className="border-t border-hair">
                <Empty title="Nothing to retrieve from">
                  This base has no chunks yet, so every query returns empty — which looks
                  identical to broken retrieval. Upload a document to{" "}
                  <span className="font-mono text-[11.5px] text-ink">
                    /knowledge/{selected.id}/documents
                  </span>{" "}
                  with a{" "}
                  <span className="font-mono text-[11.5px] text-ink">filename</span> and{" "}
                  <span className="font-mono text-[11.5px] text-ink">text</span>; blank lines
                  become chunk boundaries.
                </Empty>
              </div>
            ) : hits === null ? (
              <p className="border-t border-hair px-5 py-8 text-center text-[12.5px] text-ink-mid">
                Run a query to see which chunks it would pull.
              </p>
            ) : hits.length === 0 ? (
              <div className="border-t border-hair">
                <Empty title="No chunk shared a term with that query">
                  This is a real miss, not an error: keyword retrieval needs a word in common,
                  so a paraphrase with no shared vocabulary finds nothing. Try the wording the
                  document itself uses — and if that is the shape of the questions being
                  asked, this base needs a semantic index rather than a better query.
                </Empty>
              </div>
            ) : (
              <div className="border-t border-hair">
                <Table head={[num("Rank"), num("Score"), "Document", "Chunk"]}>
                  {hits.map((hit, index) => (
                    <Row key={`${hit.document_id}-${index}`}>
                      <Cell mono dim>
                        {index + 1}
                      </Cell>
                      <Cell mono>{hit.score.toFixed(4)}</Cell>
                      <Cell mono dim>
                        {filenames.get(hit.document_id) ?? hit.document_id}
                      </Cell>
                      <Cell>
                        <span className="block max-w-2xl whitespace-pre-wrap text-ink-mid">
                          {hit.text}
                        </span>
                      </Cell>
                    </Row>
                  ))}
                </Table>
              </div>
            )}
          </Card>
        </>
      ) : null}
    </Page>
  );
}
