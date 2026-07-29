"use client";

/**
 * Past conversations, and the per-turn latency behind each one.
 *
 * The column that earns its place is **worst end-to-end**, not the mean. A mean hides the one
 * interview in fifty with a six-second first frame, and that one is a candidate having a
 * visibly broken experience. The list is scanned to find those, so the worst case is what it
 * shows.
 *
 * Records here are append-only server-side — no PATCH, no DELETE — so this page never offers
 * an edit. A conversation record that could be revised after the fact would make every latency
 * figure derived from it uncitable.
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";

import { Button, Card, CardHeader, Cell, Chip, Empty, Metric, Page, Row, Table } from "@/components/ui";

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

type Turn = {
  epoch: number;
  heard: string;
  said: string;
  transcribed: boolean;
  llm_ttft_ms: number | null;
  tts_first_audio_ms: number | null;
  first_frame_ms: number | null;
  perceived_total_ms: number | null;
  interrupted: boolean;
};

type Session = {
  id: string;
  agent_id: string | null;
  started_at: string;
  ended_at: string | null;
  turns: Turn[];
  stale_dropped: number;
  frames_repeated: number;
};

function worstTotal(session: Session): number | null {
  const totals = session.turns.map((t) => t.perceived_total_ms).filter((v): v is number => v !== null);
  return totals.length ? Math.max(...totals) : null;
}

function when(iso: string): string {
  // Locale-formatted at render time only. Doing it during SSR would produce markup that
  // disagrees with the client's timezone and trip a hydration mismatch.
  return new Date(iso).toLocaleString();
}

export default function SessionsPage() {
  const [sessions, setSessions] = useState<Session[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const response = await fetch(`${API}/sessions`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setSessions((await response.json()) as Session[]);
      setError(null);
    } catch {
      setError(
        `Cannot reach the runtime at ${API}. Start it with: cd apps/api && python -m uvicorn avatar.server:app`,
      );
    }
  }, []);

  useEffect(() => {
    // Deferred by a microtask so the state updates land outside the effect body. The React
    // docs' own guidance for this shape -- subscribing to an external system and calling
    // setState from a callback -- and it keeps the compiler-aware lint rule satisfied without
    // suppressing it.
    let live = true;
    void Promise.resolve().then(() => {
      if (live) void load();
    });
    return () => {
      live = false;
    };
  }, [load]);

  const open = sessions?.find((s) => s.id === openId) ?? null;
  const interrupted = (s: Session) => s.turns.filter((t) => t.interrupted).length;

  return (
    <Page
      title="Sessions"
      lede="Every conversation, with the per-turn latency breakdown behind it. Worst case rather than average, because the worst case is the candidate who had a broken interview."
      action={
        <div className="flex gap-2">
          <Button onClick={() => void load()}>Refresh</Button>
          <Link href="/sessions/new"><Button variant="primary">New session</Button></Link>
        </div>
      }
    >
      {error ? (
        <Card className="border-bad/40 bg-bad/5">
          <div className="px-5 py-4">
            <p className="text-[13px] font-medium text-bad">Runtime unreachable</p>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-mid">{error}</p>
          </div>
        </Card>
      ) : null}

      <Card>
        <CardHeader title="Recorded conversations" hint={sessions ? `${sessions.length} total` : "loading…"} />
        {sessions === null && !error ? (
          <p className="px-5 py-10 text-center text-[12.5px] text-ink-low">Loading…</p>
        ) : sessions && sessions.length === 0 ? (
          <Empty
            title="No sessions recorded yet"
            action={<Link href="/sessions/new"><Button variant="primary">Start a session</Button></Link>}
          >
            A session record holds the transcript and the per-turn latency breakdown, which is
            where the claim that none of the dominant terms is the renderer actually comes from.
            Run one and it will appear here.
          </Empty>
        ) : (
          <Table head={["Started", "Agent", "Turns", "Barge-ins", "Worst end-to-end", "State", ""]}>
            {(sessions ?? []).map((session) => {
              const worst = worstTotal(session);
              return (
                <Row key={session.id}>
                  <Cell mono dim>{when(session.started_at)}</Cell>
                  <Cell mono dim>{session.agent_id ?? "—"}</Cell>
                  <Cell>{session.turns.length}</Cell>
                  <Cell>{interrupted(session) || "—"}</Cell>
                  <Cell mono>
                    {worst === null ? (
                      "—"
                    ) : (
                      <span className={worst > 1000 ? "text-bad" : "text-ok"}>
                        {Math.round(worst)} ms
                      </span>
                    )}
                  </Cell>
                  <Cell>
                    <Chip status={session.ended_at ? "neutral" : "info"}>
                      {session.ended_at ? "ended" : "open"}
                    </Chip>
                  </Cell>
                  <Cell right>
                    <Button onClick={() => setOpenId(session.id === openId ? null : session.id)}>
                      {session.id === openId ? "Hide" : "Inspect"}
                    </Button>
                  </Cell>
                </Row>
              );
            })}
          </Table>
        )}
      </Card>

      {open ? (
        <Card>
          <CardHeader
            title={`Turns — ${open.id}`}
            hint="Each turn's stage breakdown. A dash means the turn was cut off before that stage completed, which is a real record rather than a missing one."
          />
          <div className="grid grid-cols-2 divide-hair border-b border-hair sm:grid-cols-4">
            <Metric label="Turns" value={open.turns.length} />
            <Metric label="Barge-ins" value={interrupted(open)} status={interrupted(open) ? "info" : "neutral"} />
            <Metric
              label="Stale artifacts dropped"
              value={open.stale_dropped}
              target="proof barge-in worked by invalidation"
            />
            <Metric
              label="Worst end-to-end"
              value={worstTotal(open) === null ? "—" : Math.round(worstTotal(open)!)}
              unit={worstTotal(open) === null ? undefined : "ms"}
              target="target < 1000ms"
              status={(worstTotal(open) ?? 0) > 1000 ? "bad" : "ok"}
            />
          </div>
          <Table head={["#", "Heard", "Said", "LLM", "Voice", "Frame", "End-to-end"]}>
            {open.turns.map((turn, index) => (
              <Row key={`${turn.epoch}-${index}`}>
                <Cell mono dim>{turn.epoch}</Cell>
                <Cell>
                  <span className={turn.transcribed ? "" : "italic text-warn"}>
                    {turn.heard || "—"}
                  </span>
                </Cell>
                <Cell dim>
                  {turn.said || "—"}
                  {turn.interrupted ? <span className="text-warn"> (interrupted)</span> : null}
                </Cell>
                <Cell mono right>{turn.llm_ttft_ms === null ? "—" : Math.round(turn.llm_ttft_ms)}</Cell>
                <Cell mono right>{turn.tts_first_audio_ms === null ? "—" : Math.round(turn.tts_first_audio_ms)}</Cell>
                <Cell mono right>{turn.first_frame_ms === null ? "—" : Math.round(turn.first_frame_ms)}</Cell>
                <Cell mono right>
                  {turn.perceived_total_ms === null ? (
                    "—"
                  ) : (
                    <span className={turn.perceived_total_ms > 1000 ? "text-bad" : "text-ok"}>
                      {Math.round(turn.perceived_total_ms)}
                    </span>
                  )}
                </Cell>
              </Row>
            ))}
          </Table>
        </Card>
      ) : null}
    </Page>
  );
}
