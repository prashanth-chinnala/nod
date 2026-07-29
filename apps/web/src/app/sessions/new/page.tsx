"use client";

/**
 * Configure an interview, create it, hand the candidate a link.
 *
 * The shape matters and it is not cosmetic. An interview is opened *for* a candidate by
 * whoever is hiring: they pick the agent, confirm what it will use, and send a URL. The
 * candidate chooses nothing — they click a link and talk. Letting them pick an agent would be
 * letting them pick their own interviewer.
 *
 * So this page is the operator's, and `/interview/[id]` is the candidate's. The session id in
 * that URL is what carries the configuration across, which means the runtime can stop reading
 * `AVATAR_AGENT` from its environment once it honours the id — the env var was always a
 * stand-in for this.
 *
 * **What the link does and does not do.** It identifies a session record. It is not a
 * credential: anyone holding the URL can open the interview, and it does not expire. That is
 * acceptable for a demo and is not acceptable for a product — a real deployment needs a signed,
 * single-use, expiring token. Written here rather than discovered later.
 */

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";

import {
  Button,
  Card,
  CardHeader,
  Cell,
  Chip,
  Empty,
  Field,
  Page,
  Row,
  Select,
  Table,
} from "@/components/ui";

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

type Agent = {
  id: string;
  name: string;
  llm_provider: string;
  llm_model: string;
  voice_provider: string;
  voice_id: string;
  face_id: string | null;
  knowledge_base_ids: string[];
  tool_ids: string[];
  guardrail_id: string | null;
  pronunciation_id: string | null;
  turn_taking: Record<string, number>;
};

type Named = { id: string; name: string };

export default function NewSessionPage() {
  const [agents, setAgents] = useState<Agent[] | null>(null);
  const [knowledge, setKnowledge] = useState<Named[]>([]);
  const [lexicons, setLexicons] = useState<Named[]>([]);
  const [faces, setFaces] = useState<Named[]>([]);
  const [chosen, setChosen] = useState("");
  const [created, setCreated] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const load = useCallback(async () => {
    try {
      const [a, k, l, f] = await Promise.all(
        ["agents", "knowledge", "pronunciations", "faces"].map((c) =>
          fetch(`${API}/${c}`, { cache: "no-store" }).then((r) => r.json()),
        ),
      );
      setAgents(a as Agent[]);
      setKnowledge(k as Named[]);
      setLexicons(l as Named[]);
      setFaces(f as Named[]);
      setChosen((prev) => prev || ((a as Agent[])[0]?.id ?? ""));
      setError(null);
    } catch {
      setError(
        `Cannot reach the runtime at ${API}. Start it with: cd apps/api && python -m uvicorn avatar.server:app`,
      );
    }
  }, []);

  useEffect(() => {
    let live = true;
    void Promise.resolve().then(() => {
      if (live) void load();
    });
    return () => {
      live = false;
    };
  }, [load]);

  const agent = agents?.find((a) => a.id === chosen) ?? null;
  const nameOf = (list: Named[], id: string | null) =>
    (id && list.find((item) => item.id === id)?.name) || null;

  async function create() {
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(`${API}/sessions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ agent_id: chosen || null }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const record = (await response.json()) as { id: string };
      setCreated(record.id);
    } catch {
      setError("Could not create the session. Is the runtime still up?");
    } finally {
      setBusy(false);
    }
  }

  const link = created ? `${window.location.origin}/interview/${created}` : "";

  return (
    <Page
      title="New session"
      lede="Pick the interviewer, confirm what it will use, then send the candidate a link. They choose nothing — they open the URL and talk."
      action={<Link href="/sessions"><Button>All sessions</Button></Link>}
    >
      {error ? (
        <Card className="border-bad/40 bg-bad/5">
          <div className="px-5 py-4">
            <p className="text-[13px] font-medium text-bad">Runtime unreachable</p>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-mid">{error}</p>
          </div>
        </Card>
      ) : null}

      {/* ------------------------------------------------------- the handover */}
      {created ? (
        <Card className="border-ok/40 bg-ok/5">
          <CardHeader
            title="Session created"
            hint="Send this to the candidate. Open it yourself in another tab to watch the interview from their side."
            action={<Chip status="ok">ready</Chip>}
          />
          <div className="px-5 py-4">
            <div className="flex flex-wrap items-center gap-2">
              <code className="min-w-0 flex-1 overflow-x-auto rounded-lg border border-hair bg-base px-3 py-2 font-mono text-[12px] text-ink">
                {link}
              </code>
              <Button
                variant="primary"
                onClick={() => {
                  void navigator.clipboard.writeText(link);
                  setCopied(true);
                }}
              >
                {copied ? "Copied" : "Copy link"}
              </Button>
              <Link href={`/interview/${created}`}>
                <Button>Open it</Button>
              </Link>
            </div>
            {/* Stated on the screen that produces the link, not buried in docs. A URL that
                looks like an invitation gets treated as one. */}
            <p className="mt-3 max-w-2xl text-[11.5px] leading-relaxed text-ink-mid">
              This link identifies a session record. It is <strong>not a credential</strong> —
              anyone holding it can open the interview and it does not expire. Fine for a demo;
              a real deployment needs a signed, single-use, expiring token.
            </p>
            <div className="mt-4">
              <Button onClick={() => { setCreated(null); setCopied(false); }}>
                Create another
              </Button>
            </div>
          </div>
        </Card>
      ) : null}

      {/* ---------------------------------------------------------- selection */}
      <Card>
        <CardHeader title="Interviewer" hint="Everything else follows from this choice." />
        {agents === null && !error ? (
          <p className="px-5 py-10 text-center text-[12.5px] text-ink-low">Loading…</p>
        ) : agents && agents.length === 0 ? (
          <Empty
            title="No agents configured"
            action={<Link href="/agents"><Button variant="primary">Create an agent</Button></Link>}
          >
            An agent is the interviewer: its prompt, model, voice, and which knowledge base,
            lexicon and guardrail it uses. A session needs one before it can be anything more
            than a default.
          </Empty>
        ) : (
          <div className="space-y-4 px-5 py-4">
            <Field
              label="Agent"
              hint="Only agents already created in the console. Add one there if the right interviewer is missing."
            >
              <Select value={chosen} onChange={(event) => setChosen(event.target.value)}>
                {(agents ?? []).map((option) => (
                  <option key={option.id} value={option.id}>
                    {option.name}
                  </option>
                ))}
              </Select>
            </Field>

            {agent ? (
              <div>
                <p className="mb-2 text-[11px] font-medium tracking-[0.06em] uppercase text-ink-low">
                  What this session will use
                </p>
                {/* Resolved names, not ids. An operator about to send a candidate a link needs
                    to recognise the knowledge base, and `kb_47912396` is unrecognisable. */}
                <Table head={["Component", "Resolved to", ""]}>
                  <Row>
                    <Cell dim>Model</Cell>
                    <Cell mono>{agent.llm_model || agent.llm_provider || "adapter default"}</Cell>
                    <Cell right><Chip status="ok">set</Chip></Cell>
                  </Row>
                  <Row>
                    <Cell dim>Voice</Cell>
                    <Cell mono>{agent.voice_id || agent.voice_provider || "adapter default"}</Cell>
                    <Cell right><Chip status="ok">set</Chip></Cell>
                  </Row>
                  <Row>
                    <Cell dim>Face</Cell>
                    <Cell mono>{nameOf(faces, agent.face_id) ?? "placeholder renderer"}</Cell>
                    <Cell right>
                      <Chip status={agent.face_id ? "ok" : "warn"}>
                        {agent.face_id ? "set" : "no model"}
                      </Chip>
                    </Cell>
                  </Row>
                  <Row>
                    <Cell dim>Knowledge</Cell>
                    <Cell mono>
                      {agent.knowledge_base_ids.length
                        ? agent.knowledge_base_ids
                            .map((id) => nameOf(knowledge, id) ?? id)
                            .join(", ")
                        : "none — questions will be generic"}
                    </Cell>
                    <Cell right>
                      <Chip status={agent.knowledge_base_ids.length ? "ok" : "neutral"}>
                        {agent.knowledge_base_ids.length || "none"}
                      </Chip>
                    </Cell>
                  </Row>
                  <Row>
                    <Cell dim>Pronunciations</Cell>
                    <Cell mono>{nameOf(lexicons, agent.pronunciation_id) ?? "none"}</Cell>
                    <Cell right>
                      <Chip status={agent.pronunciation_id ? "ok" : "neutral"}>
                        {agent.pronunciation_id ? "set" : "none"}
                      </Chip>
                    </Cell>
                  </Row>
                  <Row>
                    <Cell dim>End-of-turn wait</Cell>
                    <Cell mono>{agent.turn_taking?.end_of_turn_silence_ms ?? 700} ms</Cell>
                    <Cell right>
                      {/* Surfaced here because it is the largest single term in the measured
                          latency budget, and it is a conversational judgment rather than a
                          performance setting — worth seeing before sending the link. */}
                      <Chip status="info">largest latency term</Chip>
                    </Cell>
                  </Row>
                  <Row>
                    <Cell dim>Guardrails · Tools</Cell>
                    <Cell mono>
                      {agent.guardrail_id || agent.tool_ids.length
                        ? "configured, not yet enforced mid-turn"
                        : "none"}
                    </Cell>
                    <Cell right><Chip status="warn">not wired</Chip></Cell>
                  </Row>
                </Table>
              </div>
            ) : null}

            <div className="flex items-center gap-3 pt-1">
              <Button variant="primary" disabled={busy || !chosen} onClick={() => void create()}>
                {busy ? "Creating…" : "Create session & get link"}
              </Button>
              <span className="text-[11.5px] text-ink-low">
                Creates a record now; the conversation starts when the candidate opens the link.
              </span>
            </div>
          </div>
        )}
      </Card>
    </Page>
  );
}
