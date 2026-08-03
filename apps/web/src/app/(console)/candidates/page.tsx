"use client";

/**
 * Candidates — the people being interviewed, their resumes, and the invite that starts it.
 *
 * **Why the resume is the centre of this screen.** Everything else about a candidate is
 * bookkeeping: a name, an email, a role. The resume is the only thing here that changes the
 * interview — the runtime appends it to the interviewer's prompt, so the same agent probes a
 * ledger engineer about ordering guarantees and a data engineer about late events without anyone
 * configuring a thing. So the drop zone is prominent, and the extracted text is shown rather than
 * hidden: an operator has to be able to see what the interviewer will actually read.
 *
 * **Why extraction failures are shown as loudly as successes.** A scanned PDF stores fine and
 * extracts to nothing. If that were quiet, the first sign would be an interviewer asking generic
 * questions, three weeks later, with nobody able to say why. So a resume that did not parse is a
 * warning on the row with the reason and the fix.
 *
 * **Why the invite is a copy button and not a mailto.** This process does not know how a company
 * sends candidate links — ATS, email, a calendar invite — and guessing would produce a workflow
 * nobody uses. It mints the session, shows the link, and gets out of the way.
 */

import { useCallback, useEffect, useRef, useState } from "react";

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

const API = "http://127.0.0.1:8000";

/** Mirrors `avatar.resume.SUPPORTED`, so the guidance cannot drift from what is enforced. */
const ACCEPT = ".pdf,.docx,.txt,.md,.markdown";
const MAX_MB = 10;

type Candidate = {
  id: string;
  name: string;
  email: string;
  role: string;
  status: string;
  notes: string | null;
  agent_id: string | null;
  resume_filename: string | null;
  resume_text: string | null;
  resume_chars: number | null;
  resume_pages: number | null;
  resume_truncated: boolean;
  resume_error: string | null;
  updated_at: string;
};

type Agent = { id: string; name: string };

type Session = {
  id: string;
  agent_name?: string | null;
  started_at: string | null;
  ended_at: string | null;
  turns?: unknown[];
  scoring?: { status?: string; weighted_score?: number | null } | null;
};

const HEAD = ["Name", "Role", "Resume", "Interviewer", "Status", "Updated (UTC)", ""] as const;

function stamp(iso: string | null): string {
  return iso ? iso.replace("T", " ").replace(/(\+00:00|Z)$/, "").slice(0, 16) : "—";
}

/**
 * The console's status vocabulary, mapped from the API's.
 *
 * `new` is deliberately not "ok": a candidate with no interview yet is neither good nor bad news,
 * and colouring it green would make a to-do list look finished.
 */
function tone(status: string): "ok" | "warn" | "neutral" {
  if (status === "interviewed" || status === "reviewed") return "ok";
  if (status === "invited") return "warn";
  return "neutral";
}

export default function CandidatesPage() {
  const [candidates, setCandidates] = useState<Candidate[] | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Record<string, Session[]>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [invite, setInvite] = useState<{ id: string; path: string } | null>(null);
  const [copied, setCopied] = useState(false);

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [role, setRole] = useState("");
  const [notes, setNotes] = useState("");
  const [agentId, setAgentId] = useState("");

  const load = useCallback(() => {
    Promise.all([
      fetch(`${API}/candidates`, { cache: "no-store" }),
      fetch(`${API}/agents`, { cache: "no-store" }),
    ])
      .then(async ([c, a]) => {
        if (!c.ok) throw new Error(`the runtime answered ${c.status}`);
        setCandidates((await c.json()) as Candidate[]);
        setAgents(a.ok ? ((await a.json()) as Agent[]) : []);
        setError(null);
      })
      .catch((cause: unknown) => {
        setCandidates(null);
        setError(
          cause instanceof Error
            ? cause.message
            : "cannot reach the runtime at http://127.0.0.1:8000",
        );
      });
  }, []);

  useEffect(load, [load]);

  const loadSessions = useCallback((id: string) => {
    fetch(`${API}/candidates/${id}/sessions`, { cache: "no-store" })
      .then(async (r) => (r.ok ? ((await r.json()) as Session[]) : []))
      .then((listed) => setSessions((prior) => ({ ...prior, [id]: listed })))
      .catch(() => setSessions((prior) => ({ ...prior, [id]: [] })));
  }, []);

  async function create() {
    if (!name.trim()) return;
    setBusy("create");
    try {
      const response = await fetch(`${API}/candidates`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          name: name.trim(),
          email: email.trim(),
          role: role.trim(),
          notes: notes.trim(),
          agent_id: agentId || null,
        }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { detail?: unknown } | null;
        throw new Error(
          typeof body?.detail === "string" ? body.detail : `the runtime answered ${response.status}`,
        );
      }
      const made = (await response.json()) as Candidate;
      setName("");
      setEmail("");
      setRole("");
      setNotes("");
      load();
      setOpen(made.id);
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "could not create the candidate");
    } finally {
      setBusy(null);
    }
  }

  async function upload(id: string, file: File) {
    if (file.size > MAX_MB * 1024 * 1024) {
      setError(
        `${file.name} is ${(file.size / 1_048_576).toFixed(1)} MB, over the ${MAX_MB} MB ceiling.`,
      );
      return;
    }
    setBusy(`resume:${id}`);
    setError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API}/candidates/${id}/resume`, {
        method: "POST",
        body: form,
      });
      const body = (await response.json().catch(() => null)) as
        | (Candidate & { detail?: unknown })
        | null;
      if (!response.ok) {
        throw new Error(
          typeof body?.detail === "string" ? body.detail : `the runtime answered ${response.status}`,
        );
      }
      load();
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "the resume could not be uploaded");
    } finally {
      setBusy(null);
    }
  }

  async function patch(id: string, body: Record<string, unknown>) {
    setBusy(`patch:${id}`);
    try {
      const response = await fetch(`${API}/candidates/${id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`the runtime answered ${response.status}`);
      load();
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "the change did not save");
    } finally {
      setBusy(null);
    }
  }

  async function mint(candidate: Candidate) {
    setBusy(`invite:${candidate.id}`);
    setCopied(false);
    try {
      const response = await fetch(`${API}/candidates/${candidate.id}/interview`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({}),
      });
      const body = (await response.json().catch(() => null)) as
        | { session_id?: string; interview_path?: string; detail?: unknown }
        | null;
      if (!response.ok || !body?.session_id) {
        throw new Error(
          typeof body?.detail === "string"
            ? body.detail
            : "no interviewer is set for this candidate",
        );
      }
      setInvite({ id: body.session_id, path: body.interview_path ?? "" });
      load();
      loadSessions(candidate.id);
    } catch (cause: unknown) {
      setError(cause instanceof Error ? cause.message : "the interview could not be created");
    } finally {
      setBusy(null);
    }
  }

  async function remove(id: string) {
    setBusy(`delete:${id}`);
    try {
      await fetch(`${API}/candidates/${id}`, { method: "DELETE" });
      if (open === id) setOpen(null);
      load();
    } finally {
      setBusy(null);
    }
  }

  const current = candidates?.find((c) => c.id === open) ?? null;

  return (
    <Page
      title="Candidates"
      lede="The people being interviewed. A resume attached here reaches the interviewer's prompt, so the same agent asks a ledger engineer and a data engineer different questions."
    >
      <Card>
        <CardHeader
          title="Add a candidate"
          hint="Name is the only requirement. The interviewer can be set now or later — an invite needs one."
        />
        <div className="grid gap-4 p-5 md:grid-cols-2">
          <Field label="Name" hint="Shown to nobody but you; the interviewer is told it once.">
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Aparna Rao"
            />
          </Field>
          <Field label="Email" hint="Where you would send the invite. Not sent from here.">
            <Input
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="aparna@example.com"
            />
          </Field>
          <Field label="Role" hint="Given to the interviewer as what they are interviewing for.">
            <Input
              value={role}
              onChange={(e) => setRole(e.target.value)}
              placeholder="Senior Backend Engineer"
            />
          </Field>
          <Field
            label="Interviewer"
            hint={
              agents.length
                ? "Carries the rubric, voice, knowledge and guardrail."
                : "No agents configured yet — create one first."
            }
          >
            <Select value={agentId} onChange={(e) => setAgentId(e.target.value)}>
              <option value="">— none yet —</option>
              {agents.map((agent) => (
                <option key={agent.id} value={agent.id}>
                  {agent.name}
                </option>
              ))}
            </Select>
          </Field>
          <div className="md:col-span-2">
            <Field
              label="Note for the interviewer"
              hint="Passed into the prompt verbatim. “Probe the ledger work” changes the interview; “nice CV” does not."
            >
              <Textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={2}
                placeholder="Referred by the payments team. Probe the ledger work."
              />
            </Field>
          </div>
          <div className="md:col-span-2">
            <Button onClick={create} disabled={!name.trim() || busy === "create"}>
              {busy === "create" ? "Adding…" : "Add candidate"}
            </Button>
          </div>
        </div>
      </Card>

      <Card>
        <CardHeader
          title={`Candidates${candidates ? ` · ${candidates.length}` : ""}`}
          hint="Click a name, or “add resume”, to open that candidate below — that is where the resume drop zone, the interviewer and the invite link live."
        />
        {error ? (
          <Empty
            title="Could not reach the runtime"
            action={<Button onClick={load}>Retry</Button>}
          >
            {error}. The console reads candidates from the runtime on 127.0.0.1:8000.
          </Empty>
        ) : candidates === null ? (
          <div className="px-5 py-14 text-center text-[12.5px] text-ink-mid">
            Loading candidates…
          </div>
        ) : candidates.length === 0 ? (
          <Empty title="No candidates yet">
            Add one above, attach their resume, and the interviewer will be briefed from it — the
            resume is the only thing on this screen that changes what gets asked.
          </Empty>
        ) : (
          <Table head={[...HEAD]}>
            {candidates.map((candidate) => {
              const agent = agents.find((a) => a.id === candidate.agent_id);
              return (
                <Row key={candidate.id}>
                  <Cell>
                    <button
                      onClick={() => {
                        const next = open === candidate.id ? null : candidate.id;
                        setOpen(next);
                        setInvite(null);
                        if (next) loadSessions(next);
                      }}
                      className="text-left font-medium text-ink underline decoration-hair underline-offset-2 transition-colors hover:text-accent"
                    >
                      {candidate.name || "—"}
                    </button>
                    {/* Stated rather than left to a hover. The resume panel used to be reachable
                        only by noticing that a name was clickable, which is an affordance nobody
                        looks for in a table -- the first question asked of this screen was "where
                        is the resume upload". */}
                    <span className="block text-[10.5px] text-ink-low">
                      {open === candidate.id ? "open below" : "open to add a resume"}
                    </span>
                    {candidate.email ? (
                      <span className="block text-[11px] text-ink-low">{candidate.email}</span>
                    ) : null}
                  </Cell>
                  <Cell dim>{candidate.role || "—"}</Cell>
                  <Cell>
                    {/* The cell is the control. A column that reports "none" and cannot be acted
                        on sends the operator hunting for the place that can. */}
                    <button
                      onClick={() => {
                        setOpen(candidate.id);
                        setInvite(null);
                        loadSessions(candidate.id);
                      }}
                      className="text-left transition-colors hover:text-accent"
                    >
                      {candidate.resume_error ? (
                        <span className="text-warn">did not parse — replace</span>
                      ) : candidate.resume_filename ? (
                        <span className="text-ink-mid">
                          {candidate.resume_chars?.toLocaleString() ?? "—"} chars
                          {candidate.resume_truncated ? (
                            <span className="text-warn"> (truncated)</span>
                          ) : null}
                        </span>
                      ) : (
                        <span className="text-accent underline decoration-accent/40 underline-offset-2">
                          + add resume
                        </span>
                      )}
                    </button>
                  </Cell>
                  <Cell dim>{agent?.name ?? (candidate.agent_id ? "(deleted)" : "—")}</Cell>
                  <Cell>
                    <Chip status={tone(candidate.status)}>{candidate.status}</Chip>
                  </Cell>
                  <Cell mono dim>
                    {stamp(candidate.updated_at)}
                  </Cell>
                  <Cell>
                    <Button
                      variant="danger"
                      onClick={() => remove(candidate.id)}
                      disabled={busy === `delete:${candidate.id}`}
                    >
                      Delete
                    </Button>
                  </Cell>
                </Row>
              );
            })}
          </Table>
        )}
      </Card>

      {current ? (
        <CandidateDetail
          candidate={current}
          agents={agents}
          sessions={sessions[current.id] ?? null}
          busy={busy}
          invite={invite?.id === undefined ? null : invite}
          copied={copied}
          onUpload={(file) => upload(current.id, file)}
          onPatch={(body) => patch(current.id, body)}
          onMint={() => mint(current)}
          onCopy={(text) => {
            void navigator.clipboard?.writeText(text);
            setCopied(true);
          }}
        />
      ) : null}
    </Page>
  );
}

function CandidateDetail({
  candidate,
  agents,
  sessions,
  busy,
  invite,
  copied,
  onUpload,
  onPatch,
  onMint,
  onCopy,
}: {
  candidate: Candidate;
  agents: Agent[];
  sessions: Session[] | null;
  busy: string | null;
  invite: { id: string; path: string } | null;
  copied: boolean;
  onUpload: (file: File) => void;
  onPatch: (body: Record<string, unknown>) => void;
  onMint: () => void;
  onCopy: (text: string) => void;
}) {
  const picker = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [showText, setShowText] = useState(false);

  const link = invite ? `${window.location.origin}${invite.path}` : "";

  return (
    <Card>
      <CardHeader
        title={candidate.name}
        hint={`${candidate.role || "no role set"} · ${candidate.status}`}
      />
      <div className="grid gap-5 p-5 lg:grid-cols-2">
        <section>
          <p className="mb-2 text-[11px] font-medium tracking-[0.07em] uppercase text-ink-low">
            Resume
          </p>

          {/* A drop zone rather than only a file input: a resume arrives as a file already on
              screen, and making someone traverse a picker for it is friction with no purpose. The
              hidden input stays because a drop zone alone is unreachable by keyboard. */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const file = e.dataTransfer.files?.[0];
              if (file) onUpload(file);
            }}
            className={[
              "rounded-lg border border-dashed p-5 text-center transition-colors",
              dragging ? "border-accent bg-accent/8" : "border-hair bg-glass",
            ].join(" ")}
          >
            <p className="text-[13px] text-ink-mid">
              Drop a resume here, or{" "}
              <button
                onClick={() => picker.current?.click()}
                className="text-accent underline decoration-accent/40 underline-offset-2"
              >
                choose a file
              </button>
            </p>
            <p className="mt-1 text-[11px] text-ink-low">
              PDF, DOCX, TXT or MD · up to {MAX_MB} MB · text is extracted now, not at interview time
            </p>
            <input
              ref={picker}
              type="file"
              accept={ACCEPT}
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) onUpload(file);
                e.target.value = "";
              }}
            />
          </div>

          {busy === `resume:${candidate.id}` ? (
            <p className="mt-3 text-[12px] text-ink-mid">Extracting…</p>
          ) : candidate.resume_error ? (
            <div className="mt-3 rounded-lg border border-warn/40 bg-warn/8 p-3">
              <p className="text-[12px] font-medium text-warn">
                {candidate.resume_filename} stored, but no text could be read
              </p>
              <p className="mt-1 text-[11.5px] leading-relaxed text-ink-mid">
                {candidate.resume_error}
              </p>
              <p className="mt-1.5 text-[11px] text-ink-low">
                The interview will run without a briefing until this is replaced.
              </p>
            </div>
          ) : candidate.resume_filename ? (
            <div className="mt-3 rounded-lg border border-hair bg-raise p-3">
              <p className="text-[12.5px] font-medium text-ink">{candidate.resume_filename}</p>
              <p className="mt-0.5 text-[11px] text-ink-low">
                {candidate.resume_chars?.toLocaleString()} characters
                {candidate.resume_pages ? ` · ${candidate.resume_pages} pages` : ""}
                {candidate.resume_truncated
                  ? " · truncated to the briefing limit"
                  : " · sent in full"}
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                <Button variant="ghost" onClick={() => setShowText(!showText)}>
                  {showText ? "Hide extracted text" : "Show what the interviewer reads"}
                </Button>
                <a
                  href={`${API}/candidates/${candidate.id}/resume/file`}
                  className="text-[12px] text-accent underline decoration-accent/40 underline-offset-2"
                >
                  Download original
                </a>
              </div>
              {showText ? (
                <pre className="mt-2 max-h-64 overflow-auto rounded border border-hair bg-glass p-2.5 text-[11px] leading-relaxed whitespace-pre-wrap text-ink-mid">
                  {candidate.resume_text}
                </pre>
              ) : null}
            </div>
          ) : (
            <p className="mt-3 text-[12px] text-ink-low">
              No resume attached. The interview still runs — the interviewer just has nothing
              specific to probe.
            </p>
          )}
        </section>

        <section className="space-y-4">
          <Field
            label="Interviewer"
            hint="Changing this changes the rubric, voice, knowledge and guardrail for the next invite."
          >
            <Select
              value={candidate.agent_id ?? ""}
              onChange={(e) => onPatch({ agent_id: e.target.value || null })}
            >
              <option value="">— none —</option>
              {agents.map((agent) => (
                <option key={agent.id} value={agent.id}>
                  {agent.name}
                </option>
              ))}
            </Select>
          </Field>

          <Field label="Note for the interviewer" hint="Passed into the prompt verbatim.">
            <Textarea
              defaultValue={candidate.notes ?? ""}
              rows={2}
              onBlur={(e) => {
                if (e.target.value !== (candidate.notes ?? "")) {
                  onPatch({ notes: e.target.value });
                }
              }}
            />
          </Field>

          <div className="rounded-lg border border-hair bg-raise p-4">
            <p className="text-[12.5px] font-medium text-ink">Invite to an interview</p>
            <p className="mt-1 text-[11.5px] leading-relaxed text-ink-mid">
              Mints a session bound to this candidate and this interviewer. The link is the whole
              credential — there is no authentication, so treat it as a secret.
            </p>
            <div className="mt-2.5 flex flex-wrap items-center gap-2">
              <Button onClick={onMint} disabled={busy === `invite:${candidate.id}`}>
                {busy === `invite:${candidate.id}` ? "Creating…" : "Create interview link"}
              </Button>
              {!candidate.agent_id ? (
                <span className="text-[11.5px] text-warn">Set an interviewer first</span>
              ) : null}
            </div>
            {invite ? (
              <div className="mt-3">
                <code className="block overflow-x-auto rounded border border-hair bg-glass px-2.5 py-2 text-[11.5px] text-ink-mid">
                  {link}
                </code>
                <div className="mt-2 flex gap-2">
                  <Button variant="ghost" onClick={() => onCopy(link)}>
                    {copied ? "Copied" : "Copy link"}
                  </Button>
                  <a
                    href={invite.path}
                    className="text-[12px] text-accent underline decoration-accent/40 underline-offset-2"
                  >
                    Open it yourself
                  </a>
                </div>
              </div>
            ) : null}
          </div>

          <div>
            <p className="mb-1.5 text-[11px] font-medium tracking-[0.07em] uppercase text-ink-low">
              Interviews
            </p>
            {sessions === null ? (
              <p className="text-[12px] text-ink-low">Loading…</p>
            ) : sessions.length === 0 ? (
              <p className="text-[12px] text-ink-low">None yet.</p>
            ) : (
              <ul className="space-y-1.5">
                {sessions.map((session) => (
                  <li
                    key={session.id}
                    className="flex items-center justify-between gap-3 rounded border border-hair bg-glass px-2.5 py-1.5"
                  >
                    <a
                      href={`/sessions/${session.id}`}
                      className="font-mono text-[11.5px] text-accent underline decoration-accent/40 underline-offset-2"
                    >
                      {session.id}
                    </a>
                    <span className="text-[11px] text-ink-low">
                      {session.turns?.length ?? 0} turns ·{" "}
                      {session.ended_at
                        ? (session.scoring?.status ?? "ended")
                        : "in progress"}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>
    </Card>
  );
}
