"use client";

/**
 * The report: one interview, laid out for whoever has to make a decision about it.
 *
 * **The quotes are the artefact, not the ratings.** `avatar.scoring` produces no hire
 * recommendation and writes `decision: null` into every record, on the grounds that a hiring call
 * made by a model reading a transcript is a call nobody can be accountable for. This page has to
 * hold that line rather than quietly undo it, so a verdict's quotes are set as body text at the
 * same weight as the rationale, and the rating is a chip beside them — a label on the evidence
 * rather than the headline. A reviewer who disagrees with a rating needs the sentence it was
 * derived from, in one glance, without expanding anything.
 *
 * **Coverage and scoring are shown together per competency because they answer different
 * questions.** Coverage is what the *interview* did: how many times an area was probed and which
 * rubric signals appeared. The rating is what a model thought of the answers. Putting them
 * side by side is what makes `no_evidence` legible — against `asked: 0` it is a fault in the
 * interview, and against `asked: 3` it is a finding about the candidate. Shown apart, the two
 * cases look identical and both read as the candidate's failure.
 *
 * **An unverified quote is the loudest thing on the page.** `verify_quotes` re-checks every quote
 * against the transcript rather than trusting the model, and a quote that is not there means the
 * judge invented evidence. That does not devalue one rating, it devalues the scorecard, so it is
 * rendered as an error and not as a footnote.
 *
 * **The latency panel lives here, not in the interview room.** It was on the candidate's page,
 * where it was instrumentation shown to the one person with no use for it. This is the screen for
 * whoever cares about a p95.
 */

import Link from "next/link";
import { use, useCallback, useEffect, useState } from "react";

import {
  Button,
  Card,
  CardHeader,
  Chip,
  Empty,
  Metric,
  Page,
  type Status,
} from "@/components/ui";

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

type Turn = {
  epoch: number;
  heard: string;
  said: string;
  transcribed: boolean;
  silent: boolean;
  llm_ttft_ms: number | null;
  tts_first_audio_ms: number | null;
  first_frame_ms: number | null;
  perceived_total_ms: number | null;
  interrupted: boolean;
};

type CoverageItem = {
  id: string;
  name: string;
  status: "unasked" | "probing" | "evidenced" | "exhausted";
  asked: number;
  signals_hit: string[];
  signals_total: number;
  evidence: string[];
};

type Verdict = {
  competency_id: string;
  name: string;
  rating: string;
  score: number;
  weight: number;
  rationale: string;
  quotes: string[];
  unverified_quotes: string[];
};

type Scoring = {
  status: "pending" | "scored" | "unavailable";
  model?: string;
  reason?: string;
  scale?: Record<string, number>;
  max_rating?: number;
  weighted_score?: number | null;
  verdicts?: Verdict[];
  decision?: null;
  note?: string;
};

type Recording = {
  status: "off" | "requested" | "unavailable";
  reason?: string;
  filepath?: string;
  room_sid?: string;
};

type Candidate = {
  id: string;
  name: string;
  email: string;
  role: string;
  status: string;
  notes: string | null;
  resume_filename: string | null;
  resume_chars: number | null;
  resume_error: string | null;
};

type Attendance = {
  confirmed_name: string;
  expected_name: string;
  matches_expected: boolean;
  consented_to_recording: boolean;
  user_agent: string;
  timezone: string;
  attested_at: string;
  verified: boolean;
  history?: Attendance[];
};

type Session = {
  id: string;
  agent_id: string | null;
  agent_name?: string | null;
  candidate_id?: string | null;
  started_at: string;
  ended_at: string | null;
  turns: Turn[];
  stale_dropped: number;
  frames_repeated: number;
  coverage?: { plan?: string; focus?: string | null; complete?: boolean;
               competencies?: CoverageItem[] } | null;
  scoring?: Scoring | null;
  recording?: Recording | null;
  attendance?: Attendance | null;
};

/**
 * Recording tone, and `requested` is deliberately `info` rather than `ok`.
 *
 * Nothing in the runtime observes a file being written — the SFU accepts an egress config whether
 * or not a worker exists to act on it — so a green chip here would assert something no code has
 * checked. `info` says "asked for", which is the strongest true statement available.
 */
const RECORDING_TONE: Record<Recording["status"], Status> = {
  requested: "info",
  unavailable: "warn",
  off: "neutral",
};

/** Rating tone. `no_evidence` is neutral, not bad: an absence is not a negative finding. */
const RATING_TONE: Record<string, Status> = {
  strong: "ok",
  adequate: "info",
  weak: "warn",
  no_evidence: "neutral",
};

/** Coverage tone. `exhausted` warns because it means the interview learned nothing there. */
const COVERAGE_TONE: Record<CoverageItem["status"], Status> = {
  evidenced: "ok",
  probing: "info",
  exhausted: "warn",
  unasked: "neutral",
};

function when(iso: string | null): string {
  return iso ? iso.replace("T", " ").replace("+00:00", " UTC") : "—";
}

/** Worst perceived total across turns, which is the figure the latency budget is judged on. */
function worstTotal(turns: Turn[]): number | null {
  const totals = turns
    .map((turn) => turn.perceived_total_ms)
    .filter((value): value is number => value !== null);
  return totals.length ? Math.max(...totals) : null;
}

export default function ReportPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [rescoring, setRescoring] = useState(false);
  const [candidate, setCandidate] = useState<Candidate | null>(null);

  const load = useCallback(() => {
    fetch(`${API}/sessions/${id}`, { cache: "no-store" })
      .then(async (response) => {
        if (response.status === 404) throw new Error(`no session ${id}`);
        if (!response.ok) throw new Error(`the runtime answered ${response.status}`);
        return (await response.json()) as Session;
      })
      .then((loaded) => {
        setSession(loaded);
        setError(null);
        // Fetched after the session rather than in parallel, because the id to fetch comes from
        // it. A missing candidate is not an error here -- the record may have been deleted, and
        // the report says so rather than failing to render.
        if (loaded.candidate_id) {
          fetch(`${API}/candidates/${loaded.candidate_id}`, { cache: "no-store" })
            .then(async (r) => (r.ok ? ((await r.json()) as Candidate) : null))
            .then(setCandidate)
            .catch(() => setCandidate(null));
        } else {
          setCandidate(null);
        }
      })
      .catch((cause: unknown) => {
        setSession(null);
        setError(cause instanceof Error ? cause.message : "the runtime is unreachable");
      });
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  // Scoring runs detached from the request that queued it, so the only way to know it finished is
  // to look again. Polled rather than pushed because a scorecard arrives once, seconds after a
  // session ends -- a socket for that would be more machinery than the problem deserves.
  const pending = session?.scoring?.status === "pending";
  useEffect(() => {
    if (!pending) return;
    const timer = setInterval(load, 2500);
    return () => clearInterval(timer);
  }, [pending, load]);

  const rescore = useCallback(async () => {
    setRescoring(true);
    try {
      await fetch(`${API}/sessions/${id}/score`, { method: "POST" });
      load();
    } finally {
      setRescoring(false);
    }
  }, [id, load]);

  if (error) {
    return (
      <Page title="Report" lede="One interview, as a reviewable record.">
        <Card>
          <Empty title="Could not load this session" action={<Button onClick={load}>Retry</Button>}>
            {error}. The console reads sessions from the runtime on 127.0.0.1:8000 — start it with{" "}
            <code className="font-mono text-ink">uvicorn avatar.server:app</code>, or go back to{" "}
            <Link href="/sessions" className="underline">
              all sessions
            </Link>
            .
          </Empty>
        </Card>
      </Page>
    );
  }

  if (session === null) {
    return (
      <Page title="Report" lede="One interview, as a reviewable record.">
        <Card>
          <div className="px-5 py-14 text-center text-[12.5px] text-ink-mid">Loading…</div>
        </Card>
      </Page>
    );
  }

  const scoring = session.scoring ?? null;
  const coverage = session.coverage?.competencies ?? [];
  const byId = new Map(coverage.map((item) => [item.id, item]));
  const worst = worstTotal(session.turns);

  return (
    <Page
      title="Report"
      lede={`Session ${session.id}${session.agent_name ? ` · ${session.agent_name}` : ""}. Started ${when(session.started_at)}.`}
      action={
        <div className="flex gap-2">
          <Link href="/sessions">
            <Button>All sessions</Button>
          </Link>
          <Button variant="primary" disabled={rescoring} onClick={() => void rescore()}>
            {rescoring ? "Queued…" : "Re-score"}
          </Button>
        </div>
      }
    >
      {candidate ? (
        <Card>
          <CardHeader
            title={candidate.name || "Candidate"}
            hint={
              candidate.role
                ? `Interviewed for ${candidate.role}`
                : "No role was recorded for this interview"
            }
            action={
              <Link href="/candidates">
                <Button>All candidates</Button>
              </Link>
            }
          />
          <div className="grid gap-x-8 gap-y-3 px-5 py-4 md:grid-cols-2">
            {candidate.email ? (
              <p className="text-[12.5px] text-ink-mid">
                <span className="text-ink-low">Email</span> · {candidate.email}
              </p>
            ) : null}
            <p className="text-[12.5px] text-ink-mid">
              <span className="text-ink-low">Resume</span> ·{" "}
              {candidate.resume_error ? (
                <span className="text-warn">
                  did not parse, so this interview ran unbriefed
                </span>
              ) : candidate.resume_filename ? (
                <>
                  {candidate.resume_filename} (
                  {candidate.resume_chars?.toLocaleString() ?? "?"} chars, in the prompt)
                </>
              ) : (
                <span className="text-ink-low">none attached</span>
              )}
            </p>
            {candidate.notes ? (
              <p className="md:col-span-2 text-[12.5px] leading-relaxed text-ink-mid">
                <span className="text-ink-low">Note given to the interviewer</span> ·{" "}
                {candidate.notes}
              </p>
            ) : null}
          </div>
          {/* Said here rather than left implied: a reader comparing two reports needs to know
              whether the interviewer was working from a resume, because an unbriefed interview
              asks more generic questions and that shows up in the coverage below. */}
          <p className="border-t border-hair px-5 py-3 text-[11.5px] leading-relaxed text-ink-low">
            {candidate.resume_filename && !candidate.resume_error
              ? "The interviewer was briefed from this resume, framed as unverified claims to probe rather than facts to accept."
              : "The interviewer had no resume for this interview, so its questions were drawn from the rubric alone."}
          </p>
        </Card>
      ) : session.candidate_id ? (
        <Card>
          <CardHeader
            title="Candidate deleted"
            hint="This interview was conducted with a candidate whose record has since been removed. The transcript is kept deliberately — it is evidence of something that happened."
          />
        </Card>
      ) : null}

      <AttendanceCard attendance={session.attendance ?? null} />

      <Card>
        <CardHeader
          title="Assessment"
          hint="Model-generated, and deliberately not a decision. Ratings summarise the quotes beneath them; the quotes are the part worth checking, and every one has been matched against the transcript."
        />
        {scoring === null || scoring.status === "pending" ? (
          <div className="px-5 py-10 text-center text-[12.5px] text-ink-mid">
            {scoring === null
              ? "This session has not been scored. End it, or press Re-score."
              : "Scoring in progress — one model call per competency, running in the background."}
          </div>
        ) : scoring.status === "unavailable" ? (
          <div className="px-5 py-5">
            <Chip status="warn">not scored</Chip>
            {/* The reason, verbatim. "Unavailable" without it is indistinguishable from a bug,
                and the reasons are all actionable: no rubric, no turns, or no model configured. */}
            <p className="mt-2.5 max-w-3xl text-[12.5px] leading-relaxed text-ink-mid">
              {scoring.reason}
            </p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 divide-hair border-b border-hair sm:grid-cols-4">
              <Metric
                label="Weighted score"
                value={
                  scoring.weighted_score === null || scoring.weighted_score === undefined
                    ? "—"
                    : `${Math.round(scoring.weighted_score * 100)}%`
                }
                target="a summary of the ratings, not a threshold"
              />
              <Metric label="Competencies" value={scoring.verdicts?.length ?? 0} />
              <Metric
                label="Judged by"
                value={scoring.model || "—"}
                target="the same transcript re-scores identically"
              />
              <Metric
                label="Decision"
                value="human"
                target="this page does not make one"
                status="neutral"
              />
            </div>

            <div className="divide-y divide-hair">
              {(scoring.verdicts ?? []).map((verdict) => {
                const cover = byId.get(verdict.competency_id);
                return (
                  <div key={verdict.competency_id} className="px-5 py-5">
                    <div className="flex flex-wrap items-center gap-2.5">
                      <p className="text-[13.5px] font-medium text-ink">{verdict.name}</p>
                      <Chip status={RATING_TONE[verdict.rating] ?? "neutral"}>
                        {verdict.rating.replace("_", " ")}
                      </Chip>
                      {/* Coverage beside the rating, because it is what makes the rating
                          legible: no_evidence against asked=0 is the interview's failure, and
                          against asked=3 it is a finding. */}
                      {cover ? (
                        <Chip status={COVERAGE_TONE[cover.status]}>
                          probed {cover.asked}× · {cover.signals_hit.length}/
                          {cover.signals_total} signals
                        </Chip>
                      ) : null}
                      {verdict.weight !== 1 ? (
                        <span className="font-mono text-[11px] text-ink-low">
                          weight {verdict.weight}
                        </span>
                      ) : null}
                    </div>

                    {verdict.rationale ? (
                      <p className="mt-2.5 max-w-3xl text-[12.5px] leading-relaxed text-ink-mid">
                        {verdict.rationale}
                      </p>
                    ) : null}

                    {verdict.quotes.length > 0 ? (
                      <div className="mt-3 space-y-2">
                        {verdict.quotes.map((quote, index) => (
                          <blockquote
                            key={index}
                            className="border-l-2 border-accent/40 pl-3 text-[13px] leading-relaxed text-ink"
                          >
                            “{quote}”
                          </blockquote>
                        ))}
                      </div>
                    ) : (
                      <p className="mt-3 text-[12px] text-ink-low">
                        No supporting quote — the rating rests on the rationale alone.
                      </p>
                    )}

                    {/* A `no_evidence` rating with quotes attached is the judge contradicting
                        itself, and it happens: on a real session it rated Debugging as
                        no_evidence and then quoted an answer about on-call. Faithfully showing
                        both without comment would present that quote as support for a rating
                        that denies any exists, so the contradiction is named instead. A caution
                        rather than an error — unlike a fabricated quote, nothing here is
                        untrue, the two halves just disagree. */}
                    {verdict.rating === "no_evidence" && verdict.quotes.length > 0 ? (
                      <p className="mt-2.5 text-[12px] leading-relaxed text-warn">
                        The model rated this no evidence but still returned a quote. Read the
                        quote against the competency yourself — one of the two is wrong.
                      </p>
                    ) : null}

                    {verdict.unverified_quotes.length > 0 ? (
                      <div className="mt-3 rounded-lg border border-bad/40 bg-bad/5 px-3.5 py-3">
                        <p className="text-[12.5px] font-medium text-bad">
                          The model attributed {verdict.unverified_quotes.length} quote
                          {verdict.unverified_quotes.length === 1 ? "" : "s"} to the candidate
                          that {verdict.unverified_quotes.length === 1 ? "does" : "do"} not
                          appear in the transcript.
                        </p>
                        <p className="mt-1 text-[12px] leading-relaxed text-ink-mid">
                          Treat this whole scorecard as unreliable and read the transcript
                          yourself. A judge that invents evidence is not one whose other ratings
                          can be trusted.
                        </p>
                        {verdict.unverified_quotes.map((quote, index) => (
                          <p key={index} className="mt-2 font-mono text-[11.5px] text-bad">
                            “{quote}”
                          </p>
                        ))}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>

            {scoring.note ? (
              <p className="border-t border-hair px-5 py-3.5 text-[11.5px] leading-relaxed text-ink-low">
                {scoring.note}
                {scoring.scale ? (
                  <>
                    {" "}
                    Scale:{" "}
                    {Object.entries(scoring.scale)
                      .sort((a, b) => a[1] - b[1])
                      .map(([name, value]) => `${name.replace("_", " ")} = ${value}`)
                      .join(", ")}
                    .
                  </>
                ) : null}
              </p>
            ) : null}
          </>
        )}
      </Card>

      {/* Competencies the scorer did not reach get their own block. A rubric can carry an area
          the interview never touched, and that must not be invisible just because there is no
          verdict row for it. */}
      {coverage.filter((item) => !(scoring?.verdicts ?? []).some((v) => v.competency_id === item.id))
        .length > 0 ? (
        <Card>
          <CardHeader
            title="Not assessed"
            hint="On the rubric, but with no verdict — either the scorer has not run, or it ran before this competency was added."
          />
          <div className="flex flex-wrap gap-2 px-5 py-4">
            {coverage
              .filter(
                (item) => !(scoring?.verdicts ?? []).some((v) => v.competency_id === item.id),
              )
              .map((item) => (
                <Chip key={item.id} status={COVERAGE_TONE[item.status]}>
                  {item.name} · probed {item.asked}×
                </Chip>
              ))}
          </div>
        </Card>
      ) : null}

      <Card>
        <CardHeader
          title="Recording"
          hint="Egress is configured on the room at creation rather than started by this code, so the SFU owns it for the room's whole lifetime. What that cannot tell us is whether a file was actually produced — that needs an egress service, and its absence is silent."
        />
        <div className="px-5 py-4">
          {session.recording ? (
            <>
              <Chip status={RECORDING_TONE[session.recording.status]}>
                {session.recording.status === "requested"
                  ? "requested"
                  : session.recording.status === "unavailable"
                    ? "not set up"
                    : "not requested"}
              </Chip>
              {session.recording.filepath ? (
                <p className="mt-2.5 font-mono text-[11.5px] text-ink-mid">
                  {session.recording.filepath}
                </p>
              ) : null}
              <p className="mt-2 max-w-3xl text-[12px] leading-relaxed text-ink-low">
                {session.recording.reason}
              </p>
            </>
          ) : (
            <p className="text-[12.5px] text-ink-mid">
              {/* No field at all means the session predates recording, or ran on the WebSocket
                  transport where there is no room to configure. Distinguished from "off", which
                  is a recording that was deliberately not asked for. */}
              This session has no recording state — it ran before recording existed, or on the
              WebSocket transport, which has no room to attach egress to.
            </p>
          )}
        </div>
      </Card>

      <Card>
        <CardHeader
          title="Transcript"
          hint="What was heard and what was said, in the order it happened. An answer with no transcript is marked rather than dropped — a silent turn and a failed transcription are different problems."
        />
        {session.turns.length === 0 ? (
          <div className="px-5 py-10 text-center text-[12.5px] text-ink-mid">
            No turns were recorded.
          </div>
        ) : (
          <div className="space-y-4 px-5 py-5">
            {/* Keyed by position, not by epoch. Epoch is not unique across a session's turns:
                a turn refused by the state machine leaves the epoch where it was, so two records
                can share one. Using it as a key rendered duplicates and dropped rows, which React
                reported and which would have gone unnoticed on any session where it did not
                happen. Position is the real identity of a record in an append-only list. */}
            {session.turns.map((turn, index) => (
              <div key={index} className="space-y-2">
                {/* Three distinct cases, and conflating any two of them loses the thing a
                    reviewer is reading the transcript for. Words: show them. Silence: say so,
                    because a re-prompt with no candidate line above it looks like a question
                    asked twice for no reason. Speech that failed to transcribe: say that
                    instead, and differently — it is a configuration fault, not a quiet
                    candidate, and it is the one that needs someone to act. */}
                {turn.silent ? (
                  <div className="flex gap-3">
                    <span className="w-20 shrink-0 pt-0.5 text-[10.5px] tracking-[0.07em] uppercase text-ink-low">
                      silence
                    </span>
                    <p className="min-w-0 flex-1 text-[13px] italic leading-relaxed text-ink-low">
                      The candidate did not answer, so the interviewer re-prompted.
                    </p>
                  </div>
                ) : turn.heard || !turn.transcribed ? (
                  <div className="flex gap-3">
                    <span className="w-20 shrink-0 pt-0.5 text-[10.5px] tracking-[0.07em] uppercase text-ink-low">
                      candidate
                    </span>
                    <p
                      className={`min-w-0 flex-1 text-[13px] leading-relaxed ${
                        turn.transcribed ? "text-ink" : "text-warn italic"
                      }`}
                    >
                      {turn.heard}
                      {turn.transcribed ? "" : " — no transcript was produced for this turn"}
                    </p>
                  </div>
                ) : null}
                {turn.said ? (
                  <div className="flex gap-3">
                    <span className="w-20 shrink-0 pt-0.5 text-[10.5px] tracking-[0.07em] uppercase text-ink-low">
                      interviewer
                    </span>
                    <p className="min-w-0 flex-1 text-[13px] leading-relaxed text-ink-mid">
                      {turn.said}
                      {turn.interrupted ? (
                        <span className="ml-2 text-[11.5px] text-warn">
                          — interrupted by the candidate
                        </span>
                      ) : null}
                    </p>
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        )}
      </Card>

      <Card>
        <CardHeader
          title="Delivery"
          hint="Moved here from the candidate's page, where it was instrumentation shown to the one person who has no use for it. A dash means the turn was cut off before that stage completed, which is a real record rather than a missing one."
        />
        <div className="grid grid-cols-2 divide-hair border-b border-hair sm:grid-cols-4">
          <Metric label="Turns" value={session.turns.length} />
          <Metric
            label="Barge-ins"
            value={session.turns.filter((turn) => turn.interrupted).length}
          />
          <Metric
            label="Stale artifacts dropped"
            value={session.stale_dropped}
            target="proof barge-in worked by invalidation"
          />
          <Metric
            label="Worst end-to-end"
            value={worst === null ? "—" : `${Math.round(worst)} ms`}
            target="sub-second"
            status={worst !== null && worst > 1000 ? "bad" : "neutral"}
          />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="border-b border-hair text-[10.5px] tracking-[0.07em] uppercase text-ink-low">
                <th className="px-5 py-2.5 text-left font-medium">epoch</th>
                <th className="w-0 px-3 py-2.5 text-right font-medium whitespace-nowrap">
                  llm ttft
                </th>
                <th className="w-0 px-3 py-2.5 text-right font-medium whitespace-nowrap">
                  tts first audio
                </th>
                <th className="w-0 px-3 py-2.5 text-right font-medium whitespace-nowrap">
                  first frame
                </th>
                <th className="w-0 px-5 py-2.5 text-right font-medium whitespace-nowrap">
                  perceived total
                </th>
              </tr>
            </thead>
            <tbody>
              {session.turns.map((turn, index) => (
                <tr key={index} className="border-b border-hair/60 last:border-0">
                  <td className="px-5 py-2.5 font-mono text-ink-mid">{turn.epoch}</td>
                  {(
                    [
                      turn.llm_ttft_ms,
                      turn.tts_first_audio_ms,
                      turn.first_frame_ms,
                      turn.perceived_total_ms,
                    ] as const
                  ).map((value, index) => (
                    <td
                      key={index}
                      className={`px-3 py-2.5 text-right font-mono whitespace-nowrap ${
                        index === 3 ? "px-5 text-ink" : "text-ink-mid"
                      }`}
                    >
                      {value === null ? "—" : Math.round(value)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {worst === null ? (
          <p className="border-t border-hair px-5 py-3.5 text-[11.5px] leading-relaxed text-ink-low">
            No end-to-end figure was reported. That measurement closes only when the client
            reports having painted a frame, so a headless driver or a backgrounded tab produces a
            complete session with no total — the stage timings above are still the server&apos;s
            own.
          </p>
        ) : null}
      </Card>
    </Page>
  );
}


/**
 * Who says they sat this interview — and the fact that nobody checked.
 *
 * **The heading is "attested" and never "verified", deliberately.** There is no authentication in
 * this system; the interview link is the whole credential. A reviewer who reads this card and comes
 * away believing identity was confirmed would be making a hiring decision on a check nobody
 * performed, which is a specific and foreseeable harm — so the limitation is stated on the card
 * rather than in a footnote or a manual.
 *
 * **A name mismatch is shown, not resolved.** The comparison is case- and space-insensitive, so a
 * differing name here is a deliberate act rather than a typo. What it means is a human's judgment:
 * a married name, a preferred name, and someone else sitting the interview all look identical from
 * here, and the card says so instead of picking one.
 */
function AttendanceCard({ attendance }: { attendance: Attendance | null }) {
  if (!attendance?.attested_at) {
    return (
      <Card>
        <CardHeader
          title="Attendance not recorded"
          hint="No one confirmed a name before this interview started — it predates that step, or was joined directly rather than through an invite."
        />
      </Card>
    );
  }

  const mismatch = Boolean(attendance.expected_name) && !attendance.matches_expected;
  const rejoins = attendance.history?.length ?? 0;

  return (
    <Card>
      <CardHeader
        title="Attendance — attested, not verified"
        hint="What the person joining typed about themselves. Nothing here proves identity; there is no authentication in this system and the link is the whole credential."
        action={
          <Chip status={mismatch ? "warn" : "neutral"}>
            {mismatch ? "name differs" : "name as arranged"}
          </Chip>
        }
      />
      <div className="grid gap-x-8 gap-y-3 px-5 py-4 md:grid-cols-2">
        <p className="text-[12.5px] text-ink-mid">
          <span className="text-ink-low">Typed</span> ·{" "}
          <span className={mismatch ? "text-warn" : "text-ink"}>
            {attendance.confirmed_name || "—"}
          </span>
        </p>
        <p className="text-[12.5px] text-ink-mid">
          <span className="text-ink-low">Arranged for</span> ·{" "}
          {attendance.expected_name || <span className="text-ink-low">no candidate on record</span>}
        </p>
        <p className="text-[12.5px] text-ink-mid">
          <span className="text-ink-low">Attested at</span> · {when(attendance.attested_at)}
        </p>
        <p className="text-[12.5px] text-ink-mid">
          <span className="text-ink-low">Recording notice</span> ·{" "}
          {attendance.consented_to_recording ? (
            "accepted"
          ) : (
            <span className="text-warn">not accepted</span>
          )}
        </p>
        {attendance.timezone ? (
          <p className="text-[12.5px] text-ink-mid">
            <span className="text-ink-low">Joined from</span> · {attendance.timezone}
          </p>
        ) : null}
        {rejoins ? (
          <p className="text-[12.5px] text-ink-mid">
            <span className="text-ink-low">Earlier attestations</span> ·{" "}
            <span className="text-warn">
              {rejoins} — {attendance.history?.map((h) => h.confirmed_name).join(", ")}
            </span>
          </p>
        ) : null}
      </div>
      {mismatch ? (
        <p className="border-t border-hair px-5 py-3 text-[11.5px] leading-relaxed text-warn">
          The name entered is not the name this interview was arranged for. A preferred name, a
          married name, and a different person all look the same from here — this needs a human to
          judge before the assessment below is used.
        </p>
      ) : null}
    </Card>
  );
}
