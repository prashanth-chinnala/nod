"use client";

/**
 * The assistant: ask questions about interviews, see what it looked at to answer.
 *
 * **The tool steps are the feature, not decoration.** An answer about six candidates is fifteen
 * seconds of tool calls and then some prose, and a bare spinner for that makes the whole thing feel
 * unreliable. Showing each call as it runs does two things: the wait becomes progress, and the
 * answer becomes checkable — you can see it read interview quality before it commented on a low
 * score, or that it never opened the transcript it is describing. That is the difference between
 * trusting an answer and taking it on faith, and it is why the steps stay visible after the answer
 * arrives instead of collapsing away.
 *
 * **Capabilities are disclosed, not asserted.** The panel lists what the assistant can read and what
 * it can change, fetched from the service so it cannot drift from the code. Every write is a
 * proposal a human applies, and saying that once in a doc is not the same as saying it on the screen
 * where someone is about to ask for one.
 *
 * **The suggested questions are chosen deliberately.** They are the things a person cannot do well
 * by hand — cross-session consistency, coverage gaps, separating a bad interview from a weak
 * candidate — rather than "summarise this session", which the report page already answers. A first
 * screen full of the easy cases teaches the wrong mental model of what this is for.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { Prose } from "@/components/prose";
import { Button, Card, CardHeader, Chip, Page } from "@/components/ui";
import {
  fetchCapabilities,
  useAssistant,
  type Capabilities,
  type Message,
} from "@/lib/assistant";

/** Openers that show what this is actually for. See the module docstring. */
const SUGGESTIONS = [
  "Where do the ratings contradict each other across candidates?",
  "Which competencies are we never actually probing?",
  "Was the last interview any good, as an interview?",
  "Which sessions have an unverified quote in their scorecard?",
] as const;

/**
 * Tool names as an operator would say them.
 *
 * A raw `consistency_audit` is legible enough to a developer and reads as machinery to everyone
 * else. Falls back to the raw name so a tool added to the service is never invisible here.
 */
const TOOL_LABELS: Record<string, string> = {
  list_sessions: "listing interviews",
  get_transcript: "reading the transcript",
  get_scorecard: "reading the scorecard",
  get_coverage: "checking what was probed",
  interview_quality: "checking interview quality",
  compare_candidates: "comparing candidates",
  consistency_audit: "auditing consistency",
  coverage_gaps: "looking for coverage gaps",
  action_history: "checking past actions",
  flag_for_review: "flagging for review",
  add_note: "recording a note",
  request_rescore: "proposing a re-score",
  propose_rubric_change: "drafting a rubric change",
  propose_calibration_anchor: "proposing a calibration anchor",
};

const WRITE_TOOLS = new Set([
  "flag_for_review",
  "add_note",
  "request_rescore",
  "propose_rubric_change",
  "propose_calibration_anchor",
]);

export default function AssistantPage() {
  // No accounts exist, so this is a claim rather than an identity. It is sent with every write so a
  // proposal carries a name, and the disclosure panel says plainly that it is unverified.
  const [actor, setActor] = useState("operator");
  const { messages, busy, send, stop, reset } = useAssistant(actor);
  const [draft, setDraft] = useState("");
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [showCapabilities, setShowCapabilities] = useState(false);
  const scroller = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetchCapabilities().then(setCapabilities);
  }, []);

  // Follow the stream. Tokens arrive continuously, so this runs on content length rather than on
  // message count.
  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const submit = useCallback(
    (question: string) => {
      void send(question);
      setDraft("");
    },
    [send],
  );

  return (
    <Page
      title="Assistant"
      lede="Ask about interviews, or about the interviews themselves. It reads records, finds contradictions between them, and drafts changes for you to apply — it never decides anything."
      action={
        <div className="flex gap-2">
          <Button onClick={() => setShowCapabilities((open) => !open)}>
            {showCapabilities ? "Hide" : "What can it do?"}
          </Button>
          <Button onClick={reset} disabled={messages.length === 0}>
            New conversation
          </Button>
        </div>
      }
    >
      {showCapabilities ? <CapabilityPanel capabilities={capabilities} actor={actor} onActor={setActor} /> : null}

      <Card>
        <CardHeader
          title="Conversation"
          hint="Every step it takes is shown. If it answers about a transcript it never opened, you will see that."
        />

        <div ref={scroller} className="max-h-[60vh] min-h-72 space-y-5 overflow-y-auto px-5 py-5">
          {messages.length === 0 ? (
            <div className="py-6">
              <p className="text-[12.5px] text-ink-mid">
                Things worth asking — these are the ones that are hard to do by hand:
              </p>
              <div className="mt-3 flex flex-col items-start gap-2">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion}
                    type="button"
                    onClick={() => submit(suggestion)}
                    className="rounded-lg border border-hair-strong bg-glass-raise px-3 py-2 text-left text-[12.5px] text-ink-mid transition-colors hover:border-ink-low hover:text-ink"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : null}

          {messages.map((message, index) => (
            <Bubble key={index} message={message} streaming={busy && index === messages.length - 1} />
          ))}
        </div>

        <form
          className="flex gap-2 border-t border-hair px-5 py-4"
          onSubmit={(event) => {
            event.preventDefault();
            submit(draft);
          }}
        >
          <input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ask about a session, or across all of them…"
            aria-label="Ask the assistant"
            className="min-w-0 flex-1 rounded-lg border border-hair-strong bg-base px-3 py-2 text-[13px] text-ink placeholder:text-ink-low"
          />
          {busy ? (
            <Button variant="danger" onClick={stop}>
              Stop
            </Button>
          ) : (
            <Button variant="primary" type="submit" disabled={!draft.trim()}>
              Ask
            </Button>
          )}
        </form>
      </Card>
    </Page>
  );
}

/**
 * One message. The assistant's carries its tool steps above the prose.
 *
 * Above rather than below, because they happen first and because they are what is on screen during
 * the wait — moving them under the answer would leave the panel empty for the ten seconds that
 * matter most.
 */
function Bubble({ message, streaming }: { message: Message; streaming: boolean }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <p className="max-w-[85%] rounded-xl border border-accent/30 bg-accent/10 px-3.5 py-2.5 text-[13px] leading-relaxed text-ink">
          {message.content}
        </p>
      </div>
    );
  }

  const tools = message.tools ?? [];
  return (
    <div className="max-w-[92%] space-y-2.5">
      {tools.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {tools.map((tool, index) => (
            <span
              key={`${tool.name}-${index}`}
              className={[
                "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11.5px]",
                // A write is marked differently from a read. The panel says writes are proposals;
                // this is where that distinction is visible at the moment one happens.
                WRITE_TOOLS.has(tool.name)
                  ? "border-warn/40 bg-warn/5 text-warn"
                  : "border-hair-strong bg-glass-raise text-ink-low",
              ].join(" ")}
            >
              <span
                aria-hidden="true"
                className={[
                  "size-1.5 rounded-full",
                  tool.done ? "bg-ok" : "animate-pulse bg-listening",
                ].join(" ")}
              />
              {TOOL_LABELS[tool.name] ?? tool.name}
              {WRITE_TOOLS.has(tool.name) ? " · proposal" : ""}
            </span>
          ))}
        </div>
      ) : null}

      {message.error ? (
        <div className="rounded-lg border border-bad/40 bg-bad/5 px-3.5 py-3">
          <p className="text-[12.5px] leading-relaxed text-bad">{message.error}</p>
        </div>
      ) : null}

      {message.content ? (
        // `Prose` renders a small markdown subset as React elements and never produces HTML from
        // the input, which matters because this text routinely contains verbatim transcript. The
        // first version showed the raw string for that reason and was wrong the other way: the
        // model emits `**` and `- ` constantly, so answers arrived full of asterisks.
        <Prose text={message.content} />
      ) : streaming && tools.length === 0 ? (
        <p className="text-[12.5px] text-ink-low">Thinking…</p>
      ) : null}
    </div>
  );
}

/** What it can read, what it can change, and the two things it cannot do. */
function CapabilityPanel({
  capabilities,
  actor,
  onActor,
}: {
  capabilities: Capabilities | null;
  actor: string;
  onActor: (value: string) => void;
}) {
  return (
    <Card>
      <CardHeader
        title="What the assistant can do"
        hint="Read from the service rather than written here, so this cannot drift from what the code actually allows."
      />
      {capabilities === null ? (
        <div className="px-5 py-5 text-[12.5px] text-ink-mid">
          The assistant service is not reachable on 127.0.0.1:8100. Start it with{" "}
          <code className="font-mono text-ink">
            uvicorn assistant.server:app --port 8100
          </code>{" "}
          from <code className="font-mono text-ink">apps/assistant</code>.
        </div>
      ) : (
        <div className="space-y-5 px-5 py-5">
          <div className="flex flex-wrap items-center gap-2">
            <Chip status="info">model {capabilities.model}</Chip>
            <Chip status={capabilities.writes_are_proposals ? "ok" : "bad"}>
              {capabilities.writes_are_proposals
                ? "writes are proposals only"
                : "writes are applied directly"}
            </Chip>
            <Chip status="warn">no authentication</Chip>
          </div>

          <div className="grid gap-5 sm:grid-cols-2">
            <div>
              <p className="text-[11px] font-medium tracking-[0.07em] uppercase text-ink-low">
                Can read ({capabilities.reads.length})
              </p>
              <ul className="mt-2 space-y-1.5">
                {capabilities.reads.map((tool) => (
                  <li key={tool.name} className="text-[12px] leading-relaxed text-ink-mid">
                    <span className="font-mono text-[11.5px] text-ink">{tool.name}</span> —{" "}
                    {tool.summary}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-[11px] font-medium tracking-[0.07em] uppercase text-ink-low">
                Can propose ({capabilities.writes.length})
              </p>
              <ul className="mt-2 space-y-1.5">
                {capabilities.writes.map((tool) => (
                  <li key={tool.name} className="text-[12px] leading-relaxed text-ink-mid">
                    <span className="font-mono text-[11.5px] text-warn">{tool.name}</span> —{" "}
                    {tool.summary}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* The two refusals, stated on the screen and not only in the prompt. Someone about to
              ask for a ranking should learn the answer here rather than from a refusal. */}
          <div className="rounded-lg border border-hair-strong bg-glass-raise px-4 py-3.5">
            <p className="text-[12px] font-medium text-ink">Two things it will not do</p>
            <p className="mt-1.5 text-[12px] leading-relaxed text-ink-mid">
              It will not recommend hiring or rejecting anyone, or rank candidates — the decision is
              yours, and a call made by a model reading a transcript is one nobody can be
              accountable for. And it will not find candidates similar to a past hire: ranking
              people by resemblance to who was hired before learns that panel&apos;s previous
              preferences, including its biases. Offer letters are useful for calibrating the{" "}
              <em>scale</em> — promoting a verified quote as an example of what a rating sounds like
              — never for modelling a person.
            </p>
          </div>

          <div className="rounded-lg border border-warn/40 bg-warn/5 px-4 py-3.5">
            <p className="text-[12px] leading-relaxed text-ink-mid">
              <span className="font-medium text-warn">No authentication exists.</span>{" "}
              {capabilities.auth}. The name below is attached to any proposal you cause and is not
              verified — it makes the audit trail complete in shape from now rather than starting on
              the day accounts exist.
            </p>
            <label className="mt-3 flex items-center gap-2 text-[12px] text-ink-mid">
              Recorded as
              <input
                value={actor}
                onChange={(event) => onActor(event.target.value)}
                aria-label="Your name, recorded on proposals"
                className="w-48 rounded-lg border border-hair-strong bg-base px-2.5 py-1.5 text-[12.5px] text-ink"
              />
            </label>
          </div>
        </div>
      )}
    </Card>
  );
}
