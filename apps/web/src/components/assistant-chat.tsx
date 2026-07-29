"use client";

/**
 * The conversation surface, shared by the docked panel and the full page.
 *
 * One component rather than two, because the docked version is the one people will actually use and
 * a second copy of it would drift — the page would keep a feature the dock lost, or the reverse, and
 * the difference would be invisible until someone compared them.
 *
 * **Tool steps are the feature.** An answer spanning six candidates is fifteen seconds of tool calls
 * and then prose, and an undifferentiated spinner for that reads as broken. Showing each call turns
 * the wait into progress and makes the answer checkable: you can see it read interview quality before
 * commenting on a low score, or that it never opened the transcript it is describing.
 *
 * **The screen context is shown, not just sent.** The panel says which screen it is reading, because
 * an assistant that silently resolves "this session" from a URL is one you cannot correct when it has
 * the wrong one.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { Prose } from "@/components/prose";
import { Button } from "@/components/ui";
import { useAssistant, type Message } from "@/lib/assistant";
import { useScreen } from "@/lib/screen";
import { useVoice, useVoiceStatus } from "@/lib/voice";

/** Tool names as an operator would say them. Falls back to the raw name, so nothing is invisible. */
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

/**
 * Openers that show what this is for.
 *
 * Deliberately the things a person cannot do well by hand — cross-session consistency, coverage
 * gaps, separating a bad interview from a weak candidate — rather than "summarise this session",
 * which the report page already answers. A first screen of easy cases teaches the wrong idea of what
 * this is.
 */
const SUGGESTIONS = [
  "Where do the ratings contradict each other across candidates?",
  "Which competencies are we never actually probing?",
  "Was this interview any good, as an interview?",
] as const;

export function AssistantChat({ actor, compact = false }: { actor: string; compact?: boolean }) {
  const screen = useScreen();
  const { messages, busy, send, stop, reset } = useAssistant(actor, screen);
  const voiceStatus = useVoiceStatus();
  const voice = useVoice();
  const [draft, setDraft] = useState("");
  const [autoSpeak, setAutoSpeak] = useState(false);
  const scroller = useRef<HTMLDivElement | null>(null);
  const spokenFor = useRef(-1);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  // Speak a completed answer once. Keyed on the message index rather than on content, because
  // content changes on every token and would restart playback mid-sentence.
  useEffect(() => {
    if (!autoSpeak || busy) return;
    const index = messages.length - 1;
    const last = messages[index];
    if (!last || last.role !== "assistant" || !last.content || index === spokenFor.current) return;
    spokenFor.current = index;
    void voice.say(last.content);
  }, [autoSpeak, busy, messages, voice]);

  const submit = useCallback(
    (question: string) => {
      void send(question);
      setDraft("");
    },
    [send],
  );

  /**
   * Push to talk. The transcript lands in the input rather than being sent, so a mis-heard word can
   * be fixed before it becomes a question — dictation that submits itself is only pleasant when it
   * is always right.
   */
  const toggleMic = useCallback(async () => {
    if (voice.recording) {
      const text = await voice.stopRecording();
      if (text) setDraft((current) => (current ? `${current} ${text}` : text));
      return;
    }
    await voice.startRecording();
  }, [voice]);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-hair px-4 py-2.5">
        {/* What it will resolve "this" to. Shown so a wrong context is correctable. */}
        <span className="text-[11.5px] text-ink-low">
          reading <span className="text-ink-mid">{screen.label}</span>
        </span>
        <span className="ml-auto flex items-center gap-1.5">
          {voiceStatus?.available ? (
            <button
              type="button"
              onClick={() => {
                if (voice.speaking) voice.stopSpeaking();
                setAutoSpeak((on) => !on);
              }}
              aria-pressed={autoSpeak}
              title={autoSpeak ? "Stop reading answers aloud" : "Read answers aloud"}
              className={[
                "rounded-md border px-2 py-1 text-[11.5px] transition-colors",
                autoSpeak
                  ? "border-accent/50 bg-accent/15 text-accent"
                  : "border-hair-strong bg-glass-raise text-ink-low hover:text-ink",
              ].join(" ")}
            >
              {voice.speaking ? "speaking…" : autoSpeak ? "voice on" : "voice off"}
            </button>
          ) : null}
          <button
            type="button"
            onClick={reset}
            disabled={messages.length === 0}
            className="rounded-md border border-hair-strong bg-glass-raise px-2 py-1 text-[11.5px] text-ink-low transition-colors hover:text-ink disabled:opacity-40"
          >
            clear
          </button>
        </span>
      </div>

      <div
        ref={scroller}
        className={[
          "min-h-0 flex-1 space-y-4 overflow-y-auto px-4 py-4",
          compact ? "" : "max-h-[58vh] min-h-72",
        ].join(" ")}
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-start gap-2 py-2">
            <p className="text-[12px] text-ink-mid">Hard to do by hand:</p>
            {SUGGESTIONS.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => submit(suggestion)}
                className="rounded-lg border border-hair-strong bg-glass-raise px-3 py-2 text-left text-[12px] text-ink-mid transition-colors hover:border-ink-low hover:text-ink"
              >
                {suggestion}
              </button>
            ))}
          </div>
        ) : null}

        {messages.map((message, index) => (
          <Bubble
            key={index}
            message={message}
            streaming={busy && index === messages.length - 1}
          />
        ))}
      </div>

      {voice.problem ? (
        <p className="border-t border-warn/40 bg-warn/5 px-4 py-2 text-[11.5px] text-warn">
          {voice.problem}
        </p>
      ) : null}

      <form
        className="flex items-center gap-2 border-t border-hair px-4 py-3"
        onSubmit={(event) => {
          event.preventDefault();
          submit(draft);
        }}
      >
        {voiceStatus?.available ? (
          <button
            type="button"
            onClick={() => void toggleMic()}
            disabled={voice.transcribing}
            aria-label={voice.recording ? "Stop recording" : "Ask by voice"}
            title={voice.recording ? "Stop recording" : "Ask by voice"}
            className={[
              "grid size-9 shrink-0 place-items-center rounded-full border transition-colors",
              voice.recording
                ? "animate-pulse border-bad/50 bg-bad/15 text-bad"
                : "border-hair-strong bg-glass-raise text-ink-mid hover:text-ink",
            ].join(" ")}
          >
            <MicGlyph />
          </button>
        ) : null}
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={
            voice.transcribing
              ? "transcribing…"
              : voice.recording
                ? "listening — press the mic again to stop"
                : "Ask about this screen, or across all interviews…"
          }
          aria-label="Ask the assistant"
          className="min-w-0 flex-1 rounded-lg border border-hair-strong bg-base px-3 py-2 text-[12.5px] text-ink placeholder:text-ink-low"
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
    </div>
  );
}

function Bubble({ message, streaming }: { message: Message; streaming: boolean }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <p className="max-w-[88%] rounded-xl border border-accent/30 bg-accent/10 px-3 py-2 text-[12.5px] leading-relaxed text-ink">
          {message.content}
        </p>
      </div>
    );
  }

  const tools = message.tools ?? [];
  return (
    <div className="space-y-2">
      {tools.length > 0 ? (
        <div className="flex flex-wrap gap-1.5">
          {tools.map((tool, index) => (
            <span
              key={`${tool.name}-${index}`}
              className={[
                "inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px]",
                // A write looks different from a read, at the moment it happens. The panel says
                // writes are proposals; this is where that becomes visible rather than a policy.
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
        <p className="rounded-lg border border-bad/40 bg-bad/5 px-3 py-2 text-[12px] leading-relaxed text-bad">
          {message.error}
        </p>
      ) : null}

      {message.content ? (
        <Prose text={message.content} />
      ) : streaming && tools.length === 0 ? (
        <p className="text-[12px] text-ink-low">Thinking…</p>
      ) : null}
    </div>
  );
}

function MicGlyph() {
  return (
    <svg
      width={15}
      height={15}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M12 3.5a2.75 2.75 0 0 0-2.75 2.75v5a2.75 2.75 0 0 0 5.5 0v-5A2.75 2.75 0 0 0 12 3.5Z" />
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
    </svg>
  );
}
