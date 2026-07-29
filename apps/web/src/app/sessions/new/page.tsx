"use client";

/**
 * The live interview. Video on the left, conversation on the right, instrumentation beneath.
 *
 * Laid out the way the thing is actually used rather than the way the data is shaped: a
 * candidate watches a face and reads what was said, so those get the space, and the telemetry
 * that makes the system provable sits below the fold where it does not compete.
 *
 * The instrumentation is on the same page as the conversation on purpose. "The avatar reacted
 * immediately" is an assertion; a turn epoch incrementing beside a transcript bubble going
 * dashed is evidence, and a demo that has to switch tabs to show it convinces nobody.
 */

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

import { Button, Card, CardHeader, Chip, Field, Input, Metric, Page, type Status } from "@/components/ui";
import { AUDIO_LEAD_MS, useSession, type SessionState } from "@/lib/session";

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

/** Session state maps onto the semantic palette, matching the runtime's own colours. */
const STATE_TONE: Record<SessionState, Status> = {
  INITIALIZING: "neutral",
  IDLE: "neutral",
  LISTENING: "info",
  THINKING: "info",
  SPEAKING: "warn",
  CANCELLING: "info",
  CLOSED: "neutral",
};

const STAGE_LABELS: Array<[string, string, number | undefined]> = [
  ["turn_detect", "End-of-turn", 300],
  ["llm_ttft", "LLM first token", 500],
  ["tts_first_audio", "Voice first audio", 300],
  ["avatar_first_frame", "First frame", 150],
  ["perceived_total", "End-to-end, to paint", 1000],
];

export default function LiveSessionPage() {
  const session = useSession(API);
  const { state, hello, connected, transcript, metrics, error } = session;
  const { connect, disconnect, say, send, startMic, stopMic, attachCanvas } = session;
  const [draft, setDraft] = useState("");
  const [micOn, setMicOn] = useState(false);
  const scroller = useRef<HTMLDivElement | null>(null);

  // Follow the conversation as it grows. Without this the newest turn is off-screen exactly
  // when someone is watching for it.
  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [transcript]);

  async function toggleMic() {
    if (micOn) {
      stopMic();
      setMicOn(false);
      return;
    }
    try {
      await startMic();
      setMicOn(true);
    } catch {
      /* permission denied — the mic-uploaded metric staying at 0 is the tell */
    }
  }

  return (
    <Page
      title="Live session"
      lede="Talk or type. Interrupt mid-answer to see the turn epoch increment and the interrupted question go dashed — that is barge-in, shown rather than claimed."
      action={
        <div className="flex gap-2">
          {connected ? (
            <Button variant="danger" onClick={disconnect}>End session</Button>
          ) : (
            <Button variant="primary" onClick={() => void connect()}>Start session</Button>
          )}
          <Link href="/sessions"><Button>All sessions</Button></Link>
        </div>
      }
    >
      {error ? (
        <Card className="border-bad/40 bg-bad/5">
          <div className="px-5 py-4">
            <p className="text-[13px] font-medium text-bad">Cannot reach the runtime</p>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-mid">{error}</p>
          </div>
        </Card>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        {/* ---------------------------------------------------------- video */}
        <Card>
          <CardHeader
            title="Interviewer"
            hint={
              hello
                ? `${hello.renderer} · ${hello.frame_width}×${hello.frame_height} · target ${hello.target_fps}fps`
                : "not connected"
            }
            action={<Chip status={STATE_TONE[state]}>{state.toLowerCase()}</Chip>}
          />
          <div className="p-5">
            <div className="relative overflow-hidden rounded-lg border border-hair bg-black">
              {/* Fixed aspect so the panel does not jump when the renderer's size changes. */}
              <div className="aspect-video w-full">
                <canvas
                  ref={attachCanvas}
                  width={256}
                  height={144}
                  aria-label="Interviewer video"
                  className="size-full object-contain"
                  // Nearest-neighbour: the placeholder renders 256×144 flat rectangles, and
                  // smoothing them into a blur hides whether the mouth is actually moving.
                  style={{ imageRendering: "pixelated" }}
                />
              </div>
              {!connected ? (
                <div className="absolute inset-0 grid place-items-center bg-base/80 text-[12.5px] text-ink-mid">
                  Press Start session
                </div>
              ) : null}
            </div>

            <div className="mt-4 flex flex-wrap items-center gap-2">
              <Button disabled={!connected} onClick={() => send({ type: "speech_start" })}>
                Starts speaking
              </Button>
              <Button
                disabled={!connected}
                onClick={() =>
                  send({ type: "end_of_turn", transcript: "[candidate finished speaking]" })
                }
              >
                Stops speaking
              </Button>
              <Button
                variant={micOn ? "primary" : "ghost"}
                disabled={!connected}
                onClick={() => void toggleMic()}
              >
                {micOn ? "Mic on — server detects turns" : "Stream mic"}
              </Button>
              {hello ? (
                <span className="nums ml-auto text-[11.5px] text-ink-low">
                  end-of-turn after {hello.end_of_turn_silence_ms}ms silence
                </span>
              ) : null}
            </div>
          </div>
        </Card>

        {/* ----------------------------------------------------- conversation */}
        <Card className="flex min-h-0 flex-col">
          <CardHeader
            title="Conversation"
            hint="What the interviewer was actually given, and what it said back"
          />
          <div ref={scroller} className="flex-1 space-y-2.5 overflow-y-auto px-5 py-4" style={{ maxHeight: 420 }}>
            {transcript.length === 0 ? (
              <p className="py-8 text-center text-[12.5px] text-ink-low">
                No turns yet. Start the session, then type below or stream your microphone.
              </p>
            ) : null}
            {transcript.map((line, index) => (
              <div
                key={`${line.who}-${line.id}-${index}`}
                className={[
                  "max-w-[85%] rounded-xl border px-3 py-2 text-[12.5px] leading-relaxed",
                  line.who === "candidate"
                    ? line.empty
                      ? // A turn the transcriber produced no words for. Shown rather than
                        // omitted, because the interviewer's next question cannot follow up on
                        // something it never received, and that is invisible otherwise.
                        "ml-auto border-dashed border-warn bg-transparent italic text-warn"
                      : "ml-auto border-accent/30 bg-accent/10 text-ink"
                    : "border-hair bg-glass-raise text-ink",
                  line.interrupted ? "border-dashed opacity-75" : "",
                ].join(" ")}
              >
                <span className="mb-1 block text-[10px] tracking-[0.07em] uppercase text-ink-low">
                  {line.who === "candidate"
                    ? line.empty
                      ? "you — nothing transcribed"
                      : "you"
                    : "interviewer"}
                </span>
                {line.text}
                {line.interrupted ? " …" : ""}
              </div>
            ))}
          </div>

          <form
            className="flex gap-2 border-t border-hair px-5 py-4"
            onSubmit={(event) => {
              event.preventDefault();
              say(draft);
              setDraft("");
            }}
          >
            <Field label="">
              <Input
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                disabled={!connected}
                placeholder="Type an answer — no microphone needed"
                aria-label="Type an answer to the interviewer"
              />
            </Field>
            <div className="self-end">
              {/* Enabled while speaking, deliberately: sending then is a barge-in, which is
                  the same thing talking over it does. */}
              <Button type="submit" variant="primary" disabled={!connected}>Send</Button>
            </div>
          </form>
        </Card>
      </div>

      {/* --------------------------------------------------------- telemetry */}
      <Card>
        <CardHeader
          title="Measured this session"
          hint="Every figure comes from a real run. Targets are the reasoned budget, not achieved numbers — the gap is the point."
        />
        <div className="grid grid-cols-2 divide-hair sm:grid-cols-3 lg:grid-cols-5">
          {STAGE_LABELS.map(([key, label, target]) => {
            const value = metrics.stages[key];
            const over = target !== undefined && value !== undefined && value > target;
            return (
              <div key={key} className="border-t border-hair">
                <Metric
                  label={label}
                  value={value === undefined ? "—" : Math.round(value)}
                  unit={value === undefined ? undefined : "ms"}
                  target={target ? `target < ${target}ms` : undefined}
                  status={value === undefined ? "neutral" : over ? "bad" : "ok"}
                />
              </div>
            );
          })}
        </div>
        <div className="grid grid-cols-2 border-t border-hair sm:grid-cols-3 lg:grid-cols-5">
          <Metric label="Client fps" value={metrics.fps} status={metrics.fps >= 20 ? "ok" : "neutral"} />
          <Metric label="Turn epoch" value={metrics.epoch} target="increments on every barge-in" />
          <Metric label="Audio acked" value={metrics.audioAckedMs} unit="ms" target="drives history truncation" />
          <Metric
            label="Audio underruns"
            value={metrics.underruns}
            target={`${AUDIO_LEAD_MS}ms jitter buffer`}
            status={metrics.underruns > 0 ? "warn" : "ok"}
          />
          <Metric
            label="Speech probability"
            value={metrics.speechProbability === null ? "—" : metrics.speechProbability.toFixed(2)}
            target={hello ? `${hello.vad} detector` : undefined}
          />
        </div>
      </Card>
    </Page>
  );
}
