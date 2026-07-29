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

import { use, useEffect, useRef, useState } from "react";
import Link from "next/link";

import { Button, Card, CardHeader, Chip, Field, Input, Metric, Page, type Status } from "@/components/ui";
import { AUDIO_LEAD_MS, useSession, type SessionState } from "@/lib/session";
import { useRtc } from "@/lib/rtc";

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

export default function InterviewPage({ params }: { params: Promise<{ id: string }> }) {
  // `params` is a promise in Next 16; `use` unwraps it in a client component.
  const { id } = use(params);
  const session = useSession(API, id);
  const { state, hello, connected, transcript, metrics, error } = session;
  const { connect, disconnect, say, send, startMic, stopMic, attachCanvas } = session;
  // Destructured rather than read through the object: a hook's returned object counts as ref
  // access during render under React's rules, and property reads in JSX would trip it.
  const {
    hasVideo: rtcVideo,
    audioBlocked,
    join: joinRtc,
    leave: leaveRtc,
    unblockAudio,
    attachVideo,
    attachAudio,
    attachSelf,
    publishing,
  } = useRtc(API, id);
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
      title="Interview"
      lede="Talk or type your answers. You can interrupt mid-question — the interviewer stops immediately and will not pretend you heard the rest."
      action={
        <div className="flex gap-2">
          {connected ? (
            <Button
              variant="danger"
              onClick={() => {
                disconnect();
                void leaveRtc();
              }}
            >
              End session
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={() => {
                void connect();
                // Both legs, deliberately. WebRTC carries the media when an SFU is present;
                // the socket carries the microphone, the transcript and the telemetry either
                // way, and is the whole transport when there is no SFU.
                void joinRtc();
              }}
            >
              Start session
            </Button>
          )}
          <Link href="/sessions"><Button>All sessions</Button></Link>
        </div>
      }
    >
      <p className="-mt-4 mb-2 font-mono text-[11px] text-ink-low">session {id}</p>

      {error ? (
        <Card className="border-bad/40 bg-bad/5">
          <div className="px-5 py-4">
            <p className="text-[13px] font-medium text-bad">Cannot reach the runtime</p>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-ink-mid">{error}</p>
          </div>
        </Card>
      ) : null}

      {audioBlocked ? (
        <Card className="border-warn/40 bg-warn/5">
          <div className="flex flex-wrap items-center gap-3 px-5 py-4">
            <p className="min-w-0 flex-1 text-[12.5px] leading-relaxed text-ink-mid">
              <span className="text-warn">The browser blocked audio playback.</span> Video plays
              and the mouth moves, so this looks like broken audio rather than a policy that
              needs one click to satisfy.
            </p>
            <Button variant="primary" onClick={() => void unblockAudio()}>
              Enable audio
            </Button>
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
            action={
              <div className="flex items-center gap-2">
                <Chip status={rtcVideo ? "ok" : "neutral"}>
                  {rtcVideo ? (publishing ? "webrtc · two-way" : "webrtc · receive only") : "websocket"}
                </Chip>
                <Chip status={STATE_TONE[state]}>{state.toLowerCase()}</Chip>
              </div>
            }
          />
          <div className="p-5">
            <div className="relative overflow-hidden rounded-lg border border-hair bg-black">
              {/* Fixed aspect so the panel does not jump when the renderer's size changes. */}
              <div className="aspect-video w-full">
                {/* Two surfaces, one shown at a time. The WebRTC <video> is preferred when the
                    SFU is delivering a track; the canvas is the WebSocket path's renderer and
                    also the fallback. Both are always mounted rather than swapped, because
                    attaching a track to an element that does not exist yet silently does
                    nothing — and that failure looks exactly like a black video. */}
                <video
                  ref={attachVideo}
                  autoPlay
                  playsInline
                  muted
                  aria-label="Interviewer video"
                  className={rtcVideo ? "size-full object-contain" : "hidden"}
                />
                <canvas
                  ref={attachCanvas}
                  width={256}
                  height={144}
                  aria-label="Interviewer video"
                  className={rtcVideo ? "hidden" : "size-full object-contain"}
                  // Nearest-neighbour: the placeholder renders 256×144 flat rectangles, and
                  // smoothing them into a blur hides whether the mouth is actually moving.
                  style={{ imageRendering: "pixelated" }}
                />
                {/* Audio is a separate element: the video is muted so the WebSocket path's Web
                    Audio scheduling and the WebRTC track can never both play the same speech. */}
                <audio ref={attachAudio} autoPlay />

                {/* The candidate's own camera, inset. Muted and mirrored, because both are what
                    every video call does and their absence is immediately uncanny: unmuted
                    would echo their own voice back, and unmirrored makes raising your right
                    hand look like your left. Mounted always, hidden until publishing, for the
                    same reason as the interviewer's video -- attaching a track to an element
                    that does not exist yet silently does nothing. */}
                <video
                  ref={attachSelf}
                  autoPlay
                  playsInline
                  muted
                  aria-label="Your camera"
                  className={[
                    "absolute right-3 bottom-3 w-1/4 min-w-28 rounded-lg border border-hair-strong",
                    "bg-black object-cover shadow-lg",
                    publishing ? "" : "hidden",
                  ].join(" ")}
                  style={{ transform: "scaleX(-1)" }}
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
          hint={
            rtcVideo
              ? "First-frame and end-to-end read as dashes on the WebRTC path, and that is honest rather than broken — see below."
              : "Every figure comes from a real run. Targets are the reasoned budget, not achieved numbers — the gap is the point."
          }
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
        {rtcVideo ? (
          <div className="border-t border-hair px-5 py-3">
            <p className="max-w-3xl text-[11.5px] leading-relaxed text-ink-mid">
              {/* Said plainly rather than left as dashes that look like "not yet". A number
                  nobody can source is worse than an absent one. */}
              <span className="text-warn">Not measurable over WebRTC as built.</span> The
              server-side stages above are still real. <span className="text-ink">First
              frame</span> and <span className="text-ink">end-to-end</span> were reported by the
              canvas client, which stamped the moment it painted a frame carrying a turn epoch. A
              WebRTC video track carries no epoch, so a paint cannot be attributed to a turn —
              correlating them needs the epoch sent over the data channel and matched to
              <code className="mx-1 font-mono">requestVideoFrameCallback</code>, which is real
              work and not yet done. Switch off the SFU to measure them.
            </p>
          </div>
        ) : null}
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
