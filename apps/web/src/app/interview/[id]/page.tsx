"use client";

/**
 * The candidate's interview room. Two screens: a device check, then a call.
 *
 * **What was removed, and why it mattered.** This page used to carry "Starts speaking", "Stops
 * speaking", "Stream mic" and a full latency panel. Every one of those is developer
 * instrumentation, and putting it in front of a candidate does two bad things: it asks them to
 * operate the turn-taking machinery the server-side detector exists to handle, and it shows them
 * a measurement panel belonging to whoever reviews the interview. Turn detection is server-side
 * and tested — the candidate should just talk. The telemetry did not disappear; it belongs on the
 * operator's session view, where the person who cares about a p95 can see it.
 *
 * **Why a pre-join screen rather than joining on load.** Someone about to be recorded answering
 * questions deserves to see their own camera, confirm it works, and read what is about to happen
 * before any of it starts. It is also the only honest place to say the session is recorded —
 * joining immediately means the first thing they learn about the camera is that it was already on.
 *
 * **Why it is dark and chromeless.** A candidate is nervous, and every extra control is one more
 * thing to worry about getting wrong. The console's sidebar is deliberately absent here: this is
 * the only page in the app a non-operator ever sees.
 */

import Link from "next/link";
import { use, useCallback, useEffect, useRef, useState } from "react";

import { Wordmark } from "@/components/logo";
import { Button, Chip } from "@/components/ui";
import { useSession, type SessionState } from "@/lib/session";
import { useRtc } from "@/lib/rtc";

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

/**
 * What the candidate is told, in their terms rather than the state machine's.
 *
 * `CANCELLING` reads as "Listening" on purpose: from the candidate's side, interrupting means the
 * interviewer starts listening. Showing them the internal name of a cancellation would be
 * accurate and useless.
 */
const STATUS: Record<SessionState, { label: string; tone: "ok" | "info" | "warn" | "neutral" }> = {
  INITIALIZING: { label: "Connecting", tone: "neutral" },
  IDLE: { label: "Ready", tone: "neutral" },
  LISTENING: { label: "Listening", tone: "info" },
  THINKING: { label: "Thinking", tone: "info" },
  SPEAKING: { label: "Speaking", tone: "warn" },
  CANCELLING: { label: "Listening", tone: "info" },
  CLOSED: { label: "Ended", tone: "neutral" },
};

export default function InterviewPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const session = useSession(API, id);
  const { state, connected, transcript, error } = session;
  const { connect, disconnect, say, startMic, stopMic, attachCanvas, setLocalPlayback } = session;
  const {
    hasVideo: rtcVideo,
    hasAudio: rtcAudio,
    publishing,
    audioBlocked,
    join: joinRtc,
    leave: leaveRtc,
    unblockAudio,
    attachVideo,
    attachAudio,
    attachSelf,
  } = useRtc(API, id);

  const [joined, setJoined] = useState(false);
  const [micOn, setMicOn] = useState(true);
  const [showCaptions, setShowCaptions] = useState(true);
  const [draft, setDraft] = useState("");
  const [devicesReady, setDevicesReady] = useState<boolean | null>(null);
  const previewRef = useRef<HTMLVideoElement | null>(null);
  const previewStream = useRef<MediaStream | null>(null);
  const scroller = useRef<HTMLDivElement | null>(null);

  // The pre-join preview runs entirely locally: nothing is published until Join is pressed.
  useEffect(() => {
    if (joined) return;
    let cancelled = false;
    void navigator.mediaDevices
      .getUserMedia({ video: true, audio: true })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        previewStream.current = stream;
        if (previewRef.current) previewRef.current.srcObject = stream;
        setDevicesReady(true);
      })
      .catch(() => setDevicesReady(false));
    return () => {
      cancelled = true;
      previewStream.current?.getTracks().forEach((track) => track.stop());
      previewStream.current = null;
    };
  }, [joined]);

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight });
  }, [transcript]);

  // One voice. The runtime tees the same speech down both transports, so whichever path is
  // playing over WebRTC must silence the other — otherwise the candidate hears the interviewer
  // twice, a few hundred milliseconds apart, which reads as a broken model rather than a
  // duplicated track. WebRTC wins because it is the one with a real jitter buffer.
  useEffect(() => {
    setLocalPlayback(!rtcAudio);
  }, [rtcAudio, setLocalPlayback]);

  const join = useCallback(async () => {
    // Release the preview first. Holding the camera open in two places makes the published track
    // fail on some machines, and that failure presents as a camera that simply does not work.
    previewStream.current?.getTracks().forEach((track) => track.stop());
    previewStream.current = null;
    setJoined(true);
    void connect();
    void joinRtc();
    // The microphone starts with the call, because there is no button for it. Asking a candidate
    // to announce when they begin and stop speaking is asking them to do the detector's job.
    try {
      await startMic();
      setMicOn(true);
    } catch {
      setMicOn(false);
    }
  }, [connect, joinRtc, startMic]);

  const leave = useCallback(() => {
    stopMic();
    disconnect();
    void leaveRtc();
    setJoined(false);
  }, [disconnect, leaveRtc, stopMic]);

  const toggleMic = useCallback(() => {
    if (micOn) {
      stopMic();
      setMicOn(false);
      return;
    }
    void startMic().then(() => setMicOn(true));
  }, [micOn, startMic, stopMic]);

  /* ------------------------------------------------------------- pre-join */

  if (!joined) {
    return (
      <div className="mx-auto flex min-h-dvh max-w-2xl flex-col justify-center px-8 py-12">
        <Link
          href="/"
          aria-label="nod — go to the console"
          className="mb-9 block w-fit text-ink transition-colors hover:text-accent"
        >
          {/* The full lockup, trail included. This is the one screen with vertical room for it,
              and it is the first thing a candidate ever sees of the product. */}
          <Wordmark size={32} trail />
        </Link>
        <h1 className="text-[24px] font-semibold tracking-tight text-ink">
          You are about to start your interview
        </h1>
        <p className="mt-2.5 text-[13.5px] leading-relaxed text-ink-mid">
          An interviewer will ask about your experience and follow up on what you say. Speak
          normally — it works out when you have finished, and you can interrupt it mid-question.
        </p>

        <div className="mt-7 overflow-hidden rounded-xl border border-hair bg-glass">
          <div className="aspect-video w-full bg-black">
            <video
              ref={previewRef}
              autoPlay
              playsInline
              muted
              aria-label="Your camera preview"
              className="size-full object-cover"
              // Mirrored, like every video call. Unmirrored makes raising your right hand look
              // like your left, which is subtly disorienting on your own image.
              style={{ transform: "scaleX(-1)" }}
            />
          </div>
          <div className="flex flex-wrap items-center gap-3 border-t border-hair px-5 py-4">
            <Chip
              status={devicesReady === true ? "ok" : devicesReady === false ? "warn" : "neutral"}
            >
              {devicesReady === true
                ? "camera and mic ready"
                : devicesReady === false
                  ? "no camera or mic"
                  : "checking devices"}
            </Chip>
            <div className="ml-auto">
              <Button variant="primary" onClick={() => void join()}>
                Join interview
              </Button>
            </div>
          </div>
        </div>

        {/* Said before the camera is ever published, not after. */}
        <p className="mt-4 text-[12px] leading-relaxed text-ink-low">
          This session is recorded — video, audio and a transcript — and reviewed by a person.
          {devicesReady === false
            ? " No camera or microphone was found, but you can still take the interview by typing."
            : ""}
        </p>
      </div>
    );
  }

  /* ----------------------------------------------------------------- call */

  const status = STATUS[state];

  return (
    <div className="flex min-h-dvh flex-col bg-base">
      <header className="flex items-center gap-3 border-b border-hair px-6 py-3">
        {/* The way out. The room has no sidebar by design, and without this the only exit was the
            browser's back button — which leaves the room without ever calling `leave()`, so the
            mic keeps publishing and the session never ends. This tears down first, then navigates. */}
        <Link
          href="/"
          onClick={leave}
          aria-label="nod — leave and go to the console"
          title="Leave and go to the console"
          className="flex items-center text-ink transition-colors hover:text-accent"
        >
          {/* The lamp reports the session state, which is the one place in the product where
              the mark carries information rather than decoration. The identity's two lamp
              variants are exactly this app's `listening` cyan and `speaking` amber, so "on air"
              is a literal reading of the amber cut rather than a repurposing of it. */}
          <Wordmark size={18} tone={state === "SPEAKING" ? "live" : "rest"} />
        </Link>
        <span className="h-4 w-px bg-hair-strong" aria-hidden="true" />
        <span className="text-[13px] text-ink-mid">Interview</span>
        <Chip status={status.tone}>{status.label}</Chip>
        {rtcVideo ? (
          <Chip status="ok">{publishing ? "two-way" : "receive only"}</Chip>
        ) : null}
        <div className="ml-auto">
          <Button onClick={() => setShowCaptions((on) => !on)}>
            {showCaptions ? "Hide captions" : "Show captions"}
          </Button>
        </div>
      </header>

      {error ? (
        <div className="border-b border-bad/40 bg-bad/5 px-6 py-3">
          <p className="text-[12.5px] text-bad">{error}</p>
        </div>
      ) : null}

      {audioBlocked ? (
        <div className="flex flex-wrap items-center gap-3 border-b border-warn/40 bg-warn/5 px-6 py-3">
          <p className="min-w-0 flex-1 text-[12.5px] text-ink-mid">
            Your browser blocked sound. The interviewer is speaking and you cannot hear it.
          </p>
          <Button variant="primary" onClick={() => void unblockAudio()}>
            Enable sound
          </Button>
        </div>
      ) : null}

      <main className="flex min-h-0 flex-1 flex-col gap-4 p-6 lg:flex-row">
        <div className="relative min-h-72 flex-1 overflow-hidden rounded-xl border border-hair bg-black">
          {/* Both surfaces mounted, one hidden: attaching a track to an element that does not
              exist yet silently does nothing, and that failure looks exactly like black video. */}
          <video
            ref={attachVideo}
            autoPlay
            playsInline
            muted
            aria-label="Interviewer"
            className={rtcVideo ? "size-full object-contain" : "hidden"}
          />
          <canvas
            ref={attachCanvas}
            width={256}
            height={144}
            aria-label="Interviewer"
            className={rtcVideo ? "hidden" : "size-full object-contain"}
            style={{ imageRendering: "pixelated" }}
          />
          {/* Separate element, and the video is muted, so the WebSocket path's Web Audio
              scheduling and the WebRTC track can never both play the same speech. */}
          <audio ref={attachAudio} autoPlay />

          <video
            ref={attachSelf}
            autoPlay
            playsInline
            muted
            aria-label="You"
            className={[
              "absolute right-4 bottom-4 w-1/5 min-w-32 rounded-lg border border-hair-strong",
              "bg-black object-cover shadow-xl",
              publishing ? "" : "hidden",
            ].join(" ")}
            style={{ transform: "scaleX(-1)" }}
          />

          {!connected ? (
            <div className="absolute inset-0 grid place-items-center text-[12.5px] text-ink-mid">
              Connecting…
            </div>
          ) : null}

          {/* The controls a call has, and no others. Mute is a real need — a dog barks, someone
              walks in. Announcing the start and end of your own turns is not. */}
          <div className="absolute bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-2.5 rounded-full border border-hair-strong bg-base/85 p-2 backdrop-blur">
            <CallButton
              label={micOn ? "Mute microphone" : "Unmute microphone"}
              tone={micOn ? "neutral" : "danger"}
              onClick={toggleMic}
              icon={micOn ? <MicIcon /> : <MicOffIcon />}
            />
            <CallButton label="Leave interview" tone="danger" onClick={leave} icon={<EndIcon />} />
          </div>
        </div>

        {showCaptions ? (
          <aside className="flex max-h-[70vh] min-h-0 w-full flex-col rounded-xl border border-hair bg-glass lg:max-h-none lg:w-96">
            <div className="border-b border-hair px-5 py-3">
              <p className="text-[13px] font-medium text-ink">Captions</p>
              <p className="mt-0.5 text-[11.5px] text-ink-low">
                What was heard, and what the interviewer said
              </p>
            </div>
            <div ref={scroller} className="flex-1 space-y-2.5 overflow-y-auto px-5 py-4">
              {transcript.length === 0 ? (
                <p className="py-6 text-center text-[12.5px] text-ink-low">Say hello to begin.</p>
              ) : null}
              {transcript.map((line, index) => (
                <div
                  key={`${line.who}-${line.id}-${index}`}
                  className={[
                    "max-w-[90%] rounded-xl border px-3 py-2 text-[12.5px] leading-relaxed",
                    line.who === "candidate"
                      ? line.empty
                        ? // Shown rather than hidden: the interviewer cannot follow up on
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
                        ? "you — not heard"
                        : "you"
                      : "interviewer"}
                  </span>
                  {line.text}
                  {line.interrupted ? " …" : ""}
                </div>
              ))}
            </div>
            {/* Typing stays as an accessibility path and as the fallback when a microphone
                fails — secondary to speaking rather than a peer of it. */}
            <form
              className="flex gap-2 border-t border-hair px-5 py-4"
              onSubmit={(event) => {
                event.preventDefault();
                say(draft);
                setDraft("");
              }}
            >
              <input
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="Or type your answer"
                aria-label="Type your answer"
                className="min-w-0 flex-1 rounded-lg border border-hair-strong bg-base px-3 py-2 text-[13px] text-ink placeholder:text-ink-low"
              />
              <Button type="submit" disabled={!connected}>
                Send
              </Button>
            </form>
          </aside>
        ) : null}
      </main>
    </div>
  );
}

/* ------------------------------------------------------------- call controls */

/**
 * A round icon button, the shape every video call uses.
 *
 * Icon-only, but never label-less: `aria-label` carries the name for a screen reader and `title`
 * shows it on hover, because a bare glyph is only obvious to someone who already knows the
 * convention. The mic is the one control whose *state* must be readable at a glance, so muted is
 * both a different icon and a different colour rather than colour alone — someone who cannot
 * distinguish red from grey still sees the slash.
 *
 * `size-11` rather than something smaller: this is the one thing on the page a nervous candidate
 * may need to hit quickly, and 44px is the smallest comfortable touch target.
 */
function CallButton({
  label,
  icon,
  tone,
  onClick,
}: {
  label: string;
  icon: React.ReactNode;
  tone: "neutral" | "danger";
  onClick: () => void;
}) {
  const style =
    tone === "danger"
      ? "border-bad/40 bg-bad/15 text-bad hover:bg-bad/25"
      : "border-hair-strong bg-glass-raise text-ink hover:border-ink-low hover:bg-glass";
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={`grid size-11 place-items-center rounded-full border transition-colors ${style}`}
    >
      {icon}
    </button>
  );
}

/*
  Inline SVG rather than an icon package. Three icons do not justify a dependency, and inlining
  means they inherit `currentColor` from the button's tone with no configuration.
*/

const STROKE = {
  width: 18,
  height: 18,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
} as const;

function MicIcon() {
  return (
    <svg {...STROKE} aria-hidden="true">
      <path d="M12 3.5a2.75 2.75 0 0 0-2.75 2.75v5a2.75 2.75 0 0 0 5.5 0v-5A2.75 2.75 0 0 0 12 3.5Z" />
      <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
    </svg>
  );
}

function MicOffIcon() {
  return (
    <svg {...STROKE} aria-hidden="true">
      <path d="M9.25 6.25a2.75 2.75 0 0 1 5.5 0v5c0 .35-.07.69-.18 1M12 14.75A2.75 2.75 0 0 1 9.25 12" />
      <path d="M5.5 11a6.5 6.5 0 0 0 10.1 5.4M12 17.5V21" />
      {/* The slash is the whole point — it reads as muted without relying on the colour. */}
      <path d="M4 4l16 16" />
    </svg>
  );
}

function EndIcon() {
  /* A handset rotated down: the universal hang-up, and unmistakably not "call". */
  return (
    <svg {...STROKE} aria-hidden="true">
      <path d="M15.5 12.7c-.6-.35-.86-1.08-.6-1.72l.5-1.24a9.6 9.6 0 0 0-6.8 0l.5 1.24c.26.64 0 1.37-.6 1.72l-1.7 1a1.4 1.4 0 0 1-1.87-.42l-1.2-1.8a1.4 1.4 0 0 1 .2-1.8 12.6 12.6 0 0 1 17.34 0 1.4 1.4 0 0 1 .2 1.8l-1.2 1.8a1.4 1.4 0 0 1-1.87.42Z" />
    </svg>
  );
}
