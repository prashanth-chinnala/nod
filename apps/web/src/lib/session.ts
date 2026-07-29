/**
 * The live session, as a hook.
 *
 * Originally a port of a plain-JS client the API served itself; that page is gone, and this
 * is now the only client. The API serves JSON and a WebSocket, nothing else — which is the
 * right split, and it means the runtime no longer has an opinion about how it is rendered.
 *
 * Three things here are load-bearing and were each a bug before they were code:
 *
 * **The jitter buffer.** Audio is scheduled AUDIO_LEAD_MS ahead of the playhead, not at
 * `currentTime`. Locally chunks arrive far faster than real time so the cursor always sits in
 * the future and the absence of a buffer is invisible; through a tunnel a chunk arrives after
 * its slot has passed, which is an audible gap, and clamping to "now" then pins the cursor to
 * packet-arrival time so every later late packet gaps again.
 *
 * **Format sniffing.** The wire carries whatever the renderer emits — PNG from the
 * placeholder, JPEG from a photographic model. Reading the format off the bytes means that
 * swap needs no protocol change and no coordinated deploy.
 *
 * **Reporting playback back.** `audio_played` is the only evidence the server's history
 * truncation trusts, and `first_paint` is the only way end-to-end latency can be measured to
 * the screen rather than to the socket write. A client that renders without reporting looks
 * identical and makes both numbers wrong.
 */

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const HEADER_BYTES = 13; // kind:u8, pts:u32, epoch:u32, length:u32 — big-endian
const KIND_VIDEO = 1;

export const AUDIO_LEAD_MS = 150;
/**
 * How far ahead of the playhead to schedule the first chunk of a run.
 *
 * Latency paid deliberately: it is added to what the candidate hears, so a session recorded
 * over a tunnel with this cushion is not comparable to a local one. 150ms absorbs a
 * cross-region round trip and stays under the ~200ms where a reply starts to feel delayed.
 */

export type SessionState =
  | "INITIALIZING"
  | "IDLE"
  | "LISTENING"
  | "THINKING"
  | "SPEAKING"
  | "CANCELLING"
  | "CLOSED";

export type Utterance = {
  id: number;
  who: "candidate" | "interviewer";
  text: string;
  /** A turn the transcriber produced no words for. Shown, not hidden — see the hook body. */
  empty?: boolean;
  /** Cut short by a barge-in. The transcript should agree with the truncated history. */
  interrupted?: boolean;
};

export type Metrics = {
  fps: number;
  epoch: number;
  framesDrawn: number;
  framesDropped: number;
  audioAckedMs: number;
  underruns: number;
  speechProbability: number | null;
  micKb: number;
  stages: Record<string, number>;
};

export type Hello = {
  renderer: string;
  llm: string;
  tts: string;
  stt: string;
  vad: string;
  sample_rate: number;
  target_fps: number;
  frame_width: number;
  frame_height: number;
  end_of_turn_silence_ms: number;
};

const EMPTY_METRICS: Metrics = {
  fps: 0,
  epoch: 0,
  framesDrawn: 0,
  framesDropped: 0,
  audioAckedMs: 0,
  underruns: 0,
  speechProbability: null,
  micKb: 0,
  stages: {},
};

/** PNG, JPEG, or BMP, read off the bytes rather than trusting a declaration. */
function sniffImageType(bytes: Uint8Array): string {
  if (bytes.length > 8 && bytes[0] === 0x89 && bytes[1] === 0x50) return "image/png";
  if (bytes.length > 3 && bytes[0] === 0xff && bytes[1] === 0xd8) return "image/jpeg";
  if (bytes.length > 2 && bytes[0] === 0x42 && bytes[1] === 0x4d) return "image/bmp";
  return "";
}

export function useSession(apiBase: string, sessionId?: string) {
  const [state, setState] = useState<SessionState>("INITIALIZING");
  const [hello, setHello] = useState<Hello | null>(null);
  const [connected, setConnected] = useState(false);
  const [transcript, setTranscript] = useState<Utterance[]>([]);
  const [metrics, setMetrics] = useState<Metrics>(EMPTY_METRICS);
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const audioRef = useRef<AudioContext | null>(null);
  const cursorRef = useRef(0); // next free moment on the audio clock; 0 = no run in progress
  const scheduledRef = useRef<Array<{ src: AudioBufferSourceNode; at: number; dur: number }>>([]);
  const drawingRef = useRef(false);
  const paintedEpochRef = useRef(0);
  const fpsWindowRef = useRef<number[]>([]);
  const uid = useRef(0);
  const micRef = useRef<{ stream: MediaStream; node: ScriptProcessorNode } | null>(null);

  const send = useCallback((payload: unknown) => {
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(payload));
  }, []);

  /** Append, or grow the interviewer's current bubble — sentences stream in one at a time. */
  const pushSaid = useCallback((text: string, epoch: number) => {
    setTranscript((prev) => {
      const last = prev.at(-1);
      if (last && last.who === "interviewer" && last.id === epoch && !last.interrupted) {
        return [...prev.slice(0, -1), { ...last, text: last.text + text }];
      }
      return [...prev, { id: epoch, who: "interviewer", text }];
    });
  }, []);

  const flushAudio = useCallback(() => {
    for (const entry of scheduledRef.current) {
      try {
        entry.src.stop();
      } catch {
        /* already finished */
      }
    }
    scheduledRef.current = [];
    // 0, not currentTime: a barge-in abandons the run deliberately, so the next chunk starts
    // a new one and must re-arm the lead without being counted an underrun.
    cursorRef.current = 0;
    setTranscript((prev) => {
      const last = prev.at(-1);
      if (!last || last.who !== "interviewer") return prev;
      return [...prev.slice(0, -1), { ...last, interrupted: true }];
    });
  }, []);

  const playAudio = useCallback((pcm: Uint8Array, epoch: number) => {
    const ctx = audioRef.current;
    if (!ctx || !hello) return;
    const samples = pcm.byteLength / 2;
    if (!samples) return;

    const buffer = ctx.createBuffer(1, samples, hello.sample_rate);
    const channel = buffer.getChannelData(0);
    const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    for (let i = 0; i < samples; i++) channel[i] = view.getInt16(i * 2, true) / 32768;

    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);

    const now = ctx.currentTime;
    let startAt: number;
    if (cursorRef.current > now) {
      startAt = cursorRef.current; // continuing an unbroken run
    } else {
      startAt = now + AUDIO_LEAD_MS / 1000;
      if (cursorRef.current > 0) {
        // Fell behind. Counted rather than hidden: "it sounded choppy" is not a diagnosis.
        setMetrics((m) => ({ ...m, underruns: m.underruns + 1 }));
      }
    }
    cursorRef.current = startAt + buffer.duration;

    const entry = { src, at: startAt, dur: buffer.duration };
    scheduledRef.current.push(entry);
    src.onended = () => {
      scheduledRef.current = scheduledRef.current.filter((e) => e !== entry);
      // How much actually reached the candidate's ears — full duration on a natural end,
      // however far the clock got on a barge-in stop. The only evidence the server's history
      // truncation trusts.
      const played = Math.max(0, Math.min(entry.dur, ctx.currentTime - entry.at));
      const ms = Math.round(played * 1000);
      if (ms > 0) {
        setMetrics((m) => ({ ...m, audioAckedMs: m.audioAckedMs + ms }));
        send({ type: "audio_played", ms, epoch });
      }
    };
    src.start(startAt);
  }, [hello, send]);

  const drawFrame = useCallback(async (payload: Uint8Array, epoch: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // A frame arriving mid-decode is dropped, not queued: queueing trades a dropped frame
    // for unbounded latency growth, which is the wrong direction for a conversation.
    if (drawingRef.current) {
      setMetrics((m) => ({ ...m, framesDropped: m.framesDropped + 1 }));
      return;
    }
    drawingRef.current = true;
    try {
      const blob = new Blob([payload as BlobPart], { type: sniffImageType(payload) });
      const bitmap = await createImageBitmap(blob);
      if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
      }
      canvas.getContext("2d", { alpha: false })?.drawImage(bitmap, 0, 0);
      bitmap.close();

      const now = performance.now();
      fpsWindowRef.current = [...fpsWindowRef.current, now].filter((t) => now - t < 1000);
      setMetrics((m) => ({
        ...m,
        framesDrawn: m.framesDrawn + 1,
        fps: fpsWindowRef.current.length,
      }));

      if (epoch > paintedEpochRef.current) {
        paintedEpochRef.current = epoch;
        // Closes the end-to-end measurement. The server cannot see encode, socket, decode,
        // or paint; this is the only report of that tail.
        send({ type: "first_paint", epoch });
      }
    } catch {
      /* a malformed frame must not tear down the session */
    } finally {
      drawingRef.current = false;
    }
  }, [send]);

  const handleControl = useCallback((message: Record<string, unknown>) => {
    switch (message.type) {
      case "hello":
        setHello(message as unknown as Hello);
        break;
      case "flush_audio":
        flushAudio();
        break;
      case "stats":
        setMetrics((m) => ({
          ...m,
          epoch: Number(message.epoch ?? m.epoch),
          speechProbability:
            message.speech_probability === undefined
              ? m.speechProbability
              : Number(message.speech_probability),
        }));
        break;
      case "error":
        setError(String(message.detail ?? "protocol error"));
        break;
      case "event": {
        const event = String(message.event);
        if (event === "state_change") setState(String(message.to) as SessionState);
        if (event === "latency") {
          setMetrics((m) => ({
            ...m,
            stages: { ...m.stages, [String(message.stage)]: Number(message.ms) },
          }));
        }
        if (event === "heard") {
          const transcribed = Boolean(message.transcribed);
          setTranscript((prev) => [
            ...prev,
            {
              id: uid.current++,
              who: "candidate",
              text: String(message.text),
              empty: !transcribed,
            },
          ]);
        }
        if (event === "said") pushSaid(String(message.text), Number(message.epoch));
        break;
      }
    }
  }, [flushAudio, pushSaid]);

  const connect = useCallback(async () => {
    if (socketRef.current) return;
    setError(null);
    setTranscript([]);
    setMetrics(EMPTY_METRICS);
    paintedEpochRef.current = 0;
    cursorRef.current = 0;

    // Created at the server's rate so neither playback nor capture needs resampling: audio
    // arrives at 16kHz and the voice-activity detector expects 16kHz.
    const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = audioRef.current ?? new Ctx({ sampleRate: 16000 });
    audioRef.current = ctx;
    await ctx.resume();

    // The session id travels as a query parameter, which is how the candidate's link reaches
    // the runtime: the record it names carries the agent, and the agent carries everything
    // else. Without it the runtime falls back to its environment, which is fine for the
    // prototype and wrong for a real interview.
    const query = sessionId ? `?session=${encodeURIComponent(sessionId)}` : "";
    const url = apiBase.replace(/^http/, "ws") + "/session" + query;
    const socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    socket.onopen = () => setConnected(true);
    socket.onclose = () => {
      setConnected(false);
      setState("CLOSED");
      socketRef.current = null;
    };
    socket.onerror = () =>
      setError(`cannot reach the runtime at ${apiBase}. Start it with: cd apps/api && python -m uvicorn avatar.server:app`);
    socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        handleControl(JSON.parse(event.data) as Record<string, unknown>);
        return;
      }
      const frame = new Uint8Array(event.data as ArrayBuffer);
      if (frame.byteLength < HEADER_BYTES) return;
      const header = new DataView(frame.buffer, frame.byteOffset, HEADER_BYTES);
      const kind = header.getUint8(0);
      const epoch = header.getUint32(5);
      const payload = frame.subarray(HEADER_BYTES);
      if (kind === KIND_VIDEO) void drawFrame(payload, epoch);
      else playAudio(payload, epoch);
    };
  }, [apiBase, sessionId, drawFrame, handleControl, playAudio]);

  const disconnect = useCallback(() => {
    socketRef.current?.close();
    socketRef.current = null;
    flushAudio();
  }, [flushAudio]);

  /** Type an answer. Sends the same pair the microphone produces, so it travels the identical
   *  route through the state machine — including cancelling a turn already in progress, which
   *  makes sending mid-answer a genuine barge-in rather than a queued message. */
  const say = useCallback((text: string) => {
    if (!text.trim()) return;
    send({ type: "speech_start" });
    send({ type: "end_of_turn", transcript: text.trim() });
  }, [send]);

  const startMic = useCallback(async () => {
    if (micRef.current) return;
    const ctx = audioRef.current;
    if (!ctx) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    const source = ctx.createMediaStreamSource(stream);
    // ScriptProcessor is deprecated in favour of AudioWorklet, which needs either a separate
    // module file or a Blob URL. Kept deliberately: it works everywhere, and the capture path
    // is not where this product's latency lives.
    const node = ctx.createScriptProcessor(1024, 1, 1);
    node.onaudioprocess = (event) => {
      const input = event.inputBuffer.getChannelData(0);
      const pcm = new Int16Array(input.length);
      for (let i = 0; i < input.length; i++) {
        pcm[i] = Math.max(-1, Math.min(1, input[i])) * 32767;
      }
      const socket = socketRef.current;
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(pcm.buffer);
        setMetrics((m) => ({ ...m, micKb: m.micKb + pcm.byteLength / 1024 }));
      }
    };
    source.connect(node);
    node.connect(ctx.destination);
    micRef.current = { stream, node };
  }, []);

  const stopMic = useCallback(() => {
    micRef.current?.node.disconnect();
    micRef.current?.stream.getTracks().forEach((track) => track.stop());
    micRef.current = null;
  }, []);

  useEffect(() => () => {
    stopMic();
    socketRef.current?.close();
  }, [stopMic]);

  /**
   * Attach the canvas. A callback ref rather than a returned ref object: handing a ref out of
   * a hook means every property read on the returned object counts as accessing a ref during
   * render, which React's rules flag and which genuinely can miss updates.
   */
  const attachCanvas = useCallback((node: HTMLCanvasElement | null) => {
    canvasRef.current = node;
  }, []);

  return {
    state, hello, connected, transcript, metrics, error,
    attachCanvas, connect, disconnect, say, send, startMic, stopMic,
  };
}
