"use client";

/**
 * Speech in and out for the assistant.
 *
 * **Push to talk, not voice activity detection.** The interview path has a whole tested module for
 * deciding when someone has stopped speaking, and none of it belongs here: this is a screen where
 * the operator is also reading and typing, and a detector guessing at end-of-turn produces a
 * question that submits itself mid-sentence. A button that starts and stops is not a simpler version
 * of the right answer — for dictation into a form, it *is* the right answer.
 *
 * **Recording goes to our own service, not to `SpeechRecognition`.** The browser API is Chrome-only
 * and ships audio to Google, which is a third processor of interview-adjacent speech that nobody
 * consented to. Same vendor as the rest of the product, one credential, and it works in Safari.
 *
 * **Playback is an `<audio>` element on a Blob URL.** No Web Audio scheduling, no jitter buffer, no
 * sample-rate agreement. All of that exists on the interview path because audio arrives there while
 * it is still being generated; here the file is complete before it is sent, and reaching for the
 * same apparatus would be copying the shape of a solution without its problem.
 */

import { useCallback, useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_ASSISTANT_BASE ?? "http://127.0.0.1:8100";

export type VoiceStatus = { available: boolean; voice: string | null; detail: string };

/** Whether speech is configured at all, so the UI can omit the microphone rather than fail on it. */
export function useVoiceStatus(): VoiceStatus | null {
  const [status, setStatus] = useState<VoiceStatus | null>(null);
  useEffect(() => {
    fetch(`${API}/voice`, { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : null))
      .then(setStatus)
      .catch(() => setStatus({ available: false, voice: null, detail: "assistant unreachable" }));
  }, []);
  return status;
}

export function useVoice() {
  const [recording, setRecording] = useState(false);
  const [transcribing, setTranscribing] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const audio = useRef<HTMLAudioElement | null>(null);
  const objectUrl = useRef<string | null>(null);

  /** Release the previous Blob URL. Without this every answer leaks one for the tab's lifetime. */
  const releaseUrl = useCallback(() => {
    if (objectUrl.current) {
      URL.revokeObjectURL(objectUrl.current);
      objectUrl.current = null;
    }
  }, []);

  useEffect(
    () => () => {
      recorder.current?.stream.getTracks().forEach((track) => track.stop());
      audio.current?.pause();
      releaseUrl();
    },
    [releaseUrl],
  );

  const startRecording = useCallback(async () => {
    setProblem(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      chunks.current = [];
      // No mimeType requested. Chrome gives webm/opus and Safari gives mp4, and the recorder's own
      // type is forwarded to the service, which forwards it to a transcriber that sniffs the
      // container. Asking for a specific format is what breaks one of the two browsers.
      const media = new MediaRecorder(stream);
      media.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.current.push(event.data);
      };
      recorder.current = media;
      media.start();
      setRecording(true);
    } catch {
      setProblem("no microphone, or permission was refused");
    }
  }, []);

  /**
   * Stop, upload, and resolve to the transcript.
   *
   * Resolves rather than setting state, so the caller decides what a transcript means — this hook
   * does not know whether it should land in the input box or be sent immediately, and the two are
   * different products.
   */
  const stopRecording = useCallback(async (): Promise<string> => {
    const media = recorder.current;
    if (!media) return "";
    setRecording(false);
    setTranscribing(true);

    const blob = await new Promise<Blob>((resolve) => {
      // `onstop` fires after the final `ondataavailable`, which is the only ordering that
      // guarantees the last chunk is included -- reading `chunks` on the stop *call* drops it.
      media.onstop = () => resolve(new Blob(chunks.current, { type: media.mimeType }));
      media.stop();
    });
    media.stream.getTracks().forEach((track) => track.stop());
    recorder.current = null;

    try {
      const response = await fetch(`${API}/transcribe`, {
        method: "POST",
        headers: { "Content-Type": blob.type || "audio/webm" },
        body: blob,
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(body?.detail ?? `transcription failed (${response.status})`);
      }
      const { text } = (await response.json()) as { text: string };
      if (!text) setProblem("nothing was heard");
      return text;
    } catch (cause) {
      setProblem(cause instanceof Error ? cause.message : "transcription failed");
      return "";
    } finally {
      setTranscribing(false);
    }
  }, []);

  const cancelRecording = useCallback(() => {
    recorder.current?.stream.getTracks().forEach((track) => track.stop());
    recorder.current = null;
    chunks.current = [];
    setRecording(false);
  }, []);

  /** Speak one answer. Any previous playback is stopped first, so two answers never overlap. */
  const say = useCallback(
    async (text: string) => {
      if (!text.trim()) return;
      setProblem(null);
      audio.current?.pause();
      releaseUrl();
      setSpeaking(true);
      try {
        const response = await fetch(`${API}/speak`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (!response.ok) {
          const body = (await response.json().catch(() => null)) as { detail?: string } | null;
          throw new Error(body?.detail ?? `synthesis failed (${response.status})`);
        }
        const url = URL.createObjectURL(await response.blob());
        objectUrl.current = url;
        // A fresh element per utterance, fully configured before it is stored. Reusing one and
        // reassigning `src` mutates a value the cleanup effect already reads, which React's
        // compiler lint rejects -- and it is right that this is the cleaner shape: the previous
        // element is paused and dropped rather than half-reset, so a stale `onended` from an
        // abandoned answer cannot clear `speaking` for the current one.
        const element = new Audio(url);
        element.onended = () => setSpeaking(false);
        audio.current?.pause();
        audio.current = element;
        await element.play();
      } catch (cause) {
        setSpeaking(false);
        setProblem(cause instanceof Error ? cause.message : "could not play the answer");
      }
    },
    [releaseUrl],
  );

  const stopSpeaking = useCallback(() => {
    audio.current?.pause();
    setSpeaking(false);
  }, []);

  return {
    recording,
    transcribing,
    speaking,
    problem,
    startRecording,
    stopRecording,
    cancelRecording,
    say,
    stopSpeaking,
  };
}
