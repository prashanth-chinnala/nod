"use client";

/**
 * Joining the LiveKit room and attaching the agent's tracks.
 *
 * Kept separate from `useSession` on purpose. The WebSocket hook is the fallback and still owns
 * telemetry, the transcript, the microphone and every control message; this owns media only. Two
 * small hooks with one job each beat one hook with a mode flag, because the failure modes are
 * genuinely different — a WebRTC failure should degrade to the socket, not take the page down.
 *
 * **Why the agent's identity is matched rather than "the first remote participant".** Taking the
 * first works only while there is exactly one, and breaks the moment an observer joins an
 * interview — which is a feature someone will ask for. The server publishes under a fixed
 * identity precisely so this can be explicit.
 *
 * **Autoplay.** A browser will refuse to start audio without a user gesture, and the failure is
 * silent: video plays, the avatar's mouth moves, and nothing is heard. `startAudio()` is called
 * after the join, which is itself inside a click, and the blocked case is reported rather than
 * left as a mystery.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  RoomEvent,
  Track,
  type RemoteTrack,
  type RemoteParticipant,
  type RemoteTrackPublication,
  type Room,
} from "livekit-client";

export type RtcCredentials = {
  available: boolean;
  url?: string;
  room?: string;
  token?: string;
  agent_identity?: string;
  detail?: string;
};

export type RtcState = {
  /** Whether the SFU is configured at all. `false` means fall back to the socket. */
  available: boolean;
  connected: boolean;
  /** Tracks actually attached, so the UI can say "video but no audio" rather than guessing. */
  hasVideo: boolean;
  hasAudio: boolean;
  audioBlocked: boolean;
  error: string | null;
};

const IDLE: RtcState = {
  available: false,
  connected: false,
  hasVideo: false,
  hasAudio: false,
  audioBlocked: false,
  error: null,
};

export function useRtc(apiBase: string, sessionId: string) {
  const [state, setState] = useState<RtcState>(IDLE);
  const roomRef = useRef<Room | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const attachVideo = useCallback((node: HTMLVideoElement | null) => {
    videoRef.current = node;
  }, []);
  const attachAudio = useCallback((node: HTMLAudioElement | null) => {
    audioRef.current = node;
  }, []);

  const join = useCallback(async () => {
    if (roomRef.current) return;
    setState({ ...IDLE, available: true });

    let credentials: RtcCredentials;
    try {
      const response = await fetch(`${apiBase}/sessions/${sessionId}/rtc`, {
        cache: "no-store",
      });
      credentials = (await response.json()) as RtcCredentials;
    } catch {
      setState({ ...IDLE, error: "could not reach the runtime for RTC credentials" });
      return;
    }

    if (!credentials.available || !credentials.url || !credentials.token) {
      // Not an error. WebRTC is opt-in, and a deployment without an SFU is expected to run on
      // the WebSocket path — so this reports unavailability and the caller falls back.
      setState({ ...IDLE, available: false, error: credentials.detail ?? null });
      return;
    }

    const { Room: LiveKitRoom } = await import("livekit-client");
    const room = new LiveKitRoom({ adaptiveStream: true, dynacast: true });
    roomRef.current = room;

    const wanted = credentials.agent_identity ?? "avatar-agent";

    const attach = (
      track: RemoteTrack,
      _publication: RemoteTrackPublication,
      participant: RemoteParticipant,
    ) => {
      if (participant.identity !== wanted) return;
      if (track.kind === Track.Kind.Video && videoRef.current) {
        track.attach(videoRef.current);
        setState((s) => ({ ...s, hasVideo: true }));
      }
      if (track.kind === Track.Kind.Audio && audioRef.current) {
        track.attach(audioRef.current);
        setState((s) => ({ ...s, hasAudio: true }));
      }
    };

    room
      .on(RoomEvent.TrackSubscribed, attach)
      .on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
        track.detach();
        setState((s) => ({
          ...s,
          hasVideo: track.kind === Track.Kind.Video ? false : s.hasVideo,
          hasAudio: track.kind === Track.Kind.Audio ? false : s.hasAudio,
        }));
      })
      .on(RoomEvent.Disconnected, () =>
        setState((s) => ({ ...s, connected: false, hasVideo: false, hasAudio: false })),
      )
      // Reported rather than swallowed: video plays, the mouth moves, and nothing is heard —
      // which reads as broken audio rather than as a browser policy the user can satisfy.
      .on(RoomEvent.AudioPlaybackStatusChanged, () =>
        setState((s) => ({ ...s, audioBlocked: !room.canPlaybackAudio })),
      );

    try {
      await room.connect(credentials.url, credentials.token);
      await room.startAudio();
      setState((s) => ({ ...s, available: true, connected: true }));
    } catch (cause) {
      setState({
        ...IDLE,
        available: true,
        error: cause instanceof Error ? cause.message : "could not join the room",
      });
      roomRef.current = null;
    }
  }, [apiBase, sessionId]);

  const leave = useCallback(async () => {
    const room = roomRef.current;
    roomRef.current = null;
    if (room) await room.disconnect();
    setState(IDLE);
  }, []);

  /** Retry playback after a gesture, for the case the browser blocked it. */
  const unblockAudio = useCallback(async () => {
    const room = roomRef.current;
    if (!room) return;
    await room.startAudio();
    setState((s) => ({ ...s, audioBlocked: !room.canPlaybackAudio }));
  }, []);

  useEffect(
    () => () => {
      void roomRef.current?.disconnect();
      roomRef.current = null;
    },
    [],
  );

  return { ...state, join, leave, unblockAudio, attachVideo, attachAudio };
}
