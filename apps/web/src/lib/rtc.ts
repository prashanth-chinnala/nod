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
  Room,
  RoomEvent,
  Track,
  type RemoteTrack,
  type RemoteParticipant,
  type RemoteTrackPublication,
} from "livekit-client";

/**
 * `requestVideoFrameCallback`, which is not in TypeScript's DOM lib yet.
 *
 * Declared narrowly rather than cast to `any` at the call site, so the one place that knows the
 * shape is the one place that documents it. Optional on the element type because Firefox has not
 * shipped it — the caller checks before using it and simply reports nothing where it is absent,
 * which is the honest degradation: a missing measurement rather than a wrong one.
 */
type FrameCallbackMetadata = { presentedFrames: number; presentationTime?: number };
type VideoWithFrameCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (
    callback: (now: number, metadata: FrameCallbackMetadata) => void,
  ) => number;
};

export type RtcCredentials = {
  available: boolean;
  url?: string;
  room?: string;
  token?: string;
  agent_identity?: string;
  detail?: string;
};

export type RtcState = {
  /** Whether the candidate's own camera and mic are published to the room. */
  publishing: boolean;
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
  publishing: false,
  available: false,
  connected: false,
  hasVideo: false,
  hasAudio: false,
  audioBlocked: false,
  error: null,
};

export function useRtc(
  apiBase: string,
  sessionId: string,
  /**
   * Called when a turn's first frame has been presented, with the epoch it belongs to.
   *
   * A callback rather than a returned value because this closes a measurement on the server: the
   * caller forwards it as `first_paint` over the WebSocket the session already owns, keeping one
   * reporting path rather than teaching the data channel to talk back.
   */
  onFirstPaint?: (epoch: number) => void,
) {
  const [state, setState] = useState<RtcState>(IDLE);
  const roomRef = useRef<Room | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  // Kept in a ref, synced in an effect rather than assigned during render -- React's compiler
  // lint rejects the latter, and it is right to: a ref written during render is not a render
  // output, so a re-render that is thrown away still mutates it. The ref exists so that a caller
  // passing an inline closure does not tear down and rebuild the room on every render.
  const paintRef = useRef(onFirstPaint);
  useEffect(() => {
    paintRef.current = onFirstPaint;
  }, [onFirstPaint]);
  /**
   * The epoch the server has announced but whose frame has not yet been seen presented.
   *
   * One slot rather than a queue: turns do not overlap, and a barge-in replaces the pending epoch
   * rather than queueing behind it — measuring the paint of a turn that was already cancelled
   * would be worse than measuring nothing.
   */
  const pendingEpoch = useRef<number | null>(null);
  const reportedEpoch = useRef(-1);

  const attachVideo = useCallback((node: HTMLVideoElement | null) => {
    videoRef.current = node;
  }, []);
  const attachAudio = useCallback((node: HTMLAudioElement | null) => {
    audioRef.current = node;
  }, []);
  const selfRef = useRef<HTMLVideoElement | null>(null);
  const attachSelf = useCallback((node: HTMLVideoElement | null) => {
    selfRef.current = node;
  }, []);

  /**
   * Report the pending epoch once the video element presents its next frame.
   *
   * **What this measures, precisely.** The server announces an epoch on the data channel
   * immediately before pushing that turn's first frame into the video source. `rVFC` then fires
   * when the compositor has presented a frame. So the figure closed here is *the time from the
   * server marking a turn's first frame to the client presenting its next frame* — which includes
   * encode, the SFU, decode, the jitter buffer and a compositor tick, and is what the candidate
   * actually waited for.
   *
   * **What it is not.** The marker and the frame travel different paths — SCTP data and an RTP
   * media stream — so nothing guarantees the frame `rVFC` reports is the frame the marker
   * described. In practice the marker is a few dozen bytes on a lower-latency path and arrives
   * first, so the next presented frame is the right one; under loss or congestion it could be a
   * frame from the tail of the previous turn, which would understate the figure. It is an
   * attribution by adjacency, not by identity, and the honest reading is "close, with a known bias
   * toward optimism". Exact attribution needs the epoch inside the frame — an SEI message or
   * `RTCEncodedVideoFrame` metadata — which means an encoded-transform pipeline this does not have.
   *
   * One callback per marker rather than a standing loop: `rVFC` is not repeating, and re-arming it
   * every frame would run this at 25fps to answer a question asked once a turn.
   */
  const watchForPaint = useCallback(() => {
    const element = videoRef.current as VideoWithFrameCallback | null;
    const epoch = pendingEpoch.current;
    if (!element || epoch === null) return;
    // Absent on Firefox. Reporting nothing is the right degradation -- a missing measurement is
    // recoverable, a fabricated one poisons the latency table this product is judged on.
    if (typeof element.requestVideoFrameCallback !== "function") return;

    element.requestVideoFrameCallback(() => {
      // Monotonic guard, matching the orchestrator's own rule that there is only one first frame
      // per turn. Without it a late marker for an old epoch could report a paint out of order.
      if (epoch <= reportedEpoch.current) return;
      reportedEpoch.current = epoch;
      pendingEpoch.current = null;
      paintRef.current?.(epoch);
    });
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

    // Statically imported. A dynamic `import()` here shared the module with the static
    // `RoomEvent`/`Track` imports in principle, and in practice was the only place two
    // instances could diverge -- not worth the code-splitting it bought for one small library.
    const room = new Room({ adaptiveStream: true, dynacast: true });
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
      .on(RoomEvent.DataReceived, (payload: Uint8Array, _p, _kind, topic?: string) => {
        // The epoch marker the transport sends on the first frame of each turn. Without it a
        // WebRTC video track is anonymous pixels and no painted frame can be attributed to a
        // turn, which is why first-frame latency was unmeasurable on this path.
        if (topic !== undefined && topic !== "control") return;
        try {
          const message = JSON.parse(new TextDecoder().decode(payload)) as {
            type?: string;
            epoch?: number;
          };
          if (message.type === "frame_epoch" && typeof message.epoch === "number") {
            pendingEpoch.current = message.epoch;
            watchForPaint();
          }
        } catch {
          /* a malformed control message must not take the call down */
        }
      })
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

      // Publish the candidate's camera and mic. Two reasons, and the second is the one that
      // matters: it makes the call two-way rather than the candidate watching a broadcast, and
      // it puts their tracks in the room so egress records *both* sides. A recording with only
      // the avatar in it is not a reviewable artifact of an interview.
      //
      // Failure here is not fatal. A candidate who declines the camera should still be
      // interviewed -- audio and the typed path both still work -- so this degrades rather than
      // aborting the join.
      try {
        await room.localParticipant.enableCameraAndMicrophone();
        const camera = room.localParticipant.getTrackPublication(Track.Source.Camera);
        if (camera?.track && selfRef.current) camera.track.attach(selfRef.current);
        setState((s) => ({ ...s, publishing: true }));
      } catch {
        setState((s) => ({ ...s, publishing: false }));
      }
    } catch (cause) {
      setState({
        ...IDLE,
        available: true,
        error: cause instanceof Error ? cause.message : "could not join the room",
      });
      roomRef.current = null;
    }
  }, [apiBase, sessionId, watchForPaint]);

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

  return { ...state, join, leave, unblockAudio, attachVideo, attachAudio, attachSelf };
}
