"use client";

/**
 * Voices — a recording of someone speaking, which an agent can be given to sound like.
 *
 * **Why auditioning is the primary action here, not an afterthought.** A face can be judged from a
 * thumbnail; a voice cannot be judged from a filename, a duration, or a sample rate. The only way
 * to know whether a clone is usable is to listen to it, and the alternative to doing that here is
 * discovering it during an interview. So every row has a play button and the page is built around
 * it.
 *
 * **Why the requirements are stated before the file picker.** Recording someone is not free, and
 * telling an operator the 5s minimum after they have already uploaded 3s wastes their time twice.
 * The same reasoning as the Faces screen.
 *
 * **What the warnings are for.** The runtime distinguishes a refusal from a caution: too short is
 * rejected, but 8 seconds, or 8kHz, or two channels are accepted *and* disappointing. Those
 * cautions are shown as prominently as the success, because they are still cheap to act on at this
 * moment and expensive to discover later.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  Button,
  Card,
  CardHeader,
  Cell,
  Chip,
  Empty,
  Field,
  Input,
  Page,
  Row,
  Table,
} from "@/components/ui";

const API = "http://127.0.0.1:8000/voices";

/** Mirrors `avatar.media`'s bounds, so the guidance cannot drift from what is enforced. */
const MIN_SECONDS = 5;
const RECOMMENDED_SECONDS = 15;
const MAX_SECONDS = 120;

type Voice = {
  id: string;
  name: string;
  duration_seconds: number | null;
  sample_rate: number | null;
  channels: number | null;
  status: string;
  failure_reason: string | null;
  updated_at: string;
  warnings?: string[];
};

const HEAD = ["Name", "Length", "Audio", "Status", "Updated (UTC)", ""] as const;

function stamp(iso: string): string {
  return iso ? iso.replace("T", " ").replace(/(\+00:00|Z)$/, "").slice(0, 16) : "—";
}

export default function VoicesPage() {
  const [voices, setVoices] = useState<Voice[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    fetch(API, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`the runtime answered ${response.status}`);
        return (await response.json()) as Voice[];
      })
      .then((listed) => {
        setVoices(listed);
        setError(null);
      })
      .catch((cause: unknown) => {
        setVoices(null);
        setError(cause instanceof Error ? cause.message : "the runtime is unreachable");
      });
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <Page
      title="Voices"
      lede={`A voice is a recording of someone speaking. Cloning is zero-shot — there is no enrollment step, so a voice is usable the moment it validates. ${RECOMMENDED_SECONDS}s or more of ordinary speech captures how a person speaks rather than only how they sound.`}
    >
      <VoiceCreate onCreated={load} />

      <Card>
        <CardHeader
          title="Cloned voices"
          hint="Audition before attaching one. A voice cannot be judged from its duration, and the alternative to listening here is finding out during an interview."
        />
        {error ? (
          <Empty title="Could not reach the runtime" action={<Button onClick={load}>Retry</Button>}>
            {error}. The console reads voices from the runtime on 127.0.0.1:8000.
          </Empty>
        ) : voices === null ? (
          <div className="px-5 py-14 text-center text-[12.5px] text-ink-mid">Loading voices…</div>
        ) : voices.length === 0 ? (
          <Empty title="No voices yet">
            Upload a recording of the person an interviewer should sound like. Everything else about
            the persona — the face, the questions — is configured separately, so a voice can be
            reused or replaced on its own.
          </Empty>
        ) : (
          <Table head={HEAD}>
            {voices.map((voice) => (
              <VoiceRow key={voice.id} voice={voice} onChanged={load} />
            ))}
          </Table>
        )}
      </Card>
    </Page>
  );
}

/* --------------------------------------------------------------------- a row */

function VoiceRow({ voice, onChanged }: { voice: Voice; onChanged: () => void }) {
  const [busy, setBusy] = useState<"audition" | "delete" | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const audio = useRef<HTMLAudioElement | null>(null);

  const audition = useCallback(async () => {
    setBusy("audition");
    setProblem(null);
    try {
      const response = await fetch(`${API}/${voice.id}/audition`, { method: "POST" });
      if (!response.ok) {
        const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(payload?.detail ?? `the runtime answered ${response.status}`);
      }
      // An object URL rather than a data URL: the WAV is hundreds of kilobytes, and base64 would
      // inflate it by a third to no benefit. Revoked when playback ends so the blob is not held
      // for the life of the page.
      const url = URL.createObjectURL(await response.blob());
      const player = new Audio(url);
      audio.current = player;
      player.onended = () => URL.revokeObjectURL(url);
      await player.play();
    } catch (cause) {
      setProblem(cause instanceof Error ? cause.message : "could not audition this voice");
    } finally {
      setBusy(null);
    }
  }, [voice.id]);

  const remove = useCallback(async () => {
    setBusy("delete");
    try {
      const response = await fetch(`${API}/${voice.id}`, { method: "DELETE" });
      if (!response.ok) throw new Error(`the runtime answered ${response.status}`);
      onChanged();
    } catch (cause) {
      setProblem(cause instanceof Error ? cause.message : "could not delete this voice");
      setBusy(null);
    }
  }, [voice.id, onChanged]);

  return (
    <Row>
      <Cell>
        {voice.name}
        {problem ? <p className="mt-1 text-[11.5px] text-bad">{problem}</p> : null}
      </Cell>
      <Cell dim mono>
        {voice.duration_seconds ? `${voice.duration_seconds.toFixed(0)}s` : "—"}
      </Cell>
      <Cell dim mono>
        {voice.sample_rate ? `${(voice.sample_rate / 1000).toFixed(0)}kHz` : "—"}
        {voice.channels && voice.channels > 1 ? ` · ${voice.channels}ch` : ""}
      </Cell>
      <Cell>
        <Chip status={voice.status === "ready" ? "ok" : "warn"}>{voice.status}</Chip>
      </Cell>
      <Cell dim mono>
        {stamp(voice.updated_at)}
      </Cell>
      <Cell right>
        <div className="flex justify-end gap-2">
          <Button disabled={busy !== null} onClick={() => void audition()}>
            {busy === "audition" ? "Speaking…" : "Audition"}
          </Button>
          <Button disabled={busy !== null} onClick={() => void remove()}>
            {busy === "delete" ? "Deleting…" : "Delete"}
          </Button>
        </div>
      </Cell>
    </Row>
  );
}

/* ------------------------------------------------------------------- creating */

function VoiceCreate({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [created, setCreated] = useState<Voice | null>(null);

  const submit = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setProblem(null);
    try {
      // FormData, so the browser sets the multipart boundary. Setting Content-Type by hand is the
      // classic way to make a multipart upload fail with an unhelpful 422.
      const body = new FormData();
      body.append("name", name.trim() || file.name);
      body.append("file", file);
      const response = await fetch(`${API}/upload`, { method: "POST", body });
      const payload = (await response.json().catch(() => null)) as
        | (Voice & { detail?: string })
        | null;
      if (!response.ok) {
        // The runtime's own sentence, verbatim. Every rejection names something about the file the
        // person who chose it can act on, and paraphrasing loses the number they need.
        throw new Error(payload?.detail ?? `the runtime rejected this (${response.status})`);
      }
      setCreated(payload as Voice);
      setFile(null);
      setName("");
      onCreated();
    } catch (cause) {
      setProblem(
        cause instanceof Error ? cause.message : "could not reach the runtime on 127.0.0.1:8000",
      );
    } finally {
      setBusy(false);
    }
  }, [file, name, onCreated]);

  if (created) {
    const warnings = created.warnings ?? [];
    return (
      <Card>
        <CardHeader
          title="Voice added"
          hint="Attach it to an agent below the Agents table. There is no preparation step — cloning is zero-shot."
          action={<Button onClick={() => setCreated(null)}>Add another</Button>}
        />
        <div className="space-y-4 px-5 py-5">
          <div className="flex flex-wrap items-center gap-2.5">
            <p className="text-[13.5px] font-medium text-ink">{created.name}</p>
            <Chip status="ok">
              {created.duration_seconds ? `${created.duration_seconds.toFixed(0)}s` : "—"}
            </Chip>
            <Chip status="neutral">
              {created.sample_rate ? `${(created.sample_rate / 1000).toFixed(0)}kHz` : "—"}
              {created.channels && created.channels > 1 ? ` · ${created.channels}ch` : " · mono"}
            </Chip>
            <span className="font-mono text-[11px] text-ink-low">{created.id}</span>
          </div>
          {warnings.length > 0 ? (
            <div className="rounded-lg border border-warn/40 bg-warn/5 px-4 py-3.5">
              <p className="text-[12px] font-medium text-warn">Worth knowing before you use it</p>
              {warnings.map((warning) => (
                <p key={warning} className="mt-1.5 text-[12px] leading-relaxed text-ink-mid">
                  {warning}
                </p>
              ))}
            </div>
          ) : (
            <p className="text-[12.5px] text-ink-mid">
              No cautions — long enough and clean enough to clone well. Audition it below to hear
              what it actually sounds like.
            </p>
          )}
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        title="Add a voice"
        hint="A recording of one person speaking naturally. It is encoded into a speaker embedding at first use — nothing is trained, and the recording is never sent anywhere."
      />
      <div className="space-y-5 px-5 py-5">
        {/* Before the picker, not after a rejection: recording someone is not free. */}
        <div className="rounded-lg border border-hair-strong bg-glass-raise px-4 py-3.5">
          <p className="text-[11px] font-medium tracking-[0.07em] uppercase text-ink-low">
            What the runtime will accept
          </p>
          <ul className="mt-2 space-y-1 text-[12px] leading-relaxed text-ink-mid">
            <li>
              At least {MIN_SECONDS}s, and {RECOMMENDED_SECONDS}s or more is better — below that
              there is enough to copy how someone sounds but not how they speak
            </li>
            <li>
              Up to {MAX_SECONDS / 60} minutes. Longer is refused rather than truncated: the
              embedding saturates, and extra minutes only slow things down
            </li>
            <li>One speaker, mono, no background music. wav, mp3, m4a, flac, ogg or webm</li>
          </ul>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Name" hint="How it appears when attaching a voice to an agent">
            <Input
              value={name}
              placeholder={file?.name ?? "Interviewer voice"}
              onChange={(event) => setName(event.target.value)}
            />
          </Field>
          <Field
            label="Recording"
            hint={file ? `${(file.size / 1_048_576).toFixed(1)} MB selected` : "Nothing selected"}
          >
            <input
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.flac,.ogg,.webm"
              onChange={(event) => {
                setFile(event.target.files?.[0] ?? null);
                setProblem(null);
              }}
              className="w-full rounded-lg border border-hair-strong bg-base px-3 py-2 text-[12.5px] text-ink-mid file:mr-3 file:rounded-md file:border-0 file:bg-glass-raise file:px-2.5 file:py-1 file:text-[12px] file:text-ink"
            />
          </Field>
        </div>

        {problem ? (
          <div className="rounded-lg border border-bad/40 bg-bad/5 px-4 py-3">
            <p className="text-[12.5px] leading-relaxed text-bad">{problem}</p>
          </div>
        ) : null}

        <div className="flex items-center gap-3">
          <Button variant="primary" disabled={!file || busy} onClick={() => void submit()}>
            {busy ? "Uploading…" : "Add voice"}
          </Button>
          <p className="text-[11.5px] text-ink-low">
            {/* Said plainly, because it is a real person's voice. */}
            The recording is kept and read whenever this voice speaks. Deleting the voice deletes it.
          </p>
        </div>
      </div>
    </Card>
  );
}
