"use client";

/**
 * Creating a face: choose a method, upload, see what the runtime made of it.
 *
 * **Why two methods and not one file picker.** Video and image are genuinely different inputs to a
 * reference-driven renderer, not two encodings of the same thing. The final frames *are* the
 * reference frames with the mouth repainted, so a video lends its head motion, blinks and posture
 * to every second of every session — and a photograph has none to lend. Presenting one "upload"
 * box would hide a quality decision behind a file dialog, so the trade-off is on the cards where it
 * is being made.
 *
 * **Why the warnings are shown as prominently as success.** The runtime distinguishes a refusal
 * from a caution: too small or too short is rejected, but a 12-second clip or a still image is
 * accepted *and* disappointing. Those cautions are the whole reason this screen exists rather than
 * a bare file input — an operator who uploads a photo should learn immediately that the persona
 * will hold one pose, not discover it in front of a candidate.
 *
 * **Why the reference requirements are stated before the picker, not after a rejection.** Recording
 * a clip is not free. Telling someone the minimum after they have already uploaded a 3-second one
 * wastes their time twice.
 */

import { useCallback, useRef, useState } from "react";

import { Button, Card, CardHeader, Chip, Field, Input } from "@/components/ui";

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

/** Mirrors `avatar.media`'s bounds, so the guidance here cannot drift from what is enforced. */
const MIN_SECONDS = 5;
const RECOMMENDED_SECONDS = 20;
const MIN_PIXELS = 256;
const MAX_MB = 200;

type Method = "video" | "image";

type Created = {
  id: string;
  name: string;
  source_kind: string;
  duration_seconds: number | null;
  width: number;
  height: number;
  warnings?: string[];
};

const METHODS: {
  key: Method;
  title: string;
  blurb: string;
  points: string[];
  accept: string;
}[] = [
  {
    key: "video",
    title: "Create with a video",
    blurb:
      "Upload a clip of the person sitting still and looking ahead. Their head motion, blinks and posture are reused for the whole session.",
    points: [
      `${RECOMMENDED_SECONDS}s or more — the reference loops`,
      "Looks alive, because the movement is real",
      "One person, front-facing, steady framing",
    ],
    accept: "video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm,.m4v",
  },
  {
    key: "image",
    title: "Create with an image",
    blurb:
      "Upload a front-facing photo, illustration or character. It is expanded into a short clip so the rest of the pipeline is unchanged.",
    points: [
      "Fastest way to get a persona",
      "Holds one pose — there is no motion to reuse",
      "Pair it with any voice",
    ],
    accept: "image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp",
  },
];

export function FaceCreate({ onCreated }: { onCreated: () => void }) {
  const [method, setMethod] = useState<Method | null>(null);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [created, setCreated] = useState<Created | null>(null);
  const input = useRef<HTMLInputElement | null>(null);

  const reset = useCallback(() => {
    setMethod(null);
    setName("");
    setFile(null);
    setProblem(null);
    setCreated(null);
  }, []);

  const submit = useCallback(async () => {
    if (!file) return;
    setBusy(true);
    setProblem(null);
    try {
      // FormData, so the browser sets the multipart boundary. Setting Content-Type by hand here is
      // the classic way to make a multipart upload fail with an unhelpful 422.
      const body = new FormData();
      body.append("name", name.trim() || file.name);
      body.append("file", file);

      const response = await fetch(`${API}/faces/upload`, { method: "POST", body });
      const payload = (await response.json().catch(() => null)) as
        | (Created & { detail?: string })
        | null;
      if (!response.ok) {
        // The runtime's own reason, verbatim. Every rejection here is something about the file
        // that the person who chose it can act on, and paraphrasing it loses the number they need.
        throw new Error(payload?.detail ?? `the runtime rejected this (${response.status})`);
      }
      setCreated(payload as Created);
      onCreated();
    } catch (cause) {
      setProblem(
        cause instanceof Error
          ? cause.message
          : "could not reach the runtime on 127.0.0.1:8000",
      );
    } finally {
      setBusy(false);
    }
  }, [file, name, onCreated]);

  /* ------------------------------------------------------- after uploading */

  if (created) {
    const warnings = created.warnings ?? [];
    return (
      <Card>
        <CardHeader
          title="Face created"
          hint="Prepare it next — that caches the face detection and encoding so a session does not pay for it."
          action={<Button onClick={reset}>Add another</Button>}
        />
        <div className="space-y-4 px-5 py-5">
          <div className="flex flex-wrap items-center gap-2.5">
            <p className="text-[13.5px] font-medium text-ink">{created.name}</p>
            <Chip status="ok">{created.source_kind}</Chip>
            <Chip status="neutral">
              {created.width}×{created.height}
              {created.duration_seconds ? ` · ${created.duration_seconds.toFixed(0)}s` : ""}
            </Chip>
            <span className="font-mono text-[11px] text-ink-low">{created.id}</span>
          </div>

          {/* As prominent as the success, because these are the reasons the result will
              disappoint — and they are still fixable at this moment, cheaply. */}
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
              No cautions — this reference is long enough and large enough to work well.
            </p>
          )}
        </div>
      </Card>
    );
  }

  /* --------------------------------------------------------- choose a method */

  if (method === null) {
    return (
      <Card>
        <CardHeader
          title="Create your face"
          hint="Each method has different time and quality trade-offs — and they are real, not cosmetic: a video lends its motion to every session, an image has none to lend."
        />
        <div className="grid gap-4 px-5 py-5 sm:grid-cols-2">
          {METHODS.map((entry) => (
            <button
              key={entry.key}
              type="button"
              onClick={() => setMethod(entry.key)}
              className="rounded-xl border border-hair-strong bg-glass-raise px-5 py-5 text-left transition-colors hover:border-accent/50"
            >
              <p className="text-[13.5px] font-medium text-ink">{entry.title}</p>
              <p className="mt-2 text-[12.5px] leading-relaxed text-ink-mid">{entry.blurb}</p>
              <ul className="mt-3.5 space-y-1.5 border-t border-hair pt-3.5">
                {entry.points.map((point) => (
                  <li key={point} className="text-[12px] text-ink-low">
                    {point}
                  </li>
                ))}
              </ul>
            </button>
          ))}
        </div>
      </Card>
    );
  }

  /* ------------------------------------------------------------- the picker */

  const chosen = METHODS.find((entry) => entry.key === method)!;
  return (
    <Card>
      <CardHeader
        title={chosen.title}
        hint={chosen.blurb}
        action={<Button onClick={reset}>Back</Button>}
      />
      <div className="space-y-5 px-5 py-5">
        {/* Stated before the picker. Recording a clip is not free, and telling someone the
            minimum after they have uploaded a 3-second one wastes their time twice. */}
        <div className="rounded-lg border border-hair-strong bg-glass-raise px-4 py-3.5">
          <p className="text-[11px] font-medium tracking-[0.07em] uppercase text-ink-low">
            What the runtime will accept
          </p>
          <ul className="mt-2 space-y-1 text-[12px] leading-relaxed text-ink-mid">
            <li>At least {MIN_PIXELS}px on the short side — the face crop is 256×256</li>
            {method === "video" ? (
              <>
                <li>
                  At least {MIN_SECONDS}s, and {RECOMMENDED_SECONDS}s or more is better: the
                  reference plays forward then backward, so an N-second clip repeats every 2N
                  seconds
                </li>
                <li>mp4, mov, webm or m4v — up to {MAX_MB} MB</li>
              </>
            ) : (
              <>
                <li>png, jpg or webp — up to {MAX_MB} MB</li>
                <li>
                  It becomes a short clip automatically, and will hold one pose: there is no
                  movement to reuse
                </li>
              </>
            )}
          </ul>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Name" hint="How it appears when attaching a face to an agent">
            <Input
              value={name}
              placeholder={file?.name ?? "Interviewer persona"}
              onChange={(event) => setName(event.target.value)}
            />
          </Field>
          <Field
            label={method === "video" ? "Video file" : "Image file"}
            hint={file ? `${(file.size / 1_048_576).toFixed(1)} MB selected` : "Nothing selected"}
          >
            <input
              ref={input}
              type="file"
              accept={chosen.accept}
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
            {busy ? "Uploading…" : "Upload and create"}
          </Button>
          <p className="text-[11.5px] text-ink-low">
            {/* Said plainly: the file is kept, because it is not training data. */}
            The file is stored and read on every session that uses this face — the rendered frames
            are its frames, with a new mouth.
          </p>
        </div>
      </div>
    </Card>
  );
}
