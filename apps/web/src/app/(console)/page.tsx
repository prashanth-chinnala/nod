"use client";

/**
 * The console home: what the runtime actually resolved to, and how much is configured.
 *
 * Not a marketing page. The first question anyone opening this has is "is it running, and is
 * it running the real components or the placeholders?" — because a session on placeholders
 * looks identical to a real one until you listen to it. So that answer is the first thing here,
 * above everything else.
 *
 * A placeholder is not an error: a clean clone is meant to run with no credentials. But it
 * must never be mistaken for the real thing, and no number a placeholder produces may be
 * quoted — which is why each one states its own gap rather than just its name.
 */

import { useCallback, useEffect, useState } from "react";

import { Wordmark } from "@/components/logo";
import Link from "next/link";

import { Button, Card, CardHeader, Chip, Page, type Status } from "@/components/ui";

const API = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

type Config = {
  renderer: string;
  llm: string;
  llm_model: string;
  tts: string;
  tts_voice: string;
  stt: string;
  vad: string;
  env_files_read: string[];
};

/** Which value at each boundary means "still a placeholder". */
const PLACEHOLDER: Record<string, string> = {
  renderer: "stub",
  llm: "scripted",
  tts: "tone",
  stt: "none",
  vad: "energy",
};

const BOUNDARIES: Array<{ key: keyof Config & string; label: string; gap: string }> = [
  { key: "renderer", label: "Renderer", gap: "Five rectangles, not a face. Needs a GPU spike run." },
  { key: "llm", label: "Interviewer", gap: "Canned questions, cycled. Not reading answers." },
  { key: "tts", label: "Voice", gap: "A synthesised tone. Timing is real, the voice is not." },
  { key: "stt", label: "Transcription", gap: "Nothing transcribing. Answers reach the model as a duration." },
  { key: "vad", label: "Speech detection", gap: "An energy gate — triggers on a door as readily as a voice." },
];

const RESOURCES = [
  { href: "/agents", name: "Agents", collection: "agents", blurb: "Interviewer configuration" },
  { href: "/faces", name: "Faces", collection: "faces", blurb: "Reference personas" },
  { href: "/knowledge", name: "Knowledge", collection: "knowledge", blurb: "Retrieval context" },
  { href: "/tools", name: "Tools", collection: "tools", blurb: "Callable functions" },
  { href: "/guardrails", name: "Guardrails", collection: "guardrails", blurb: "Input & output policy" },
  { href: "/pronunciations", name: "Pronunciations", collection: "pronunciations", blurb: "Lexicon overrides" },
  { href: "/sessions", name: "Sessions", collection: "sessions", blurb: "Transcripts & latency" },
];

export default function HomePage() {
  const [config, setConfig] = useState<Config | null>(null);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [down, setDown] = useState(false);

  const load = useCallback(async () => {
    try {
      const response = await fetch(`${API}/config`, { cache: "no-store" });
      if (!response.ok) throw new Error();
      setConfig((await response.json()) as Config);
      setDown(false);
    } catch {
      setDown(true);
      return;
    }
    // Best-effort: one unreachable collection must not blank the whole page.
    const entries = await Promise.all(
      RESOURCES.map(async ({ collection }) => {
        try {
          const response = await fetch(`${API}/${collection}`, { cache: "no-store" });
          const rows = (await response.json()) as unknown[];
          return [collection, Array.isArray(rows) ? rows.length : 0] as const;
        } catch {
          return [collection, 0] as const;
        }
      }),
    );
    setCounts(Object.fromEntries(entries));
  }, []);

  useEffect(() => {
    // Deferred by a microtask so the state updates land outside the effect body. The React
    // docs' own guidance for this shape -- subscribing to an external system and calling
    // setState from a callback -- and it keeps the compiler-aware lint rule satisfied without
    // suppressing it.
    let live = true;
    void Promise.resolve().then(() => {
      if (live) void load();
    });
    return () => {
      live = false;
    };
  }, [load]);

  const placeholders = config
    ? BOUNDARIES.filter(({ key }) => config[key] === PLACEHOLDER[key])
    : [];

  return (
    <Page
      title={<Wordmark size={34} trail />}
      lede="A real-time conversational avatar: audio in, talking-head video out, with session lifecycle and interruption handling. This console configures it; a separate runtime serves it."
      action={
        <div className="flex gap-2">
          <Button onClick={() => void load()}>Refresh</Button>
          <Link href="/sessions/new"><Button variant="primary">New session</Button></Link>
        </div>
      }
    >
      {down ? (
        <Card className="border-bad/40 bg-bad/5">
          <div className="px-5 py-4">
            <p className="text-[13px] font-medium text-bad">The runtime is not responding</p>
            <p className="mt-1.5 max-w-2xl text-[12.5px] leading-relaxed text-ink-mid">
              Nothing on this page is live. The console configures a separate process, so it
              cannot tell you anything useful while that process is down — worth saying rather
              than rendering empty tables.
            </p>
            <pre className="mt-3 overflow-x-auto rounded-lg border border-hair bg-base px-3 py-2 font-mono text-[11.5px] text-ink-mid">
              cd apps/api &amp;&amp; python -m uvicorn avatar.server:app
            </pre>
          </div>
        </Card>
      ) : null}

      <Card>
        <CardHeader
          title="What each boundary resolved to"
          hint="A session on placeholders looks identical to a real one until you listen to it, so this comes first."
          action={
            config ? (
              <Chip status={placeholders.length === 0 ? "ok" : "warn"}>
                {placeholders.length === 0
                  ? "all real"
                  : `${placeholders.length} placeholder${placeholders.length > 1 ? "s" : ""}`}
              </Chip>
            ) : null
          }
        />
        <div className="grid gap-px bg-hair sm:grid-cols-2 lg:grid-cols-3">
          {BOUNDARIES.map(({ key, label, gap }) => {
            const value = config?.[key];
            const isPlaceholder = value !== undefined && value === PLACEHOLDER[key];
            const tone: Status = value === undefined ? "neutral" : isPlaceholder ? "warn" : "ok";
            return (
              <div key={key} className="bg-base px-5 py-4">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-[11px] font-medium tracking-[0.06em] uppercase text-ink-low">
                    {label}
                  </span>
                  <Chip status={tone}>{String(value ?? "—")}</Chip>
                </div>
                {/* Shown only when it applies. A caveat about something that works is noise,
                    and noise trains people to skip caveats. */}
                {isPlaceholder ? (
                  <p className="mt-2 text-[11.5px] leading-relaxed text-ink-mid">{gap}</p>
                ) : (
                  <p className="mt-2 font-mono text-[11.5px] text-ink-low">
                    {key === "llm"
                      ? (config?.llm_model ?? "live")
                      : key === "tts"
                        ? (config?.tts_voice ?? "live")
                        : "live"}
                  </p>
                )}
              </div>
            );
          })}
        </div>
        {config?.env_files_read?.length ? (
          <div className="border-t border-hair px-5 py-3">
            <p className="font-mono text-[11px] text-ink-low">
              config from {config.env_files_read.join(", ")} — names only, never values
            </p>
          </div>
        ) : null}
      </Card>

      <Card>
        <CardHeader
          title="Configuration"
          hint="Attach a knowledge base or lexicon to an agent, then select it with AVATAR_AGENT so the live conversation honours it."
        />
        <div className="grid gap-px bg-hair sm:grid-cols-2 lg:grid-cols-4">
          {RESOURCES.map(({ href, name, collection, blurb }) => (
            <Link
              key={href}
              href={href}
              className="group bg-base px-5 py-4 transition-colors hover:bg-glass-raise"
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[13px] font-medium text-ink group-hover:text-accent">
                  {name}
                </span>
                <span className="nums text-[16px] font-semibold text-ink">
                  {counts[collection] ?? "—"}
                </span>
              </div>
              <p className="mt-1 text-[11.5px] text-ink-low">{blurb}</p>
            </Link>
          ))}
        </div>
      </Card>

      <Card>
        <CardHeader
          title="What this does not do yet"
          hint="On the home page deliberately. A reader who checks one claim and finds it false stops trusting the rest."
        />
        <ul className="space-y-2.5 px-5 py-4 text-[12.5px] leading-relaxed text-ink-mid">
          <li>
            <span className="text-ink">No talking-head model is integrated.</span> The renderer
            behind the interface is written and unit-tested, and has never executed — there is no
            GPU here, and the model&apos;s pinned stack does not install on the current free cloud
            runtime.
          </li>
          <li>
            <span className="text-ink">A full turn measures 2.7–5.8s</span> against a sub-second
            target, and none of the three dominant terms is the renderer. Removing it entirely
            still leaves roughly 2.6–5.7s.
          </li>
          <li>
            <span className="text-ink">Guardrails and tools are configurable but not enforced
            mid-turn.</span> Their APIs work and are tested; the live conversation does not call
            them yet.
          </li>
          <li>
            <span className="text-ink">Interruption latency is measured server-side only.</span>{" "}
            It stops when the flush is dispatched, not when audio stops in the candidate&apos;s ear.
          </li>
        </ul>
      </Card>
    </Page>
  );
}
