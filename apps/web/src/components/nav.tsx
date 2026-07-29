"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/**
 * The sidebar.
 *
 * Grouped rather than a flat list, and the grouping is a claim about the product: an Agent is
 * the thing you configure, everything under "Attach" is a component it references, and
 * "Observe" is where you find out whether any of it worked. A flat list of nine items would
 * hide that an Agent is the centre and the rest hang off it.
 */

const GROUPS = [
  {
    label: "Configure",
    items: [
      { href: "/agents", name: "Agents", hint: "Interviewer configuration" },
      { href: "/faces", name: "Faces", hint: "Reference personas" },
    ],
  },
  {
    label: "Attach",
    items: [
      { href: "/knowledge", name: "Knowledge", hint: "Retrieval context" },
      { href: "/tools", name: "Tools", hint: "Callable functions" },
      { href: "/guardrails", name: "Guardrails", hint: "Input & output policy" },
      { href: "/pronunciations", name: "Pronunciations", hint: "Lexicon overrides" },
    ],
  },
  {
    label: "Observe",
    items: [{ href: "/sessions", name: "Sessions", hint: "Transcripts & latency" }],
  },
] as const;

export function Nav() {
  const path = usePathname();

  return (
    <nav
      aria-label="Console"
      className="sticky top-0 flex h-dvh w-60 shrink-0 flex-col border-r border-hair bg-raise"
    >
      <div className="border-b border-hair px-5 py-5">
        <Link href="/" className="block">
          <span className="text-[15px] font-semibold tracking-tight text-ink">nod</span>
          <span className="mt-0.5 block text-[11px] text-ink-low">
            conversational avatar console
          </span>
        </Link>
      </div>

      <div className="flex-1 overflow-y-auto py-4">
        {GROUPS.map((group) => (
          <div key={group.label} className="mb-5">
            <p className="px-5 pb-1.5 text-[10.5px] font-medium tracking-[0.09em] uppercase text-ink-low">
              {group.label}
            </p>
            {group.items.map((item) => {
              const active = path === item.href || path.startsWith(item.href + "/");
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={[
                    "block border-l-2 px-5 py-2 transition-colors",
                    active
                      ? "border-accent bg-accent/8 text-ink"
                      : "border-transparent text-ink-mid hover:bg-glass hover:text-ink",
                  ].join(" ")}
                >
                  <span className="block text-[13px] font-medium">{item.name}</span>
                  <span className="block text-[11px] text-ink-low">{item.hint}</span>
                </Link>
              );
            })}
          </div>
        ))}
      </div>

      {/*
        The live session lives in this app, not on the runtime's own port. The plain-JS page
        the API still serves at :8000 is kept as a minimal-dependency reference — the API's
        README promises a clean clone reaches a running prototype with no build step — but the
        product surface is here, so the console never sends anyone to another origin.
      */}
      <div className="border-t border-hair px-5 py-4">
        <Link
          href="/sessions/new"
          className="block rounded-lg border border-accent/45 bg-accent/12 px-3 py-2 text-center text-[12.5px] font-medium text-accent transition-colors hover:bg-accent/20"
        >
          Start a session
        </Link>
        <p className="mt-2 text-[11px] leading-relaxed text-ink-low">
          Talk or type, then interrupt mid-answer
        </p>
      </div>
    </nav>
  );
}
