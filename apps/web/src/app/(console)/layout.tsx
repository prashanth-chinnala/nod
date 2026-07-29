import type { Metadata } from "next";

import { AssistantDock } from "@/components/assistant-dock";
import { Nav } from "@/components/nav";

/**
 * The operator's console: sidebar plus content.
 *
 * A route group rather than a path segment, so the URLs stay `/agents` and `/sessions` — the
 * grouping is about which chrome wraps a page, not about how it is addressed. The candidate's
 * interview room sits outside this group and therefore gets no navigation, which is the whole
 * reason the group exists.
 */

export const metadata: Metadata = {
  title: "nod — console",
  description:
    "Configure interviewer agents, faces, knowledge, tools and guardrails for a real-time conversational avatar.",
};

export default function ConsoleLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="flex min-h-dvh">
      <Nav />
      {/* min-w-0 is load-bearing: without it a wide table inside a flex child refuses to shrink
          and the whole page scrolls sideways instead of the table. */}
      <main className="min-w-0 flex-1">{children}</main>
      {/* Console-only, deliberately. The candidate's interview room sits outside this route group
          and must not get a tool that can read every transcript in the store. */}
      <AssistantDock />
    </div>
  );
}
