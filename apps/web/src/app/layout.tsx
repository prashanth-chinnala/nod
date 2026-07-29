import type { Metadata } from "next";
import { Inter_Tight } from "next/font/google";

import { Nav } from "@/components/nav";
import "./globals.css";

/*
  Inter Tight rather than Inter: the tighter widths hold up better at the 11px label sizes
  this UI leans on, and it matches the live session client, which already asks for it. The
  mono face is deliberately a system stack — transcripts and IDs need a monospace, not a
  webfont download, and every platform ships a good one.
*/
const sans = Inter_Tight({
  variable: "--font-inter-tight",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "nod — console",
  description:
    "Configure interviewer agents, faces, knowledge, tools and guardrails for a real-time conversational avatar.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} h-full antialiased`}>
      <body className="flex min-h-full">
        <Nav />
        {/* min-w-0 is load-bearing: without it a wide table inside a flex child refuses to
            shrink and the whole page scrolls sideways instead of the table. */}
        <main className="min-w-0 flex-1">{children}</main>
      </body>
    </html>
  );
}
