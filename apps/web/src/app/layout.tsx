import type { Metadata } from "next";
import { Inter_Tight, Outfit } from "next/font/google";

import "./globals.css";

/*
  The root layout holds only what both surfaces share: the document, the font, and the tokens.

  The console's sidebar lives in `(console)/layout.tsx` rather than here, because the candidate's
  interview room must not have it. A candidate is the one person in this product who is not an
  operator, and offering them navigation into Agents and Guardrails would be both confusing and
  wrong. A route group is how that opt-out is expressed without a client-side pathname check.

  Inter Tight rather than Inter: the tighter widths hold up better at the 11px label sizes this UI
  leans on. The mono face is deliberately a system stack — transcripts and IDs need a monospace,
  not a webfont download, and every platform ships a good one.
*/
const sans = Inter_Tight({
  variable: "--font-inter-tight",
  subsets: ["latin"],
  display: "swap",
});

/*
  Outfit carries the wordmark and nothing else -- the identity specifies it for the logo and
  Inter Tight for the interface, and mixing the two into body copy would lose the distinction
  that makes a wordmark read as a mark.

  Loaded through `next/font` rather than a stylesheet link, which self-hosts the file at build
  time: no request to a font CDN at runtime, and the weight is preloaded, so the wordmark does
  not flash in the fallback face on first paint. Only weight 500 is fetched, because that is
  the only weight the mark uses.
*/
const logo = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
  weight: ["500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "nod",
  description:
    "A real-time conversational avatar: audio in, talking-head video out, with session lifecycle and interruption handling.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${logo.variable} h-full antialiased`}>
      <body className="min-h-full">{children}</body>
    </html>
  );
}
