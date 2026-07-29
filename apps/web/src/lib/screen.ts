"use client";

/**
 * What the operator is looking at, derived from the URL.
 *
 * **Why derived and not registered.** The obvious design is a context each page pushes into — "I am
 * the report for sess_abc" — and it goes stale the first time someone adds a screen and forgets. A
 * stale context is worse than none: the assistant would answer confidently about the wrong
 * candidate, which is the exact failure this product spends most of its care avoiding. The route
 * already contains the id, cannot disagree with what is rendered, and needs no cooperation from any
 * page.
 *
 * The cost is that this knows the route table, so a new dynamic route needs a line here. That is a
 * visible, compile-adjacent kind of forgetting — a new screen reports a generic label — rather than
 * an invisible one where an id keeps pointing at a page nobody is on.
 */

import { usePathname } from "next/navigation";

export type Screen = {
  route: string;
  /** How a person would name this screen, for the assistant to say back. */
  label: string;
  session_id?: string;
  rubric_id?: string;
  agent_id?: string;
};

/** Route prefix to a human label. Longest match wins, so `/sessions/new` beats `/sessions`. */
const LABELS: [string, string][] = [
  ["/sessions/new", "the new-session form"],
  ["/sessions", "the sessions list"],
  ["/assistant", "the assistant"],
  ["/agents", "the agents list"],
  ["/rubrics", "the rubrics list"],
  ["/knowledge", "the knowledge bases"],
  ["/tools", "the tools list"],
  ["/guardrails", "the guardrails list"],
  ["/pronunciations", "the pronunciation lexicons"],
  ["/faces", "the faces list"],
  ["/", "the console home"],
];

export function useScreen(): Screen {
  const pathname = usePathname() ?? "/";

  // A session report is `/sessions/<id>`, which must not be confused with `/sessions/new`. Matched
  // on the id's own prefix rather than on "a segment after /sessions", so a future
  // `/sessions/export` is a generic label instead of a phantom session id.
  const report = /^\/sessions\/(sess_[A-Za-z0-9_-]+)/.exec(pathname);
  if (report) {
    return {
      route: pathname,
      label: `the report for session ${report[1]}`,
      session_id: report[1],
    };
  }

  const interview = /^\/interview\/(sess_[A-Za-z0-9_-]+)/.exec(pathname);
  if (interview) {
    return {
      route: pathname,
      label: `a live interview room for ${interview[1]}`,
      session_id: interview[1],
    };
  }

  const label = LABELS.find(([prefix]) => pathname.startsWith(prefix))?.[1] ?? "the console";
  return { route: pathname, label };
}
