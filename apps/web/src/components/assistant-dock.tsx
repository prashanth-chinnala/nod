"use client";

/**
 * The assistant, docked in the console chrome.
 *
 * **Why a dock and not only a page.** The questions worth asking are about what is on screen — "why
 * did she score badly on this", "was this interview any good" — and a separate page means leaving
 * the thing you are asking about to go and ask. The panel keeps the record visible beside the answer,
 * which is also what makes the answer checkable.
 *
 * **Mounted in the console layout, so the interview room never gets it.** A candidate is the one
 * person here who is not an operator, and offering them a tool that reads every transcript in the
 * store would be a straightforward data leak. The route group that keeps the sidebar off that page
 * keeps this off too, which is the point of the group existing.
 *
 * **State lives here, above the panel.** A conversation must survive closing the panel and changing
 * page — an assistant that forgets what you asked when you click a link is one you stop using — so
 * the chat is mounted once and hidden with CSS rather than unmounted.
 */

import { useEffect, useState } from "react";

import { AssistantChat } from "@/components/assistant-chat";

export function AssistantDock() {
  const [open, setOpen] = useState(false);
  // No accounts exist, so this is a claim rather than an identity. Sent with every write so a
  // proposal carries a name; the /assistant page states plainly that it is unverified.
  const [actor] = useState("operator");

  // Escape closes, which is the one keyboard convention a slide-over must honour.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((current) => !current)}
        aria-expanded={open}
        aria-label={open ? "Close the assistant" : "Open the assistant"}
        className="fixed top-4 right-5 z-40 flex items-center gap-2 rounded-full border border-hair-strong bg-raise/90 px-3.5 py-2 text-[12.5px] text-ink shadow-lg backdrop-blur transition-colors hover:border-accent/50 hover:text-accent"
      >
        <span aria-hidden="true" className="size-1.5 rounded-full bg-listening" />
        {open ? "Close" : "Ask nod"}
      </button>

      {/* Rendered always, hidden when closed. Unmounting would discard the conversation, and the
          panel is a place you return to mid-thought. */}
      <aside
        aria-label="Assistant"
        aria-hidden={!open}
        className={[
          "fixed top-0 right-0 z-30 flex h-dvh w-full flex-col border-l border-hair bg-raise",
          "shadow-2xl transition-transform duration-200 sm:w-[30rem]",
          open ? "translate-x-0" : "pointer-events-none translate-x-full",
        ].join(" ")}
      >
        <div className="flex items-center gap-2 border-b border-hair px-4 pt-4 pb-3">
          <p className="text-[13px] font-medium text-ink">Assistant</p>
          <p className="text-[11.5px] text-ink-low">reads records · proposes, never decides</p>
        </div>
        <AssistantChat actor={actor} compact />
      </aside>
    </>
  );
}
