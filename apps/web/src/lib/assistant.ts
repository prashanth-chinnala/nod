"use client";

/**
 * The assistant conversation, as a hook.
 *
 * **Why `fetch` and not `EventSource`.** The browser's SSE client is GET-only, and the question plus
 * its history has to go in a body — a conversation in a query string would be logged by every proxy
 * between here and the service, with transcript quotes in it. So this reads the response stream
 * directly and parses `data:` frames, which is a dozen lines and buys the right request shape.
 *
 * **Tool activity is part of the transcript, not a spinner.** A consistency audit across six
 * candidates spends most of its time in tool calls with nothing to show, and an undifferentiated
 * spinner for fifteen seconds reads as broken. Each tool appears as a step attached to the answer
 * being composed, so the wait becomes progress — and it makes the assistant legible: you can see it
 * checked interview quality before it commented on a low score, which is the difference between
 * trusting the answer and taking it on faith.
 *
 * **The client owns the conversation.** The service is stateless and history is replayed on every
 * request. That keeps a second store of candidate data from existing, with its own retention
 * question, and it means the service can restart mid-conversation without losing the thread.
 */

import { useCallback, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_ASSISTANT_BASE ?? "http://127.0.0.1:8100";

export type ToolStep = { name: string; done: boolean };

export type Message = {
  role: "user" | "assistant";
  content: string;
  /** Tools this answer called, in order, with whether each has finished. */
  tools?: ToolStep[];
  error?: string;
};

/** One `data:` frame from the service. Discriminated so nothing has to parse prose. */
type Event =
  | { type: "token"; text: string }
  | { type: "tool"; name: string; state: "start" | "end" }
  | { type: "error"; detail: string }
  | { type: "done" };

export function useAssistant(actor: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);

  /** Rewrite the in-flight assistant message. Every stream event is one of these. */
  const patchLast = useCallback((change: (draft: Message) => Message) => {
    setMessages((prev) => {
      const last = prev.at(-1);
      if (!last || last.role !== "assistant") return prev;
      return [...prev.slice(0, -1), change(last)];
    });
  }, []);

  const send = useCallback(
    async (question: string) => {
      const text = question.trim();
      if (!text || busy) return;
      setBusy(true);

      // History is captured before the two new messages are appended, so the service receives the
      // conversation as it stood when the question was asked.
      const history = messages.map(({ role, content }) => ({ role, content }));
      setMessages((prev) => [
        ...prev,
        { role: "user", content: text },
        { role: "assistant", content: "", tools: [] },
      ]);

      const controller = new AbortController();
      abort.current = controller;

      try {
        const response = await fetch(`${API}/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, history, actor }),
          signal: controller.signal,
        });
        if (!response.ok || !response.body) {
          throw new Error(`the assistant answered ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        // Frames can be split across chunks, so what is left after the last newline pair is kept
        // rather than parsed. Dropping it loses a token mid-word, which reads as a typo in the
        // model's output and is impossible to attribute later.
        let buffer = "";

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          // CRLF normalised first. SSE specifies either, and `sse_starlette` sends `\r\n\r\n`
          // -- splitting on `\n\n` never matched, so every frame stayed in the buffer and the UI
          // showed a question with no answer while the network tab showed 6KB arriving. Nothing
          // errored, which is what made it worth a comment.
          buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";

          for (const frame of frames) {
            const line = frame.split("\n").find((l) => l.startsWith("data:"));
            if (!line) continue;
            let event: Event;
            try {
              event = JSON.parse(line.slice(5).trim()) as Event;
            } catch {
              continue; // a malformed frame must not end a working stream
            }

            if (event.type === "token") {
              patchLast((draft) => ({ ...draft, content: draft.content + event.text }));
            } else if (event.type === "tool") {
              patchLast((draft) => {
                const tools = [...(draft.tools ?? [])];
                if (event.state === "start") {
                  tools.push({ name: event.name, done: false });
                } else {
                  // Mark the most recent unfinished call of that name. Matching by name alone
                  // would close the wrong step when a tool is used twice in one answer.
                  const index = tools.findLastIndex((t) => t.name === event.name && !t.done);
                  if (index >= 0) tools[index] = { ...tools[index], done: true };
                }
                return { ...draft, tools };
              });
            } else if (event.type === "error") {
              patchLast((draft) => ({ ...draft, error: event.detail }));
            }
          }
        }
      } catch (cause) {
        if (!controller.signal.aborted) {
          patchLast((draft) => ({
            ...draft,
            error:
              cause instanceof Error
                ? cause.message
                : "the assistant service is unreachable — start it on :8100",
          }));
        }
      } finally {
        abort.current = null;
        setBusy(false);
      }
    },
    [actor, busy, messages, patchLast],
  );

  /** Stop generating. The partial answer is kept: it is what the tools already found. */
  const stop = useCallback(() => {
    abort.current?.abort();
    setBusy(false);
  }, []);

  const reset = useCallback(() => {
    abort.current?.abort();
    setMessages([]);
    setBusy(false);
  }, []);

  return { messages, busy, send, stop, reset };
}

export type Capabilities = {
  reads: { name: string; summary: string }[];
  writes: { name: string; summary: string }[];
  model: string;
  auth: string;
  writes_are_proposals: boolean;
  note: string;
};

/** Fetch what the assistant can see and change, so the UI can disclose it rather than assert it. */
export async function fetchCapabilities(): Promise<Capabilities | null> {
  try {
    const response = await fetch(`${API}/capabilities`, { cache: "no-store" });
    if (!response.ok) return null;
    return (await response.json()) as Capabilities;
  } catch {
    return null;
  }
}
