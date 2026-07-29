"use client";

/**
 * The assistant's full page: the same conversation as the docked panel, plus the disclosure.
 *
 * **Why both a page and a dock.** The dock is where the work happens — the questions worth asking
 * are about the record on screen, and leaving that screen to ask about it is the wrong shape. The
 * page exists for two things the panel cannot do well: reading a long comparison without a 30rem
 * column, and disclosing what the assistant can see and change. That disclosure belongs on a page
 * someone can link to and read, not behind a toggle in a side panel.
 *
 * The conversation itself is `AssistantChat`, the same component the dock renders. One surface, so a
 * feature cannot exist in one place and quietly go missing from the other.
 */

import { useEffect, useState } from "react";

import { AssistantChat } from "@/components/assistant-chat";
import { Card, CardHeader, Chip, Page } from "@/components/ui";
import { fetchCapabilities, type Capabilities } from "@/lib/assistant";

export default function AssistantPage() {
  // No accounts exist, so this is a claim rather than an identity. Sent with every write so a
  // proposal carries a name; the panel below says plainly that it is unverified.
  const [actor, setActor] = useState("operator");
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);

  useEffect(() => {
    fetchCapabilities().then(setCapabilities);
  }, []);

  return (
    <Page
      title="Assistant"
      lede="The same assistant as the panel, with room for a long answer — and the disclosure of what it can see and change. It reads records, finds contradictions between them, and drafts changes for you to apply. It never decides anything."
    >
      <Card>
        <CardHeader
          title="Conversation"
          hint="Every step it takes is shown. If it answers about a transcript it never opened, you will see that."
        />
        <AssistantChat actor={actor} />
      </Card>

      <CapabilityPanel capabilities={capabilities} actor={actor} onActor={setActor} />
    </Page>
  );
}

/** What it can read, what it can change, and the two things it cannot do. */
function CapabilityPanel({
  capabilities,
  actor,
  onActor,
}: {
  capabilities: Capabilities | null;
  actor: string;
  onActor: (value: string) => void;
}) {
  return (
    <Card>
      <CardHeader
        title="What the assistant can do"
        hint="Read from the service rather than written here, so this cannot drift from what the code actually allows."
      />
      {capabilities === null ? (
        <div className="px-5 py-5 text-[12.5px] text-ink-mid">
          The assistant service is not reachable on 127.0.0.1:8100. Start it with{" "}
          <code className="font-mono text-ink">
            uvicorn assistant.server:app --port 8100
          </code>{" "}
          from <code className="font-mono text-ink">apps/assistant</code>.
        </div>
      ) : (
        <div className="space-y-5 px-5 py-5">
          <div className="flex flex-wrap items-center gap-2">
            <Chip status="info">model {capabilities.model}</Chip>
            <Chip status={capabilities.writes_are_proposals ? "ok" : "bad"}>
              {capabilities.writes_are_proposals
                ? "writes are proposals only"
                : "writes are applied directly"}
            </Chip>
            <Chip status="warn">no authentication</Chip>
          </div>

          <div className="grid gap-5 sm:grid-cols-2">
            <div>
              <p className="text-[11px] font-medium tracking-[0.07em] uppercase text-ink-low">
                Can read ({capabilities.reads.length})
              </p>
              <ul className="mt-2 space-y-1.5">
                {capabilities.reads.map((tool) => (
                  <li key={tool.name} className="text-[12px] leading-relaxed text-ink-mid">
                    <span className="font-mono text-[11.5px] text-ink">{tool.name}</span> —{" "}
                    {tool.summary}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-[11px] font-medium tracking-[0.07em] uppercase text-ink-low">
                Can propose ({capabilities.writes.length})
              </p>
              <ul className="mt-2 space-y-1.5">
                {capabilities.writes.map((tool) => (
                  <li key={tool.name} className="text-[12px] leading-relaxed text-ink-mid">
                    <span className="font-mono text-[11.5px] text-warn">{tool.name}</span> —{" "}
                    {tool.summary}
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* The two refusals, stated on the screen and not only in the prompt. Someone about to
              ask for a ranking should learn the answer here rather than from a refusal. */}
          <div className="rounded-lg border border-hair-strong bg-glass-raise px-4 py-3.5">
            <p className="text-[12px] font-medium text-ink">Two things it will not do</p>
            <p className="mt-1.5 text-[12px] leading-relaxed text-ink-mid">
              It will not recommend hiring or rejecting anyone, or rank candidates — the decision is
              yours, and a call made by a model reading a transcript is one nobody can be
              accountable for. And it will not find candidates similar to a past hire: ranking
              people by resemblance to who was hired before learns that panel&apos;s previous
              preferences, including its biases. Offer letters are useful for calibrating the{" "}
              <em>scale</em> — promoting a verified quote as an example of what a rating sounds like
              — never for modelling a person.
            </p>
          </div>

          <div className="rounded-lg border border-warn/40 bg-warn/5 px-4 py-3.5">
            <p className="text-[12px] leading-relaxed text-ink-mid">
              <span className="font-medium text-warn">No authentication exists.</span>{" "}
              {capabilities.auth}. The name below is attached to any proposal you cause and is not
              verified — it makes the audit trail complete in shape from now rather than starting on
              the day accounts exist.
            </p>
            <label className="mt-3 flex items-center gap-2 text-[12px] text-ink-mid">
              Recorded as
              <input
                value={actor}
                onChange={(event) => onActor(event.target.value)}
                aria-label="Your name, recorded on proposals"
                className="w-48 rounded-lg border border-hair-strong bg-base px-2.5 py-1.5 text-[12.5px] text-ink"
              />
            </label>
          </div>
        </div>
      )}
    </Card>
  );
}
