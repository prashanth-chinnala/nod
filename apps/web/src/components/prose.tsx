/**
 * A deliberately small markdown renderer for model output.
 *
 * **Why not a markdown library.** This text comes from a model and frequently contains verbatim
 * candidate transcript quoted back. Handing that to a renderer that emits HTML — even a sanitising
 * one — means the safety of a page that displays interview records depends on a sanitiser
 * configuration. Nothing here produces HTML from the input: every branch returns React elements
 * built from plain strings, so there is no path by which a transcript could inject markup. That
 * property is worth more than footnote support.
 *
 * **Why render anything at all.** The first version showed the raw text, on the reasoning above,
 * and it was the wrong call in the other direction: the model emits `**bold**` and `- ` bullets
 * constantly, so an answer arrived full of asterisks and read as broken. Supporting the four things
 * it actually uses — bold, inline code, bullets, and pipe tables — covers essentially all of it.
 *
 * Anything unrecognised falls through as text. An unsupported construct renders as its source,
 * which is mildly ugly and never wrong — the failure mode of a partial implementation should be
 * visible, not a mangled sentence.
 */

import type { ReactNode } from "react";

/** Split on `**bold**` and `` `code` ``, keeping the delimiters so the parts can be typed. */
const INLINE = /(\*\*[^*]+\*\*|`[^`]+`)/g;

function inline(text: string, keyPrefix: string): ReactNode[] {
  return text.split(INLINE).map((part, index) => {
    const key = `${keyPrefix}-${index}`;
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return (
        <strong key={key} className="font-medium text-ink">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      return (
        <code key={key} className="font-mono text-[12px] text-accent">
          {part.slice(1, -1)}
        </code>
      );
    }
    return <span key={key}>{part}</span>;
  });
}

/** A pipe-table row split into cells, with the outer pipes dropped. */
function cells(line: string): string[] {
  return line
    .trim()
    .replace(/^\||\|$/g, "")
    .split("|")
    .map((cell) => cell.trim());
}

const IS_DIVIDER = /^[\s|:-]+$/;

export function Prose({ text }: { text: string }) {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    // Tables. Gathered as a run rather than line by line, because a row only means anything
    // alongside its siblings -- and the model reaches for a table whenever it compares candidates,
    // which is the case this whole screen exists for.
    if (line.includes("|") && line.trim().startsWith("|")) {
      const run: string[] = [];
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        run.push(lines[index]);
        index += 1;
      }
      const rows = run.filter((row) => !IS_DIVIDER.test(row));
      if (rows.length > 0) {
        const [head, ...body] = rows;
        blocks.push(
          // Scrolls in its own container: a six-candidate comparison is wider than the panel, and
          // the page body must never scroll sideways.
          <div key={`t-${index}`} className="my-2 overflow-x-auto">
            <table className="w-full border-collapse text-[12.5px]">
              <thead>
                <tr className="border-b border-hair">
                  {cells(head).map((cell, i) => (
                    <th
                      key={i}
                      className="px-2.5 py-1.5 text-left font-medium whitespace-nowrap text-ink-low"
                    >
                      {inline(cell, `th-${i}`)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {body.map((row, r) => (
                  <tr key={r} className="border-b border-hair/50 last:border-0">
                    {cells(row).map((cell, c) => (
                      <td key={c} className="px-2.5 py-1.5 align-top text-ink-mid">
                        {inline(cell, `td-${r}-${c}`)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>,
        );
      }
      continue;
    }

    // Bullets, including the `*` and `•` the model also uses. Grouped into one list so the spacing
    // is a list's rather than a stack of paragraphs'.
    if (/^\s*[-*•]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\s*[-*•]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*[-*•]\s+/, ""));
        index += 1;
      }
      blocks.push(
        <ul key={`u-${index}`} className="my-1.5 ml-4 list-disc space-y-1">
          {items.map((item, i) => (
            <li key={i} className="text-ink-mid">
              {inline(item, `li-${i}`)}
            </li>
          ))}
        </ul>,
      );
      continue;
    }

    if (!line.trim()) {
      index += 1;
      continue;
    }

    blocks.push(
      <p key={`p-${index}`} className="my-1.5 text-ink">
        {inline(line, `p-${index}`)}
      </p>,
    );
    index += 1;
  }

  return <div className="text-[13px] leading-relaxed">{blocks}</div>;
}
