/**
 * The console's primitives.
 *
 * Deliberately hand-written rather than pulled from a component kit: there are nine of them,
 * they are each a few lines, and owning them means the look is a decision rather than a
 * default. A kit would also bring its own opinions about colour, which fights the palette in
 * globals.css.
 *
 * The one rule running through all of them: state is encoded in *form* as well as in text.
 * A failed row and a ready row must be distinguishable without reading the word, because
 * these screens are scanned rather than read.
 */

import type { ReactNode } from "react";

function cx(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/* ---------------------------------------------------------------- surfaces */

export function Card({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cx(
        "rounded-xl border border-hair bg-glass backdrop-blur-sm",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  hint,
  action,
}: {
  title: string;
  hint?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex items-start gap-4 border-b border-hair px-5 py-4">
      <div className="min-w-0 flex-1">
        <h2 className="text-[13px] font-medium tracking-wide text-ink">{title}</h2>
        {hint ? <p className="mt-1 text-xs leading-relaxed text-ink-mid">{hint}</p> : null}
      </div>
      {action}
    </div>
  );
}

/* ------------------------------------------------------------------ status */

export type Status = "ok" | "warn" | "bad" | "info" | "neutral";

const STATUS_STYLE: Record<Status, string> = {
  ok: "border-ok/35 bg-ok/10 text-ok",
  warn: "border-warn/35 bg-warn/10 text-warn",
  bad: "border-bad/35 bg-bad/10 text-bad",
  info: "border-info/35 bg-info/10 text-info",
  neutral: "border-hair-strong bg-glass-raise text-ink-mid",
};

/**
 * A status chip. Carries a dot as well as a colour and a word, so it survives being
 * screenshotted in greyscale and read by someone who cannot distinguish the hues — colour
 * alone is never the only signal.
 */
export function Chip({ status, children }: { status: Status; children: ReactNode }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5",
        "text-[11px] font-medium tracking-wide uppercase",
        STATUS_STYLE[status],
      )}
    >
      <span aria-hidden className="size-1.5 rounded-full bg-current" />
      {children}
    </span>
  );
}

/* ----------------------------------------------------------------- actions */

export function Button({
  children,
  variant = "ghost",
  type = "button",
  disabled,
  onClick,
}: {
  children: ReactNode;
  variant?: "primary" | "ghost" | "danger";
  type?: "button" | "submit";
  disabled?: boolean;
  onClick?: () => void;
}) {
  const style = {
    primary: "border-accent/50 bg-accent/15 text-accent hover:bg-accent/25",
    ghost: "border-hair-strong bg-glass-raise text-ink hover:border-ink-low",
    danger: "border-bad/40 bg-bad/10 text-bad hover:bg-bad/20",
  }[variant];

  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={cx(
        "rounded-lg border px-3 py-1.5 text-[12.5px] font-medium",
        "transition-colors disabled:pointer-events-none disabled:opacity-40",
        style,
      )}
    >
      {children}
    </button>
  );
}

/* ------------------------------------------------------------------ tables */

/**
 * A column heading. A bare string is a left-aligned text column; `{ label, align: "right" }`
 * is a numeric one.
 *
 * Alignment lives on the header rather than being inferred, because the header and its cells
 * have to agree and nothing else can enforce that. They did not agree before this existed:
 * numeric cells were right-aligned via `<Cell right>` while every header stayed left, so a
 * count sat visibly adrift from the word naming it — the wider the column, the worse the gap.
 */
export type Column = string | { label: string; align?: "left" | "right" };

export function Table({
  head,
  children,
}: {
  head: readonly Column[];
  children: ReactNode;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-full border-collapse text-[13px]">
        <thead>
          <tr>
            {head.map((column) => {
              const { label, align } =
                typeof column === "string" ? { label: column, align: "left" as const } : column;
              return (
                <th
                  key={label}
                  scope="col"
                  className={cx(
                    "border-b border-hair px-5 py-2.5",
                    align === "right" ? "text-right" : "text-left",
                    "text-[11px] font-medium tracking-[0.06em] uppercase text-ink-low",
                  )}
                >
                  {label}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

/** Shorthand for a numeric column, so call sites stay readable. */
export function num(label: string): Column {
  return { label, align: "right" };
}

export function Row({ children }: { children: ReactNode }) {
  return (
    <tr className="border-b border-hair/60 last:border-0 hover:bg-glass-raise/40">
      {children}
    </tr>
  );
}

export function Cell({
  children,
  mono,
  dim,
  right,
}: {
  children: ReactNode;
  mono?: boolean;
  dim?: boolean;
  right?: boolean;
}) {
  return (
    <td
      className={cx(
        "px-5 py-3 align-middle",
        mono && "font-mono text-[12px]",
        dim && "text-ink-mid",
        right && "text-right",
      )}
    >
      {children}
    </td>
  );
}

/* ------------------------------------------------------------ empty states */

/**
 * Every page starts here, which makes this the most-viewed screen in the console.
 *
 * So it gets a real one: what this is, why it matters, and the single action that fills it.
 * "No data" is a wasted screen at the exact moment someone needs orienting.
 */
export function Empty({
  title,
  children,
  action,
}: {
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="px-5 py-14 text-center">
      <p className="text-[13.5px] font-medium text-ink">{title}</p>
      <p className="mx-auto mt-2 max-w-md text-[12.5px] leading-relaxed text-ink-mid">
        {children}
      </p>
      {action ? <div className="mt-5 flex justify-center">{action}</div> : null}
    </div>
  );
}

/* -------------------------------------------------------------- page frame */

export function Page({
  title,
  lede,
  action,
  children,
}: {
  title: string;
  lede: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="mx-auto max-w-6xl px-8 py-10">
      <header className="mb-7 flex items-end gap-6">
        <div className="min-w-0 flex-1">
          <h1 className="text-[22px] font-semibold tracking-tight text-ink">{title}</h1>
          <p className="mt-1.5 max-w-2xl text-[13px] leading-relaxed text-ink-mid">{lede}</p>
        </div>
        {action}
      </header>
      <div className="space-y-6">{children}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ fields */

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block">
      <span className="mb-1.5 block text-[12px] font-medium text-ink">{label}</span>
      {children}
      {hint ? <span className="mt-1.5 block text-[11.5px] text-ink-low">{hint}</span> : null}
    </label>
  );
}

const CONTROL =
  "w-full rounded-lg border border-hair-strong bg-base px-3 py-2 text-[13px] " +
  "text-ink placeholder:text-ink-low";

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cx(CONTROL, props.className)} />;
}

export function Textarea(props: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea {...props} className={cx(CONTROL, "min-h-24 resize-y", props.className)} />
  );
}

export function Select(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select {...props} className={cx(CONTROL, props.className)} />;
}

/* ----------------------------------------------------------------- metrics */

/**
 * A single measured figure.
 *
 * `target` renders as a comparison rather than decoration: this product's whole story is a
 * latency budget it does not currently meet, so a number without its target beside it is
 * less than half the information.
 */
export function Metric({
  label,
  value,
  unit,
  target,
  status = "neutral",
}: {
  label: string;
  value: string | number;
  unit?: string;
  target?: string;
  status?: Status;
}) {
  const tone = {
    ok: "text-ok",
    warn: "text-warn",
    bad: "text-bad",
    info: "text-info",
    neutral: "text-ink",
  }[status];

  return (
    <div className="px-5 py-4">
      <p className="text-[11px] font-medium tracking-[0.06em] uppercase text-ink-low">
        {label}
      </p>
      <p className={cx("nums mt-1.5 text-[24px] leading-none font-semibold", tone)}>
        {value}
        {unit ? <span className="ml-1 text-[13px] font-normal text-ink-mid">{unit}</span> : null}
      </p>
      {target ? <p className="nums mt-1.5 text-[11.5px] text-ink-low">{target}</p> : null}
    </div>
  );
}
