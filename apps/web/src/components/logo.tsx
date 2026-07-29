/**
 * The nod wordmark and app mark.
 *
 * **The mark is the word.** "nod" with the `o` replaced by a filled circle — the lamp — and,
 * in the animated variant, a ghost of that circle sitting above it. The ghost is the nod: a head
 * dipping, caught mid-motion. That means there is no separate glyph to set beside the word, and
 * putting one there would say the same thing twice. Every placement below renders the wordmark
 * itself; `Mark` exists only for the app icon, where there is no room for letters at all.
 *
 * **Built in HTML and CSS rather than as SVG paths.** Two reasons, and the first is the one that
 * decides it: the letters inherit `currentColor`, so the wordmark works on any ground and picks up
 * a hover state for free — an SVG with baked fills would need a variant per surface. The second is
 * that tracing Outfit's `n` and `d` into paths would freeze a licensed typeface into this repo at
 * one weight, and the identity specifies the live font. The cost is a dependency on Outfit
 * loading; `next/font` self-hosts and preloads it, and the failure mode is legible rather than
 * broken — the fallback stack renders "n●d" in Inter Tight, visibly wrong but still the product's
 * name.
 *
 * **The app icon is the opposite choice: pure SVG circles, no text.** It has to survive 16px and
 * has no font to depend on, so it is two `<circle>` elements and nothing else.
 *
 * All geometry is in `em`, so one `size` prop scales the whole lockup and the proportions cannot
 * drift between placements. Lamp `0.515em` and tracking `-0.045em` are the identity's; the trail
 * offset is not, and the comment at that line says why it was changed and what it looked like
 * before.
 */

type LampTone = "rest" | "live" | "ink";

/**
 * Which colour the lamp burns.
 *
 * `rest` and `live` are the identity's two lamp variants, and they are not arbitrary: the
 * identity's default cyan is exactly the app's `listening` state colour, and its "on air" amber is
 * exactly `speaking`. That coincidence is worth using rather than noting — in the interview room
 * the lamp is driven by the session state, so the logo reports whether the interviewer is
 * listening or talking. A brand mark that carries real information earns its place in a UI that is
 * otherwise deliberately quiet.
 *
 * `ink` is the knockout case, for placement on the accent itself where a coloured lamp would
 * vibrate against the ground.
 */
const LAMP: Record<LampTone, string> = {
  rest: "var(--color-listening)",
  live: "var(--color-speaking)",
  ink: "currentColor",
};

export function Wordmark({
  size = 15,
  tone = "rest",
  trail = false,
  className = "",
}: {
  /** Cap height in px. Everything else is derived, so proportions cannot drift. */
  size?: number;
  tone?: LampTone;
  /**
   * Show the ghost circle above the lamp.
   *
   * Off by default so a caller in a tight space gets the quiet cut without asking, which is what
   * the identity means by its static variant for "dense ui, favicons". The trail is safe to turn
   * on anywhere, though — the element reserves its own headroom — so the choice is about whether
   * the surface wants a mark that moves, not about whether it will fit.
   */
  trail?: boolean;
  className?: string;
}) {
  const lamp = LAMP[tone];
  return (
    <span
      // `inline-flex` with baseline alignment is what keeps the lamp sitting on the same line as
      // the letters. `align-items: center` would centre it against the cap height and leave it
      // floating a pixel or two high, which is visible at hero sizes and not at nav sizes -- the
      // worst kind of bug, because it only appears on the screen you check last.
      className={`inline-flex select-none items-baseline whitespace-nowrap font-logo ${className}`}
      style={{
        fontSize: `${size}px`,
        fontWeight: 500,
        letterSpacing: "-0.045em",
        lineHeight: 1,
        // The trail is absolutely positioned above the lamp and contributes nothing to layout, so
        // left alone it draws outside the element and gets clipped by the nearest header padding.
        // Reserving the space here makes the mark self-contained: a caller can drop it into a
        // heading without knowing it needs headroom. Sized to the ghost's actual reach -- it tops
        // out ~0.56em above the content box, so 0.62em clears it with a little margin and no more.
        paddingTop: trail ? "0.62em" : undefined,
      }}
      // The accessible name, because the lamp is a decorative element standing in for a letter:
      // a screen reader would otherwise announce "n d".
      role="img"
      aria-label="nod"
    >
      <span aria-hidden="true">n</span>
      <span
        aria-hidden="true"
        className="relative inline-block rounded-full"
        style={{
          width: "0.515em",
          height: "0.515em",
          margin: "0 0.028em",
          background: lamp,
        }}
      >
        {trail ? (
          <span
            className="absolute left-0 rounded-full"
            style={{
              // The identity's wordmark CSS specifies `-1.28em`, which puts 0.765em of clear air
              // between the ghost and the lamp -- more than one and a half lamp diameters. Built
              // and looked at, that does not read as a trail: it reads as an unrelated bullet
              // floating above the word, and at 64px it reads as a bug. Tightened to a 0.32em gap,
              // which is close enough that the eye groups the two as one object in motion. The
              // identity's own app icon agrees with the tighter reading -- its two circles sit
              // 19 units apart on a 64 grid with r=8, a gap of 0.34 diameters.
              top: "-0.78em",
              width: "100%",
              height: "100%",
              // The lamp colour, not `currentColor`. The identity's wordmark CSS uses the text
              // colour here, which on this ground renders a grey disc that looks like a different
              // element rather than the same lamp a moment ago. Its app icon uses the cyan at 22%,
              // and that is the version that reads correctly -- an after-image is the same light,
              // dimmer.
              background: lamp,
              // 0.45, not the identity's 0.2. Any low alpha over a near-black ground mixes toward
              // black, so at 20 % the trail reads as a hole punched in the surface rather than as
              // a dimmer light -- the identity's own hero gets away with it because a cyan radial
              // wash lifts the ground behind it, and this console has no such wash. Raised until
              // the ghost reads as the same lamp a moment ago, which is the whole idea.
              opacity: 0.45,
            }}
          />
        ) : null}
      </span>
      <span aria-hidden="true">d</span>
    </span>
  );
}

/**
 * The app mark: two circles, no letters.
 *
 * For the icon only. `tile` draws the rounded background the identity specifies for an app icon;
 * without it the circles sit transparent, which is what a monochrome or in-page placement wants.
 *
 * The 16px case is a different drawing in the identity — one circle rather than two, because two
 * become mush — and that is handled by shipping a separate file rather than by a prop, since a
 * favicon is chosen by the browser and cannot ask a component anything.
 */
export function Mark({
  size = 32,
  tile = true,
  className = "",
}: {
  size?: number;
  tile?: boolean;
  className?: string;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      className={className}
      role="img"
      aria-label="nod"
    >
      {tile ? <rect width="64" height="64" rx="15" fill="var(--color-raise)" /> : null}
      {/* The ghost first, so the lamp overlaps it rather than the reverse. */}
      <circle cx="32" cy="22" r="8" fill="var(--color-listening)" opacity="0.22" />
      <circle cx="32" cy="41" r="8" fill="var(--color-listening)" />
    </svg>
  );
}
