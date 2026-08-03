"use client";

/**
 * The Faces list.
 *
 * `enrollment_ms` is a column rather than a detail-page footnote, and that is the one real
 * design decision on this screen. Identity preparation is offline, one-time work: its cost is
 * paid before a conversation starts and never appears in the latency budget a candidate feels.
 * That is exactly why it is worth seeing — a cheap enrollment is what makes a pool of warm,
 * pre-prepared identities affordable, and an expensive one is what forces enrollment to be a
 * scheduled job instead of something an operator does while waiting. The number decides an
 * architecture, so it goes where it will be read.
 *
 * Fetched client-side from the runtime on :8000 rather than proxied through a route handler:
 * the console and the runtime are separate processes, and pretending otherwise would hide the
 * failure that actually happens — the API not being up. That case gets its own state below,
 * with the command that fixes it.
 */

import { useCallback, useEffect, useState } from "react";

import { FaceCreate } from "@/components/face-create";
import {
  Button,
  Card,
  CardHeader,
  Cell,
  Chip,
  Empty,
  Field,
  Input,
  Page,
  Row,
  Table,
  type Status,
  num,
} from "@/components/ui";

const API = "http://127.0.0.1:8000";

type FaceStatus = "queued" | "preparing" | "ready" | "failed";

type Face = {
  id: string;
  name: string;
  reference_path: string;
  /** Present only on uploaded faces; a path-created face has no preview frame. */
  thumbnail_path?: string | null;
  source_kind?: string | null;
  duration_seconds?: number | null;
  width?: number | null;
  height?: number | null;
  status: FaceStatus;
  enrollment_ms: number | null;
  frame_count: number | null;
  failure_reason: string | null;
  created_at: string;
  updated_at: string;
};

/**
 * Status colour is meaning, never decoration — see the note in globals.css about keeping the
 * accent out of semantics. `queued` is warn rather than neutral on purpose: it is not a
 * resting state, it is a face that cannot be used until someone presses Prepare.
 */
const TONE: Record<FaceStatus, Status> = {
  queued: "warn",
  preparing: "info",
  ready: "ok",
  failed: "bad",
};

/** Statuses the API will accept a prepare for. Mirrors `PREPARABLE` in faces.py. */
function preparable(status: FaceStatus): boolean {
  return status === "queued" || status === "failed";
}

/**
 * ISO-8601 UTC, trimmed to the minute and left in UTC.
 *
 * Not localised: these timestamps are compared against server logs, and a column silently
 * shifted into the reader's timezone is how two people end up describing the same event an
 * hour apart.
 */
function stamp(iso: string): string {
  return iso.slice(0, 16).replace("T", " ");
}

const NAME_FIELD_ID = "new-face-name";

/**
 * Reached through the DOM id rather than a ref, because `Input` in ui.tsx spreads props onto
 * the element and does not forward one — and adding a primitive to solve a focus call would be
 * a worse trade than one `getElementById`.
 */
function focusNameField(): void {
  const field = document.getElementById(NAME_FIELD_ID);
  field?.scrollIntoView({ block: "center", behavior: "smooth" });
  field?.focus();
}

export default function FacesPage() {
  const [faces, setFaces] = useState<Face[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [referencePath, setReferencePath] = useState("");
  const [formError, setFormError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const load = useCallback(async (signal?: AbortSignal) => {
    try {
      const response = await fetch(`${API}/faces`, { signal, cache: "no-store" });
      if (!response.ok) throw new Error(`GET /faces returned ${response.status}`);
      setFaces((await response.json()) as Face[]);
      setError(null);
    } catch (cause) {
      // An aborted request is a cleanup, not a failure: reporting it would flash "could not
      // reach the runtime" on every mount, because React mounts effects twice in development.
      if (signal?.aborted) return;
      // Otherwise the message is kept and shown. "Failed to fetch" says nothing on its own,
      // which is why the copy around it names the likely cause instead of leaving it guessed.
      setError(cause instanceof Error ? cause.message : String(cause));
      setFaces(null);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    // Awaited inside an inner function rather than called from the effect body, so every
    // state update lands in a later microtask instead of cascading a render synchronously.
    const run = async () => {
      await load(controller.signal);
    };
    void run();
    return () => controller.abort();
  }, [load]);

  /**
   * Delete the face, its media and any prepared identity.
   *
   * A 409 means an agent still references it, and the runtime's own message names which — more
   * useful than "could not delete", and the reason this surfaces the detail rather than a generic
   * failure.
   */
  const remove = useCallback(
    async (id: string) => {
      setBusyId(id);
      try {
        const response = await fetch(`${API}/faces/${id}`, { method: "DELETE" });
        if (!response.ok && response.status !== 404) {
          const payload = (await response.json().catch(() => null)) as { detail?: unknown } | null;
          setError(
            typeof payload?.detail === "string"
              ? payload.detail
              : `the runtime answered ${response.status}`,
          );
          return;
        }
        load();
      } catch {
        setError("could not reach the runtime to delete that face");
      } finally {
        setBusyId(null);
      }
    },
    [load],
  );

  async function create() {
    setSaving(true);
    setFormError(null);
    try {
      const response = await fetch(`${API}/faces`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, reference_path: referencePath }),
      });
      if (!response.ok) {
        throw new Error(
          response.status === 422
            ? "A name and a reference path are both required."
            : `POST /faces returned ${response.status}`,
        );
      }
      setName("");
      setReferencePath("");
      await load();
    } catch (cause) {
      setFormError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setSaving(false);
    }
  }

  async function prepare(id: string) {
    setBusyId(id);
    try {
      // A failed enrollment is a 200 with `status: "failed"` on the record, so there is
      // nothing to branch on here: reloading shows the outcome either way, in the row.
      const response = await fetch(`${API}/faces/${id}/prepare`, { method: "POST" });
      if (!response.ok) throw new Error(`prepare returned ${response.status}`);
      await load();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <Page
      title="Faces"
      lede="A face is a reference clip or image plus the identity artifact prepared from it.
        Preparation is offline, one-time work, so its cost is listed here rather than hidden:
        it is what decides whether identities can be warmed in a pool or have to be enrolled
        as a scheduled job."
    >
      {/* Upload first: it is how a face is actually made now. The path-based form below stays for
          a reference already on the API's disk -- a scripted demo, or a clip too large to push
          through a browser -- and is collapsed, because it is the rarer case. */}
      <FaceCreate onCreated={() => void load()} />

      <Card>
        <CardHeader
          title="Prepared personas"
          hint="Enrollment runs against the GPU-free stub renderer, so the measured cost is the
            stub's, not a real model's."
        />

        {error ? (
          <div className="px-5 py-10 text-center">
            <p className="text-[13.5px] font-medium text-bad">Could not reach the runtime</p>
            <p className="mx-auto mt-2 max-w-md text-[12.5px] leading-relaxed text-ink-mid">
              {error}. The console reads faces from the API process on :8000 — start it with{" "}
              <span className="font-mono text-[11.5px] text-ink">uvicorn avatar.server:app</span>{" "}
              from <span className="font-mono text-[11.5px] text-ink">apps/api</span>.
            </p>
            <div className="mt-5 flex justify-center">
              <Button onClick={() => void load()}>Retry</Button>
            </div>
          </div>
        ) : faces === null ? (
          <p className="px-5 py-10 text-center text-[12.5px] text-ink-mid">Loading faces…</p>
        ) : faces.length === 0 ? (
          <Empty
            title="No faces yet"
            action={<Button onClick={focusNameField}>Add a face</Button>}
          >
            An agent needs a face before it can render anyone. Point one at a reference clip on
            the machine running the API, then press Prepare to enroll it — the enrollment cost
            shows up here once it has actually been measured.
          </Empty>
        ) : (
          <Table head={["", "Name", "Status", num("Enrollment"), num("Frames"), "Updated (UTC)", ""]}>
            {faces.map((face) => (
              <Row key={face.id}>
                <Cell>
                  {/* The preview, or a placeholder that says why there is none. A face with no
                      thumbnail was created from a server-side path, which is a real and supported
                      case rather than a broken one. */}
                  {face.thumbnail_path ? (
                    /* eslint-disable-next-line @next/next/no-img-element --
                       served by the runtime on another origin, which next/image would need a
                       remotePatterns entry for; and a 480px JPEG has nothing to gain from an
                       optimisation pipeline. */
                    <img
                      src={`${API}/faces/${face.id}/thumbnail`}
                      alt={`Preview frame of ${face.name}`}
                      className="h-10 w-16 rounded border border-hair object-cover"
                    />
                  ) : (
                    <span
                      title="Created from a server-side path, so there is no preview frame"
                      className="grid h-10 w-16 place-items-center rounded border border-dashed border-hair text-[10px] text-ink-low"
                    >
                      no preview
                    </span>
                  )}
                </Cell>
                <Cell>
                  <span className="block text-ink">{face.name}</span>
                  <span className="mt-0.5 block font-mono text-[11.5px] text-ink-low">
                    {face.reference_path}
                  </span>
                  {/*
                    Only shown while the status is `failed`. The API keeps the last failure's
                    reason on the record after a successful retry — the store's merge cannot
                    unset it — so status is the authority on whether it still applies.
                  */}
                  {face.status === "failed" && face.failure_reason ? (
                    <span className="mt-1 block text-[11.5px] text-bad">
                      {face.failure_reason}
                    </span>
                  ) : null}
                </Cell>
                <Cell>
                  <Chip status={TONE[face.status]}>{face.status}</Chip>
                </Cell>
                {/* An em dash, not a zero: nothing has measured this yet, and 0 ms is a
                    measurement. */}
                <Cell mono right>
                  {face.enrollment_ms === null ? (
                    <span className="text-ink-low">—</span>
                  ) : (
                    `${face.enrollment_ms} ms`
                  )}
                </Cell>
                <Cell mono right>
                  {face.frame_count === null ? (
                    <span className="text-ink-low">—</span>
                  ) : (
                    face.frame_count
                  )}
                </Cell>
                <Cell mono dim>
                  {stamp(face.updated_at)}
                </Cell>
                <Cell right>
                  {/* Delete sits beside Prepare rather than behind a menu. A face is the largest
                      thing this product stores -- a clip, a thumbnail, and about a gigabyte of
                      prepared latents once enrolled -- and an operator who uploaded the wrong
                      person had no way to remove them from here at all. */}
                  <span className="flex justify-end gap-1.5">
                    {preparable(face.status) ? (
                      <Button
                        variant="primary"
                        disabled={busyId === face.id}
                        onClick={() => void prepare(face.id)}
                      >
                        {busyId === face.id ? "Preparing…" : "Prepare"}
                      </Button>
                    ) : null}
                    <Button
                      variant="danger"
                      disabled={busyId === face.id}
                      onClick={() => void remove(face.id)}
                    >
                      Delete
                    </Button>
                  </span>
                </Cell>
              </Row>
            ))}
          </Table>
        )}
      </Card>

      {/* Behind a disclosure, because it is now the rarer path and a second create form sitting
          open next to the uploader is an invitation to type a path that does not exist on the API
          host -- which fails at prepare time, not here, with nothing on screen explaining why. */}
      <details className="group rounded-xl border border-hair bg-glass">
        <summary className="cursor-pointer list-none px-5 py-4">
          <span className="text-[13.5px] font-medium text-ink">
            Point at a reference already on the API host
          </span>
          <span className="mt-1 block text-[12px] text-ink-mid">
            For a clip too large to push through a browser, or a scripted demo. The path is
            resolved by the API process, not by this browser — so it must exist on that machine.
          </span>
        </summary>
        <div className="grid gap-4 border-t border-hair px-5 py-5 sm:grid-cols-2">
          <Field label="Name" hint="How this persona is identified in the agent editor.">
            <Input
              id={NAME_FIELD_ID}
              value={name}
              placeholder="Ada"
              onChange={(event) => setName(event.target.value)}
            />
          </Field>
          <Field label="Reference path" hint="A clip or image on the machine running the API.">
            <Input
              value={referencePath}
              placeholder="media/ada.mp4"
              onChange={(event) => setReferencePath(event.target.value)}
            />
          </Field>
        </div>
        <div className="flex items-center gap-4 border-t border-hair px-5 py-4">
          <Button
            variant="primary"
            disabled={saving || name.trim() === "" || referencePath.trim() === ""}
            onClick={() => void create()}
          >
            {saving ? "Creating…" : "Create face"}
          </Button>
          {/* A new face is created queued, never enrolled: nothing may claim a measurement
              that no run produced. */}
          <p className="text-[11.5px] text-ink-low">
            Created queued. Enrollment happens when you press Prepare.
          </p>
          {formError ? <p className="text-[11.5px] text-bad">{formError}</p> : null}
        </div>
      </details>
    </Page>
  );
}
