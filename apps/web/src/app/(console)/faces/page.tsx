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
          <Table head={["Name", "Status", num("Enrollment"), num("Frames"), "Updated (UTC)", ""]}>
            {faces.map((face) => (
              <Row key={face.id}>
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
                  {preparable(face.status) ? (
                    <Button
                      variant="primary"
                      disabled={busyId === face.id}
                      onClick={() => void prepare(face.id)}
                    >
                      {busyId === face.id ? "Preparing…" : "Prepare"}
                    </Button>
                  ) : null}
                </Cell>
              </Row>
            ))}
          </Table>
        )}
      </Card>

      <Card>
        <CardHeader
          title="New face"
          hint="The reference path is resolved by the API process, not by this browser."
        />
        <div className="grid gap-4 px-5 py-5 sm:grid-cols-2">
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
      </Card>
    </Page>
  );
}
