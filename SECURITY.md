# Security and data handling

What this system holds, who can reach it, and what is deliberately not protected yet. Written so
that the gaps are decisions on record rather than things nobody looked at.

---

## 1. The posture, stated plainly

**There is no authentication anywhere.** Not on the API, not on the console, not on the assistant.

- The candidate link (`/interview/<session_id>`) **is not a credential.** Anyone with the id can
  join that session; anyone who can guess one can join that one.
- The console has no login. Every agent, rubric, face and transcript is readable and writable by
  anyone who can reach port 3000.
- The assistant has no login and **will read any transcript in the store** to answer a question.

This is a development posture. It is stated here rather than implied by absence, because the second
kind of gap is the one that ships.

---

## 2. What the system holds

| Data | Where | Sensitivity |
|---|---|---|
| Interview transcripts | store (JSON files or Postgres) | verbatim, what a real candidate said |
| Reference clips and photos | `media/` on disk | **a real person's face** |
| Thumbnails | `media/` | face |
| Recordings | `recordings/` | audio and video of an interview |
| Scorecards | store | a judgement about a person, with quotes |
| Provider credentials | `.env*`, never committed | keys |

Prepared identity artifacts — cycled frames, masks and VAE latents — are held **in memory only** and
lost on restart. They are derived data, and not persisting them is why a restart re-enrolls.

### Real faces change the calculation

While references were vendor demo assets, the absence of auth was a development convenience with a
low ceiling. A store of employees' or candidates' faces and voices is **biometric data**, and in
several jurisdictions it is a special category with its own consent, retention and deletion
obligations.

That is not a reason to stop; it is the reason authentication stops being a "later" item the moment
the first real person's face is uploaded. It is also the reason this system is self-hosted — see §4.

---

## 3. What is protected

**Secrets never reach git.** Every `.env*` is gitignored with **no exemption**, deliberately: a
committed template is one `git add -f` away from a committed key, so the README's configuration table
documents each variable instead. `/config` reports which env files were *read* and which variable
*names* came from them — never a value.

**Transcripts never reach git.** `data/`, `recordings/` and `media/` are all gitignored, and the
`data/` rule is deliberately **unanchored**. An earlier anchored `apps/api/data/` rule did not match
a store that landed at the repository root because `AVATAR_DATA_DIR` defaults to a *relative* path —
and a real session transcript was committed. Matching the directory at any depth is the only version
that holds.

**Media is not in the database.** Files on disk, so a 20 MB clip is not read and rewritten on every
unrelated patch to the same row — and so deleting a person's media is a file operation, not a
migration.

**Path traversal.** The store refuses ids containing separators. Uploads keep only the *suffix* of
the uploader's filename and are written under a generated name; everything else in a supplied
filename is a path the uploader chose on our disk.

**Uploads are validated by reading them.** `ffprobe`, not the extension or the browser's content
type, both of which are claims made by the uploader. A `.mp4` containing something else is the case
that matters: it passes every name-based check and fails inside the renderer hours later.

**Model output is never rendered as HTML.** The console's markdown renderer emits React elements
built from plain strings only. Model output frequently contains verbatim candidate transcript, and
handing that to an HTML renderer would make the safety of a page displaying interview records depend
on a sanitiser's configuration.

**Downloaded weights are verified, not trusted.** HTML-page detection, a size floor, and a
container-format check on every artifact. One checkpoint is re-saved from torch's legacy tar format
with `weights_only=False` — a bounded, deliberate exception, scoped to one file from
`download.pytorch.org` over TLS, taken specifically to avoid relaxing `torch.load`'s default for
every checkpoint the process ever reads.

---

## 4. Where data goes

Self-hosting is the point, and this is the honest accounting of where it is not yet true.

| Component | Runs | Data that leaves |
|---|---|---|
| Renderer (MuseTalk) | **on your GPU** | nothing |
| Store (JSON or Postgres) | **on your host** | nothing |
| Console, assistant | **on your host** | nothing |
| LiveKit SFU, egress, Redis | **your Docker** | nothing |
| Speech to text | Deepgram | **candidate audio** |
| Text to speech | Deepgram | interviewer text |
| Language model | an OpenAI-compatible endpoint | **transcript and prompt** |

So today the face never leaves, and the **conversation does**. Three of the four hosted paths have
self-hosted replacements that need no code change beyond a base URL — the LLM adapter speaks the
OpenAI wire format precisely so Ollama, vLLM or LM Studio can take over. STT and TTS would need real
replacements (Whisper for one, XTTS-v2 or F5-TTS for the other), which is genuine work, not
configuration.

---

## 5. Known gaps, in the order I would close them

1. **Authentication and authorisation.** An operator login for the console and assistant; a signed,
   expiring, single-use candidate link. Everything else on this list is secondary.
2. **Consent and retention for reference media.** Who uploaded this face, on whose authority, and
   when is it deleted? There is no consent record and no retention policy today.
3. **Deletion that actually deletes.** Removing a face should remove the clip, the thumbnail, the
   generated still-clip and any cached identity. Currently only the record goes.
4. **Self-hosted STT and TTS**, which is what makes "no candidate data leaves" true rather than
   aspirational.
5. **Rate limiting and upload quotas.** `MAX_UPLOAD_BYTES` is 200 MB per file and there is no
   per-caller limit, so filling a disk is easy.
6. **Audit trail.** The assistant's writes are attributed, but nothing else records who changed a
   rubric or deleted a session.
7. **A stuck-job reaper.** A crash mid-enrollment leaves a row claiming `preparing` forever, which
   `PREPARABLE` will not accept again. Availability rather than security, but it is the same missing
   piece — a real job queue.

---

## 6. Reporting

This is pre-production software with no authentication by design. If you find something, open an
issue — there is no security contact process yet, and pretending otherwise would be worse than
saying so.
