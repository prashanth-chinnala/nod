-- Voices: a recording of a person speaking, cloned at synthesis time.
--
-- Its own table rather than a column on `agents`, matching every other reference the console
-- manages. One recording is reused across agents, and it needs a place to be listed, auditioned
-- and deleted.
--
-- Deliberately thinner than `faces`. A face carries `status`, `enrollment_ms` and `frame_count`
-- because preparing one is expensive, offline, and can fail -- so the table has to model a job.
-- Cloning is zero-shot: the reference is encoded into a speaker embedding at first use and cached
-- in memory, with no artifact to build. There is nothing to queue, so `status` lives in `doc` and
-- carries only a failure.

create table voices (
    id         text        primary key,
    name       text        not null,
    doc        jsonb       not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint voices_doc_id_agrees check (doc->>'id' = id)
);

-- `on delete restrict`, matching `face_id`, and for the same reason: an interview conducted in a
-- different voice than the one configured is worse than one that refuses to start. Detaching is an
-- explicit patch to null, not a side effect of deleting the voice out from under an agent.
alter table agents
    add column voice_ref_id text references voices (id) on delete restrict;

create index agents_voice_ref_id_idx on agents (voice_ref_id);
