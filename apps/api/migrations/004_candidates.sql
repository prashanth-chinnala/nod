-- Candidates, and the link from an interview to the person being interviewed.
--
-- **What was missing.** A session named an agent and nothing else. So the system could say "this
-- interview used the senior backend interviewer" and could not say who sat on the other side of
-- it. Every report was an assessment of an anonymous transcript, which is fine for proving the
-- mechanics and useless for actually hiring: you cannot compare two candidates, you cannot give
-- one person two interviews with different agents, and you cannot tell an interviewer anything
-- about who they are talking to.
--
-- **Why the resume text is a column and not only a file.** The file is the record of what the
-- operator uploaded and has to be kept. The extracted text is what the interviewer is briefed
-- from, and it is derived — a different extractor, or a better one, produces different text from
-- the same PDF. Storing both means a re-extraction is possible without asking the operator to
-- upload again, and it means the briefing an interview actually used is recoverable after the
-- extractor changes. Storing only the file would make the second impossible.
--
-- **Why `sessions.candidate_id` is nullable and `on delete set null`.** A session with no
-- candidate is a real and useful state: every demo session, every smoke test, and every interview
-- run before this migration existed. And deleting a candidate must not delete the interviews they
-- sat — the transcript is a record of something that happened, and a hiring process that destroys
-- its own evidence when a row is tidied up is worse than one that keeps an orphan. `doc` keeps the
-- id it always had, so the reader can still show "candidate deleted" rather than nothing at all.
-- This is the same choice `sessions.agent_id` already makes, for the same reason.

create table candidates (
    id         text primary key,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    -- Typed because the console lists and sorts on them, and a jsonb lookup for a list view is
    -- how a fast page becomes a slow one at a few thousand rows.
    name       text not null default '',
    email      text not null default '',
    role       text not null default '',
    status     text not null default 'new',

    doc        jsonb not null default '{}'::jsonb
);

create index candidates_created_at_idx on candidates (created_at desc);
create index candidates_email_idx      on candidates (email) where email <> '';

comment on column candidates.status is
    'new | invited | interviewed | reviewed. Advanced by the API as interviews are created and '
    'completed, not by the operator typing a state name.';

comment on column candidates.doc is
    'Everything untyped: resume_path, resume_filename, resume_text, resume_chars, notes, and the '
    'agent_id an operator chose to interview them with.';

alter table sessions
    add column candidate_id text references candidates (id) on delete set null;

create index sessions_candidate_id_idx
    on sessions (candidate_id) where candidate_id is not null;

comment on column sessions.candidate_id is
    'Who was interviewed. Null is legitimate: demo sessions, smoke tests, and every interview '
    'that predates this column. Set null rather than cascade on delete, because a transcript is '
    'evidence of something that happened and should outlive the tidying of a row.';
