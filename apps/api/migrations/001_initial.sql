-- 001_initial.sql -- the whole schema, in one file, applied with psql.
--
-- There is no migration framework here on purpose. One migration does not need a tool that
-- brings a dependency, a version table and a CLI; `psql -f` is the boring thing that will
-- still work on submission day. When there is a second migration this decision gets revisited,
-- and the revisit is cheap: nothing depends on the absence of a tool.
--
--
-- WHY HYBRID, AND WHAT THE TWO HALVES ARE FOR
--
-- Each table has typed columns for identity and cross-references, and a `doc jsonb` for the
-- rest of the body. The two halves answer two different questions.
--
-- The typed columns exist so the database refuses things the application currently only hopes
-- about. `agents.rubric_id` is a plain string today: delete a rubric an agent still points at
-- and nothing complains until a candidate joins and the session will not start. That is the
-- worst possible moment to learn about it. As a foreign key with ON DELETE RESTRICT the delete
-- fails in the console, in front of the operator who asked for it, while there is still a form
-- open. The constraint is not tidiness -- it moves a failure from the candidate to the operator.
--
-- The `doc` column exists so the routers do not change. Every router in `src/avatar/api/`
-- deliberately returns records exactly as the store wrote them, and says why: a response model
-- would filter reads through one build's idea of the shape, so a field written by a newer
-- deploy would vanish from the console while still sitting on disk, and a partial deploy would
-- look like data loss. Full normalisation would put this file in charge of that shape and break
-- that property on day one -- `tools.parameters_schema` is a user-supplied JSON Schema and
-- `assistant_actions.detail` varies by kind, so neither has columns to normalise into.
--
--
-- WHY ONE TABLE PER COLLECTION, NOT ONE `records` TABLE
--
-- A single `records (collection, id, doc)` table would need no DDL per collection, which is its
-- only advantage, and it would give up the thing this schema is being built for: a foreign key
-- needs a real referenced table. `references records(id)` cannot express "must be a rubric",
-- so `agents.rubric_id` could point at a face and the database would be satisfied. Enforcing
-- kind in a trigger is how that ends up, and a trigger is a worse version of a constraint the
-- engine already has. One table per collection also means each table's columns document that
-- collection, indexes are per collection rather than shared, and a slow query on sessions
-- cannot be a slow query on everything.
--
-- The cost is nine CREATE TABLE statements and a new one per collection added. That is a
-- known, visible, one-line-per-column cost, paid at the moment someone adds a collection.
--
--
-- WHY TURNS BECOME A TABLE AND COVERAGE/SCORING/RECORDING DO NOT
--
-- This is the same test applied to four nested structures, and it gives different answers
-- because their write patterns differ.
--
-- `turns` is appended to, one row at a time, while a conversation is live. In the file store an
-- append reads the whole record, appends to the array and rewrites the file, so appending turn
-- N rewrites all N turns -- O(n^2) over a session, and the longest session in `data/` already
-- holds 17. Worse than the cost: two writers appending at once both read the same array and the
-- second write silently drops the first turn. Today that is safe only by accident of being one
-- process, and `uvicorn --workers 2` ends the accident without an error message. A child table
-- makes an append one INSERT that touches nothing else.
--
-- `coverage`, `scoring` and `recording` are written whole and read whole -- one snapshot
-- replaces the previous one, and every reader wants all of it. Normalising `coverage` into
-- competencies and evidence and signals_hit would turn one write into a delete-and-reinsert
-- across three tables and one read into three joins, to support queries nothing asks. They stay
-- in `doc` because that is the right answer for how they are used, not because nobody got to
-- them; if a report ever needs "every competency rated weak across all sessions", that is when
-- this changes.
--
--
-- THE CONTRACT BETWEEN A TYPED COLUMN AND THE DOC
--
-- Most data is in both places, and the rules for that are not symmetric. Read them before
-- implementing the store.
--
--   * `doc` is what the API returns. The reader hands back `doc` as-is, plus `turns` assembled
--     from the child table. It does not overlay the typed columns onto it.
--   * A typed column is the write-side mirror, and must be set in the SAME statement as the
--     `doc || patch` merge that changes its key -- otherwise a crash between two statements
--     leaves a column and a body disagreeing about which rubric an agent uses.
--   * Two deliberate exceptions, both on sessions, both explained at the table.
--
-- Nothing here has authentication in front of it, and this file cannot add any. Row-level
-- security would be the place; there are no principals to write policies against yet.

begin;

-- Deliberately atomic. A half-applied schema -- five tables and no foreign keys -- is harder
-- to diagnose than no schema, because it looks like it worked.


-- ---------------------------------------------------------------------------
-- Leaf resources: referenced by agents, referencing nothing themselves.
-- ---------------------------------------------------------------------------
--
-- `name text not null` on all five is safe because every create model requires a non-blank name
-- and validates it (`_stripped_name` and friends refuse a single space, which renders as a blank
-- row nobody can find). The column is here so list views can sort and search in SQL instead of
-- pulling every doc into Python to read one key.
--
-- `check (doc->>'id' = id)` on every table: the store writes the id into the body as well as
-- the key, and a row where those disagree would serve a record under an id it does not claim --
-- fetch by one id, get a body saying another. The constraint should never fire: no router's
-- patch model has an `id` field, so a patch cannot carry one. A dormant constraint is what a
-- correct invariant looks like.

create table faces (
    id         text        primary key,
    name       text        not null,
    doc        jsonb       not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint faces_doc_id_agrees check (doc->>'id' = id)
);

create table rubrics (
    -- `competencies[]` stays in doc and stays ordered, because the order is the interview's
    -- running order -- a JSON array preserves it for free, where a child table would need a
    -- position column and every reorder would rewrite it.
    id         text        primary key,
    name       text        not null,
    doc        jsonb       not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint rubrics_doc_id_agrees check (doc->>'id' = id)
);

create table guardrails (
    id         text        primary key,
    name       text        not null,
    doc        jsonb       not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint guardrails_doc_id_agrees check (doc->>'id' = id)
);

create table pronunciations (
    -- `entries[]` (term -> say) stays in doc: written whole by the console, read whole by the
    -- TTS adapter, never queried by term.
    id         text        primary key,
    name       text        not null,
    doc        jsonb       not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint pronunciations_doc_id_agrees check (doc->>'id' = id)
);

create table knowledge (
    -- `documents[]` and `chunks[]` stay in doc for now, and this is the one place where that
    -- will stop being true first: chunks are the thing retrieval scans, and the moment
    -- retrieval wants a vector index they need to be rows. Not today -- today retrieval is
    -- in-process over a few thousand characters.
    id         text        primary key,
    name       text        not null,
    doc        jsonb       not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint knowledge_doc_id_agrees check (doc->>'id' = id)
);

create table tools (
    -- `parameters_schema` is a JSON Schema supplied by whoever registers the tool. It is
    -- genuinely schemaless and there is nothing to normalise it into.
    id         text        primary key,
    name       text        not null,
    doc        jsonb       not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint tools_doc_id_agrees check (doc->>'id' = id)
);


-- ---------------------------------------------------------------------------
-- agents -- the object every other resource hangs off.
-- ---------------------------------------------------------------------------
--
-- Four single-valued references become nullable foreign keys with ON DELETE RESTRICT. Nullable
-- because an agent with no rubric is a legal, useful intermediate state (the console lets you
-- name an agent before you have written its rubric), and because clearing one back to null is
-- something the pickers do -- the file store was fixed specifically so that `{"rubric_id":
-- null}` detaches rather than being ignored, and `doc || '{"rubric_id": null}'` preserves that
-- exactly, since jsonb concatenation sets a key to null rather than dropping it.
--
-- RESTRICT rather than SET NULL: an agent that silently loses its guardrail is an agent running
-- an interview with no content policy, and it would keep working, which is why nobody would
-- notice. Failing the delete makes the operator detach it on purpose or delete the agent.

create table agents (
    id                text        primary key,
    name              text        not null,
    face_id           text        references faces (id)          on delete restrict,
    rubric_id         text        references rubrics (id)        on delete restrict,
    guardrail_id      text        references guardrails (id)     on delete restrict,
    pronunciation_id  text        references pronunciations (id) on delete restrict,
    doc               jsonb       not null,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now(),

    constraint agents_doc_id_agrees check (doc->>'id' = id),

    -- The hysteresis invariant, as a backstop and nothing more. `release_probability` must sit
    -- strictly below `onset_probability`; the gap between them is what stops the probability dip
    -- inside an ordinary word from ending the candidate's turn mid-sentence. `TurnDetector`
    -- refuses an inverted pair at construction, which is when a candidate connects -- too late.
    -- `AgentCreate` refuses it at write time, which is the right place. This is the third place,
    -- and it earns its keep only because the failure it prevents is a session that dies as
    -- somebody joins it.
    --
    -- Be clear about what it cannot do: if `turn_taking` or either key is absent the expression
    -- is NULL, a CHECK passes on NULL, and the row is accepted. So this catches an inverted pair
    -- and not a missing one. Treating it as complete coverage would be a mistake; `AgentCreate`
    -- is still the authority on a well-formed policy.
    constraint agents_hysteresis_not_inverted check (
        (doc->'turn_taking'->>'release_probability')::double precision
        < (doc->'turn_taking'->>'onset_probability')::double precision
    )
);

-- Postgres does not index the referencing side of a foreign key for you, and every RESTRICT
-- check on a delete scans that side looking for referrers. Without these, deleting a rubric
-- sequentially scans agents -- fine at five agents, and a habit worth not forming. They also
-- serve the query the console needs before offering a delete: "what still uses this?"
create index agents_face_id_idx          on agents (face_id)          where face_id is not null;
create index agents_rubric_id_idx        on agents (rubric_id)        where rubric_id is not null;
create index agents_guardrail_id_idx     on agents (guardrail_id)     where guardrail_id is not null;
create index agents_pronunciation_id_idx on agents (pronunciation_id) where pronunciation_id is not null;
create index agents_created_at_idx       on agents (created_at desc);


-- ---------------------------------------------------------------------------
-- Many-to-many: an agent's knowledge bases and its tools.
-- ---------------------------------------------------------------------------
--
-- CASCADE on the agent side, RESTRICT on the resource side, and the asymmetry is the whole
-- design. Deleting an agent should take its own attachment rows with it -- they describe the
-- agent and mean nothing without it. Deleting a knowledge base an agent still uses should fail,
-- for the same reason as the single-valued references above: an interviewer that quietly loses
-- its source material keeps answering, from nothing.
--
-- No `position` column, deliberately. `doc.knowledge_base_ids` remains the ordering authority
-- because the reader returns `doc` as-is, so a position here would be a second copy of the same
-- ordering with its own opportunity to drift. These tables are write-side integrity only: they
-- exist so the foreign key is real. The cost is that they must be rewritten inside the same
-- transaction as the `doc || patch` that changes the array -- a single statement cannot do both,
-- which is the one place rule 7's "one statement" has to become "one transaction".

create table agent_knowledge_bases (
    agent_id     text not null references agents (id)    on delete cascade,
    knowledge_id text not null references knowledge (id) on delete restrict,
    primary key (agent_id, knowledge_id)
);

-- The primary key indexes (agent_id, knowledge_id), which covers "this agent's bases" but not
-- "who uses this base" -- the direction a RESTRICT check walks on delete.
create index agent_knowledge_bases_knowledge_id_idx on agent_knowledge_bases (knowledge_id);

create table agent_tools (
    agent_id text not null references agents (id) on delete cascade,
    tool_id  text not null references tools (id)  on delete restrict,
    primary key (agent_id, tool_id)
);

create index agent_tools_tool_id_idx on agent_tools (tool_id);


-- ---------------------------------------------------------------------------
-- sessions -- the evidence. Two rules are broken here on purpose.
-- ---------------------------------------------------------------------------
--
-- FIRST EXCEPTION: `agent_id` is ON DELETE SET NULL, not RESTRICT.
--
-- `delete_agent` refuses to cascade and states why: erasing a session's reference to keep the
-- data tidy would rewrite history, and a transcript that no longer says which agent produced it
-- is worth less than a dangling id. RESTRICT would preserve integrity by making an agent
-- undeletable for as long as any transcript mentions it, which in practice means for ever --
-- sessions are append-only and never pruned. SET NULL keeps the delete possible and keeps the
-- transcript.
--
-- And here is the part that looks like a bug and is not: after that SET NULL the column is null
-- while `doc->>'agent_id'` still holds the deleted id. That disagreement is the point. The
-- column is a foreign key and must be null-or-valid; the doc is the transcript's own record of
-- what ran it, and that is exactly the dangling id `delete_agent` chose to keep. So the reader
-- must NOT overlay this column onto the doc. It is the single place in this schema where the
-- column is not the mirror of the key.
--
-- `agent_name` is denormalised for the same reason: a session list that shows "(deleted agent)"
-- instead of a name is a report that got less useful because somebody tidied up. It is
-- populated at insert from `agents.name` and re-set by any patch that would change it, and it
-- survives the delete.
--
-- Do not project `agent_name` into the returned record. The file store never wrote that key,
-- 666 tests were written against what the file store returns, and a key that appears only on
-- one backend is how two backends stop being interchangeable. It is here to be queried.
--
-- SECOND EXCEPTION: `ended_at` is a typed column as well as a doc key, which is one more typed
-- column than "identity and cross-references" allows.
--
-- It earns it by making the append guard a single statement. `append_turn` must reject a turn
-- arriving on an ended session -- a late turn from a socket that already closed would land after
-- `ended_at` and the record would claim a conversation continued past its own end. Reading the
-- doc to check, then inserting, is a read-modify-write with the race in the middle: a close and
-- an append arriving together both pass the check. As a column the insert becomes
-- `... select ... from sessions where id = %s and ended_at is null`, and the engine decides.
-- Closing that race is the reason the turns table exists at all; leaving one in the guard would
-- be an odd place to stop. The cost is a second copy of the end time, subject to the usual rule:
-- the patch that sets `doc.ended_at` sets this column in the same statement.

create table sessions (
    id         text        primary key,
    agent_id   text        references agents (id) on delete set null,
    agent_name text,
    ended_at   timestamptz,
    doc        jsonb       not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),

    constraint sessions_doc_id_agrees check (doc->>'id' = id),

    -- Turns live in the `turns` table and nowhere else. If `doc` could also hold a `turns` key,
    -- the O(n^2) append this schema exists to remove would quietly come back the first time
    -- somebody passed `{"turns": [...]}` to update -- and it would still work, just slowly and
    -- with two disagreeing copies. This makes that attempt an error instead of a regression.
    -- `coverage`, `scoring` and `recording` are in `doc` and stay there; see the header.
    constraint sessions_turns_not_in_doc check (not (doc ? 'turns'))
);

create index sessions_agent_id_idx   on sessions (agent_id) where agent_id is not null;
create index sessions_created_at_idx on sessions (created_at desc);


-- ---------------------------------------------------------------------------
-- turns -- one row per exchange, appended.
-- ---------------------------------------------------------------------------
--
-- Fully typed, with no `doc` column, and that is only safe because `Turn` is `extra="forbid"`:
-- the shape is closed at the API boundary, so there is no unknown field to lose. Every other
-- table gets a doc precisely because its shape is not closed.
--
-- THE KEY IS (session_id, seq), NOT (session_id, epoch), AND THIS MATTERS.
--
-- `epoch` is the state machine's turn counter, and it does not advance for a turn the machine
-- refuses -- a barge-in during THINKING leaves the epoch where it was. So two rows in one
-- session can carry the same epoch, legitimately, and an epoch-keyed table would reject the
-- second one: the interrupted turns would be the ones lost, and those are the ones most worth
-- counting. `seq` is arrival order and is unique by construction.
--
-- Who allocates `seq`: the insert, from `coalesce(max(seq) + 1, 0)` over the session. Two
-- concurrent appends can compute the same value, and then the primary key rejects one with a
-- unique violation for the store to retry. That is the improvement -- the file store's version
-- of the same race drops a turn and returns 201.
--
-- Timings are `double precision`, matching the `float` on the Pydantic model. Not `numeric`:
-- psycopg returns numeric as `Decimal`, which `json.dumps` refuses, so a perfectly stored
-- latency would come back as a 500 from the router. ON DELETE CASCADE because a turn without
-- its session is not evidence of anything.

create table turns (
    session_id         text    not null references sessions (id) on delete cascade,
    seq                integer not null,
    epoch              integer not null,
    heard              text    not null default '',
    said               text    not null default '',
    transcribed        boolean not null default false,
    interrupted        boolean not null default false,

    -- All four nullable, and that is a real state rather than missing data: a turn cut off
    -- during THINKING has an LLM timing and nothing after it. Rejecting those would discard the
    -- interrupted turns, which are the interesting ones.
    llm_ttft_ms        double precision,
    tts_first_audio_ms double precision,
    first_frame_ms     double precision,
    perceived_total_ms double precision,

    primary key (session_id, seq),

    -- Mirrors the `ge=0` bounds on the Pydantic model, so a direct SQL writer -- a backfill, a
    -- psql session -- cannot store a negative latency the API would have refused.
    constraint turns_seq_non_negative   check (seq >= 0),
    constraint turns_epoch_non_negative check (epoch >= 0),
    constraint turns_timings_non_negative check (
        coalesce(llm_ttft_ms, 0)        >= 0 and
        coalesce(tts_first_audio_ms, 0) >= 0 and
        coalesce(first_frame_ms, 0)     >= 0 and
        coalesce(perceived_total_ms, 0) >= 0
    )
);

-- The primary key already orders by (session_id, seq), which is the read every session detail
-- view does, so there is no second index here. Worth stating so its absence reads as a decision.


-- ---------------------------------------------------------------------------
-- assistant_actions -- the audit trail.
-- ---------------------------------------------------------------------------
--
-- `target` is typed and indexed but is NOT a foreign key, and that is not an oversight. It is
-- polymorphic by design: `audit.py` documents it as "a session, a rubric, a competency" -- and a
-- competency id is not a record id at all, it is a slug inside a rubric's doc. There is no one
-- table to reference. A foreign key would therefore have to be preceded by narrowing what the
-- assistant is allowed to comment on, which is a product decision and not a schema one.
--
-- The cost, named: a target can dangle, and after a session is deleted its actions still claim
-- to be about it. `history(target)` filters rather than joins, so those surface as actions about
-- something absent rather than as an error. The index is there because `history` is the only
-- query this collection has, and it currently pulls every action into Python to run it.
--
-- `kind`, `status` and `detail` stay in doc. `kind` and `status` are closed `Literal` sets in
-- `audit.py`, and that enumeration is deliberately the answer to "what can this thing do to my
-- data" -- putting a copy in a CHECK here would mean a schema migration to add an assistant
-- capability, in exchange for a rule the router already enforces. `detail` varies by kind and is
-- genuinely schemaless.

create table assistant_actions (
    id         text        primary key,
    target     text        not null,
    doc        jsonb       not null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint assistant_actions_doc_id_agrees check (doc->>'id' = id)
);

create index assistant_actions_target_idx     on assistant_actions (target);
create index assistant_actions_created_at_idx on assistant_actions (created_at desc);

commit;
