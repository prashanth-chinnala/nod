-- A turn the candidate never answered.
--
-- The silence watchdog re-prompts after 12s of nothing, and until now that re-prompt was
-- invisible in the record. `on_idle_tick` appended the marker to conversation history and called
-- `_begin_turn` directly -- correctly, because the re-prompt happens from IDLE and the
-- end-of-turn guard requires LISTENING -- but the server builds turns from the `heard` telemetry
-- event, and no `heard` event was ever emitted. So the interviewer asked a question that reached
-- the candidate, was spoken aloud, and appears nowhere: the stored transcript jumped straight
-- from the previous answer to the next one, and a reviewer reading it could not tell that
-- twelve seconds of silence and a nudge had happened in between.
--
-- Why a column and not an empty `heard`. `transcribed = false` with an empty `heard` already
-- means something else and something important: speech was detected and the transcriber
-- returned nothing. That is a broken-STT signal the API docstring specifically calls out.
-- Overloading it with "there was no speech at all" would make a configuration fault and a quiet
-- candidate indistinguishable, which is the exact confusion this project keeps paying to avoid.
--
-- Defaults to false, so every turn already stored keeps its current meaning.
alter table turns add column silent boolean not null default false;

comment on column turns.silent is
    'True when this turn was opened by the silence watchdog rather than by the candidate '
    'speaking. `heard` is empty because nothing was said, not because transcription failed.';
