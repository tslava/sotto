-- sotto schema: the stage 1 tables (PLAN.md) plus the stage 6 tables.
-- Applied whole through executescript when PRAGMA user_version == 0.

CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    tg_message_id INTEGER,
    chat_id INTEGER,
    user_id INTEGER,
    display_name TEXT,
    text TEXT,                       -- media without text: the pack's placeholders ([photo], [sticker], ...)
    reply_to_tg_message_id INTEGER,
    is_bot INTEGER,
    created_at INTEGER
);

CREATE TABLE bot_replies (
    id INTEGER PRIMARY KEY,
    tg_message_id INTEGER,           -- its own message: needed for reactions and replies to it
    reply_to_tg_message_id INTEGER,  -- NULL when it was posted to the stream
    trigger TEXT,                    -- mention | reply | name | ambient | places | morning | spontaneous
    trigger_tg_message_id INTEGER,   -- what triggered it
    text TEXT,
    prompt_version INTEGER,
    few_shot_version INTEGER,
    delay_sec INTEGER,               -- actual pause from the trigger to sending
    created_at INTEGER
);

CREATE TABLE state (
    key TEXT PRIMARY KEY,
    -- panic, stop_until, topic_cooldown_until,
    -- last_ambient_at, ambient_count:<YYYY-MM-DD>,
    -- last_mention_reply_at, last_mention_reply_at:<user_id>, mention_count:<YYYY-MM-DD>,
    -- spontaneous_count:<YYYY-Www>,
    -- llm_calls:<YYYY-MM-DD>, llm_spent_usd:<YYYY-MM-DD>, llm_error_streak, llm_circuit_until
    -- counters with a date in the key need no reset at midnight; retention removes old keys
    value TEXT
);

CREATE TABLE night_queue (           -- addresses that arrived in quiet_window; survive a restart
    id INTEGER PRIMARY KEY,
    tg_message_id INTEGER,
    user_id INTEGER,
    display_name TEXT,
    text TEXT,
    created_at INTEGER,
    answered_at INTEGER              -- NULL until answered; the morning job sets it in one update
);

CREATE TABLE pending_replies (       -- delayed replies to addresses; survive a restart
    id INTEGER PRIMARY KEY,
    trigger_tg_message_id INTEGER,
    user_id INTEGER,
    trigger TEXT,                    -- mention | reply | name
    due_at INTEGER,
    created_at INTEGER,
    done_at INTEGER                  -- NULL until sent or cancelled
);

CREATE TABLE muted_users (
    user_id INTEGER PRIMARY KEY,
    display_name TEXT,
    muted_by INTEGER,
    created_at INTEGER
);

CREATE TABLE config_overrides (      -- /set; takes priority over config.yaml
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at INTEGER
);

CREATE TABLE places (
    place_id TEXT PRIMARY KEY,
    name TEXT,
    district TEXT,
    category TEXT,         -- craft | cheap | outskirts
    rating REAL,
    reviews INTEGER,
    price_level INTEGER,
    quiet INTEGER,         -- set by hand, Google does not know it
    fact TEXT,             -- one condensed fact from the reviews
    operational INTEGER,
    refreshed_at INTEGER
);

CREATE TABLE filter_log (            -- every refusal at any stage; the source for /why
    id INTEGER PRIMARY KEY,
    trigger_tg_message_id INTEGER,
    candidate_text TEXT,             -- the bot's reply at the output filter; NULL at the gate
    verdict TEXT,                    -- pass | cut
    stage TEXT,                      -- gate | llm | regex | dedup | judge | send
    reason TEXT,                     -- gate:is_bot, gate:night, gate:night_queued, gate:logistics,
                                      -- gate:injection, gate:not_live, gate:cooldown, gate:cap, gate:dice,
                                      -- llm:timeout, llm:http, llm:invalid_json, llm:budget, llm:circuit_open,
                                      -- regex:length, regex:echo, regex:prompt_leak, regex:model_talk,
                                      -- dedup:jaccard, dedup:polish_freq, judge:risky, judge:obeyed,
                                      -- send:restart, send:recheck_stop, send:recheck_night ...
    shadow INTEGER,
    created_at INTEGER
);

-- Stage 6: management from Telegram.

CREATE TABLE config_audit (
    id INTEGER PRIMARY KEY, key TEXT, old_value TEXT, new_value TEXT,
    changed_by INTEGER, created_at INTEGER
);

CREATE TABLE prompt_versions (
    version INTEGER PRIMARY KEY, body TEXT, note TEXT, active INTEGER, created_at INTEGER
);

CREATE TABLE few_shot_versions (
    version INTEGER PRIMARY KEY, body_yaml TEXT, note TEXT, active INTEGER, created_at INTEGER
);

-- The character's life events (/life, CLAUDE.md "life events"): memory, not chat history.
-- retention.py leaves them alone; only /life rm removes them.
CREATE TABLE life_events (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    announced_at INTEGER,
    announced_tg_message_id INTEGER
);

-- Long chat memory (CLAUDE.md, "long chat memory"): summaries of past conversations by
-- period. It outlives the messages themselves: retention.py purges it by its own term
-- (behaviour.chat_memory.keep_days), not by message_retention_days.
CREATE TABLE chat_memory (
    id INTEGER PRIMARY KEY,
    period_start INTEGER NOT NULL,   -- unix, inclusive
    period_end INTEGER NOT NULL,     -- unix, exclusive
    text TEXT NOT NULL,              -- the summary, a few lines
    created_at INTEGER NOT NULL
);

-- Day diary (CLAUDE.md, "day diary"): facts about the character's own doings, pulled from
-- his replies by a cheap model; purged by behaviour.diary.keep_days.
CREATE TABLE self_facts (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,                       -- "went to the sauna", <= 80 characters
    bot_reply_tg_message_id INTEGER,
    created_at INTEGER NOT NULL
);

-- Chat jokes and people's stories (CLAUDE.md, "chat jokes and people's stories"): extracted
-- by the weekly ChatMemorizer; jokes live chat_memory.keep_days, stories callback.max_age_days.
CREATE TABLE chat_jokes (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,                       -- verbatim, <= 80 characters
    created_at INTEGER NOT NULL,
    last_used_at INTEGER,
    uses INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE people_threads (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    text TEXT NOT NULL,                       -- "waiting for an answer after the audition", <= 120 characters
    created_at INTEGER NOT NULL,
    asked_at INTEGER,                         -- NULL: not asked yet
    closed INTEGER NOT NULL DEFAULT 0
);

-- Vocabulary dictionaries (CLAUDE.md, "words from a dictionary"): uploaded by the owner as a
-- file in the private chat; retention leaves them alone.
CREATE TABLE vocab_dicts (
    name TEXT PRIMARY KEY,
    body TEXT NOT NULL,                       -- the dictionary YAML as the owner uploaded it
    enabled INTEGER NOT NULL DEFAULT 1,
    updated_at INTEGER NOT NULL
);

CREATE INDEX idx_messages_chat_created ON messages (chat_id, created_at);
CREATE INDEX idx_messages_tg_message_id ON messages (tg_message_id);
CREATE INDEX idx_bot_replies_created ON bot_replies (created_at);
CREATE INDEX idx_filter_log_created ON filter_log (created_at);
CREATE INDEX idx_night_queue_answered ON night_queue (answered_at);
CREATE INDEX idx_pending_replies_done_due ON pending_replies (done_at, due_at);
CREATE INDEX idx_chat_memory_period_end ON chat_memory (period_end);
CREATE INDEX idx_self_facts_created ON self_facts (created_at);

PRAGMA user_version = 6;
