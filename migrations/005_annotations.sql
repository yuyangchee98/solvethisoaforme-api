CREATE TABLE IF NOT EXISTS patent_annotations (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    patent_number   TEXT NOT NULL,
    section         TEXT NOT NULL,
    section_index   INTEGER NOT NULL,
    paragraph_index INTEGER NOT NULL,
    start_offset    INTEGER NOT NULL,
    end_offset      INTEGER NOT NULL,
    selected_text   TEXT NOT NULL,
    note            TEXT NOT NULL DEFAULT '',
    color           TEXT NOT NULL DEFAULT 'yellow',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX idx_annotations_user_patent ON patent_annotations(user_id, patent_number);
