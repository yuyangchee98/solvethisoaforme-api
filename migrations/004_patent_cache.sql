CREATE TABLE IF NOT EXISTS patent_cache (
    patent_number TEXT NOT NULL,
    data_type     TEXT NOT NULL,
    data          TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (patent_number, data_type)
);
