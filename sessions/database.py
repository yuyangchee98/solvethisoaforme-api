"""SQLite database connection management for sessions."""

import os
from pathlib import Path

import aiosqlite

# Environment variable for data directory
DATA_PATH = os.environ.get("DATA_PATH", "./data")

# Singleton connection
_connection: aiosqlite.Connection | None = None

# Schema for sessions, messages, and uploaded documents
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS uploaded_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    message_id INTEGER,
    filename TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    document_type TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_uploaded_documents_session_id ON uploaded_documents(session_id);
CREATE INDEX IF NOT EXISTS idx_uploaded_documents_message_id ON uploaded_documents(message_id);
"""


def get_data_path() -> Path:
    """Get the data directory path.

    Returns:
        Path to the data directory
    """
    return Path(DATA_PATH)


def get_db_path() -> Path:
    """Get the database file path.

    Returns:
        Path to the SQLite database file
    """
    return get_data_path() / "app.db"


async def init_db() -> None:
    """Initialize the database connection and create schema.

    Creates the data directory if it doesn't exist and runs
    the schema to create required tables.
    """
    global _connection

    # Create data directory
    data_path = get_data_path()
    data_path.mkdir(parents=True, exist_ok=True)

    # Create sessions directory
    sessions_path = data_path / "sessions"
    sessions_path.mkdir(exist_ok=True)

    # Connect to database
    db_path = get_db_path()
    _connection = await aiosqlite.connect(db_path)

    # Enable foreign keys
    await _connection.execute("PRAGMA foreign_keys = ON")

    # Run schema
    await _connection.executescript(SCHEMA)
    await _connection.commit()


async def close_db() -> None:
    """Close the database connection."""
    global _connection

    if _connection is not None:
        await _connection.close()
        _connection = None


async def get_db() -> aiosqlite.Connection:
    """Get the singleton database connection.

    Returns:
        The active database connection

    Raises:
        RuntimeError: If database is not initialized
    """
    if _connection is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _connection
