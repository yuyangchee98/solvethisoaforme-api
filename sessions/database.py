"""SQLite database connection management for sessions."""

import os
from pathlib import Path

import aiosqlite

from .migrator import run_migrations

# Environment variable for data directory
DATA_PATH = os.environ.get("DATA_PATH", "./data")

# Singleton connection
_connection: aiosqlite.Connection | None = None


def get_data_path() -> Path:
    """Get the data directory path."""
    return Path(DATA_PATH)


def get_db_path() -> Path:
    """Get the database file path."""
    return get_data_path() / "app.db"


async def init_db() -> None:
    """Initialize the database connection and run pending migrations."""
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

    # Enable foreign keys and WAL mode for concurrent access
    await _connection.execute("PRAGMA foreign_keys = ON")
    await _connection.execute("PRAGMA journal_mode = WAL")

    # Run numbered migrations
    await run_migrations(_connection)


async def close_db() -> None:
    """Close the database connection."""
    global _connection

    if _connection is not None:
        await _connection.close()
        _connection = None


async def get_db() -> aiosqlite.Connection:
    """Get the singleton database connection.

    Raises:
        RuntimeError: If database is not initialized
    """
    if _connection is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _connection
