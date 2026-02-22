"""Simple numbered migration runner for SQLite."""

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


async def _detect_baseline(db: aiosqlite.Connection) -> int:
    """Detect what version an existing database is at before migrations were introduced.

    Returns 0 for a fresh database, or the version matching existing state.
    """
    # Check if the sessions table exists at all
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
    )
    if not await cursor.fetchone():
        return 0  # Fresh database

    # Sessions table exists — this is a pre-migration database.
    # Check if user_id column was already added (the old ad-hoc migration).
    cursor = await db.execute("PRAGMA table_info(sessions)")
    columns = [row[1] for row in await cursor.fetchall()]
    if "user_id" in columns:
        return 2  # Has initial schema + user_id

    return 1  # Has initial schema only


async def run_migrations(db: aiosqlite.Connection) -> None:
    """Run all pending migrations in order."""
    # Check if this is the first time migrations are being used
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )
    has_version_table = await cursor.fetchone() is not None

    # Create version tracking table
    await db.execute(
        "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"
    )
    await db.commit()

    if not has_version_table:
        # First time: detect existing state so we don't re-run old migrations
        baseline = await _detect_baseline(db)
        if baseline > 0:
            logger.info("Detected existing database at baseline version %d", baseline)
            await db.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (baseline,)
            )
            await db.commit()

    # Get current version
    cursor = await db.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version")
    row = await cursor.fetchone()
    current_version = row[0]

    # Discover and apply pending migration files
    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

    for migration_file in migration_files:
        version = int(migration_file.name.split("_", 1)[0])
        if version <= current_version:
            continue

        logger.info("Applying migration %s", migration_file.name)
        sql = migration_file.read_text()
        await db.executescript(sql)
        await db.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (version,)
        )
        await db.commit()
        logger.info("Migration %s applied", migration_file.name)
