"""Persistent patent data cache backed by SQLite."""

import json
import logging

from sessions.database import get_db

logger = logging.getLogger(__name__)


async def get_cached(patent_number: str, data_type: str) -> dict | None:
    """Return cached data for a patent/type pair, or None on miss."""
    try:
        db = await get_db()
        cursor = await db.execute(
            "SELECT data FROM patent_cache WHERE patent_number = ? AND data_type = ?",
            (patent_number, data_type),
        )
        row = await cursor.fetchone()
        if row:
            return json.loads(row[0])
    except Exception:
        logger.debug("Cache miss (error) for %s/%s", patent_number, data_type, exc_info=True)
    return None


async def set_cached(patent_number: str, data_type: str, data: dict) -> None:
    """Persist data for a patent/type pair."""
    try:
        db = await get_db()
        await db.execute(
            "INSERT OR REPLACE INTO patent_cache (patent_number, data_type, data) VALUES (?, ?, ?)",
            (patent_number, data_type, json.dumps(data)),
        )
        await db.commit()
    except Exception:
        logger.warning("Failed to write cache for %s/%s", patent_number, data_type, exc_info=True)
