"""SQLAlchemy async engine for FastAPI-Users auth tables.

Shares the same SQLite database file as the existing aiosqlite session system.
WAL mode is essential for concurrent access from both ORMs.
"""

import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

DATA_PATH = os.environ.get("DATA_PATH", "./data")
DATABASE_URL = f"sqlite+aiosqlite:///{DATA_PATH}/app.db"

engine = create_async_engine(DATABASE_URL)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


async def init_auth_db() -> None:
    """Create auth tables and enable WAL mode."""
    from .models import Base

    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session():
    async with async_session_maker() as session:
        yield session
