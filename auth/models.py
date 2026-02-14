"""User model for FastAPI-Users with Stripe subscription fields."""

from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTableUUID, Base):
    stripe_customer_id: Mapped[str | None] = mapped_column(
        String(255), default=None
    )
    subscription_status: Mapped[str | None] = mapped_column(
        String(50), default=None
    )
    subscription_id: Mapped[str | None] = mapped_column(
        String(255), default=None
    )
    plan_type: Mapped[str | None] = mapped_column(
        String(50), default=None
    )
