"""SQLAlchemy Database configuration, session management, and Transaction model."""
from __future__ import annotations

import datetime
import uuid
from typing import Generator

from sqlalchemy import Boolean, Column, DateTime, Float, JSON, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from api.config import DATABASE_URL

# Engine configuration with sqlite specific arguments
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class TransactionRecord(Base):
    """Transaction inference log table."""

    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        index=True,
        nullable=False,
    )
    features = Column(JSON, nullable=False)
    fraud_score = Column(Float, nullable=False, index=True)
    is_fraud = Column(Boolean, nullable=False, index=True)
    threshold = Column(Float, nullable=False)
    model_version = Column(String(64), nullable=False)
    explanation = Column(JSON, nullable=True)
    latency_ms = Column(Float, nullable=False)


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI Dependency for database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
