import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


def _normalize_database_url(url: str) -> str:
    # Allow common alias.
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Prefer asyncpg in production (works with SQLAlchemy 2 via async engine),
    # but this project currently uses the sync engine; ensure a sync driver.
    # If you're using docker-compose's default DATABASE_URL (postgresql://...),
    # SQLAlchemy will try psycopg2 unless a driver is specified.
    if url.startswith("postgresql://") and "+" not in url.split("postgresql://", 1)[1]:
        # Use psycopg (v3) which is pure Python + widely supported.
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


DATABASE_URL = _normalize_database_url(os.getenv("DATABASE_URL", "sqlite:///./test.db"))

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
