from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# We will use a local PostgreSQL or fallback to SQLite for easy initial testing if PG is not set up yet
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")

# For sqlite, we need connect_args={"check_same_thread": False}. For Postgres, we don't.
connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args=connect_args
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def run_migrations():
    """기존 DB에 누락된 컬럼을 안전하게 추가합니다 (SQLite 호환)."""
    if not SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
        return
    import sqlite3
    db_path = SQLALCHEMY_DATABASE_URL.replace("sqlite:///", "")
    if not db_path or not os.path.exists(db_path):
        return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(videos)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    if "opinion" not in existing_cols:
        cursor.execute("ALTER TABLE videos ADD COLUMN opinion TEXT")
        print("[DB Migration] Added column: videos.opinion")
    conn.commit()
    conn.close()
