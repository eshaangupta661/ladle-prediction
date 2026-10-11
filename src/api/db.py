"""Database model for predictions; each row's feature snapshot doubles as the heat history."""
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, JSON, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ladle.db")
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def utc_now():
    """Naive UTC timestamp, matching how SQLite and the existing columns store times."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    requested_at = Column(DateTime, default=utc_now, nullable=False)
    ladle_no = Column(Integer, nullable=False, index=True)
    heat_no = Column(String(50), nullable=True)
    campaign_id = Column(Integer, nullable=False)
    predicted_life = Column(Float, nullable=False)
    actual_life = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=False)
    feature_snapshot = Column(JSON, nullable=False)
    is_partial_campaign = Column(Boolean, default=False, nullable=False)


def create_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    database = SessionLocal()
    try:
        yield database
    finally:
        database.close()
