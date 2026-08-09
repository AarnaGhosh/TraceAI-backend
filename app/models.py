from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from .database import Base


class User(Base):
    """Dashboard / reporting-authority login (police, NGO staff, admin)."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="viewer")  # admin | officer | viewer
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Person(Base):
    """A reported missing person case."""
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    last_seen_location = Column(String, nullable=True)
    last_seen_date = Column(String, nullable=True)
    reporter_name = Column(String, nullable=True)
    reporter_contact = Column(String, nullable=True)  # phone or email
    status = Column(String, default="missing")  # missing | found

    image_path = Column(String, nullable=False)       # stored photo, relative path
    face_embedding = Column(Text, nullable=True)       # JSON-encoded vector (Facenet, 128-d)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                         onupdate=lambda: datetime.now(timezone.utc))

    sightings = relationship("Sighting", back_populates="person", cascade="all, delete-orphan")


class Sighting(Base):
    """A search/upload event that produced a match (or attempted match) against a Person."""
    __tablename__ = "sightings"

    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=True)  # null if no match found
    image_path = Column(String, nullable=False)
    similarity_score = Column(Float, nullable=True)   # 0-100, higher = closer match
    is_match = Column(Boolean, default=False)
    location = Column(String, nullable=True)
    reported_by = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    person = relationship("Person", back_populates="sightings")
