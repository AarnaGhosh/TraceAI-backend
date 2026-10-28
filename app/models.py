from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    ForeignKey,
    Boolean,
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

from .database import Base


# =========================================================
# USER
# =========================================================

class User(Base):
    """
    Dashboard / reporting-authority login.

    Roles:
        admin
        officer
        viewer
    """

    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    name = Column(
        String,
        nullable=False,
    )

    email = Column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    hashed_password = Column(
        String,
        nullable=False,
    )

    role = Column(
        String,
        default="viewer",
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )


# =========================================================
# MISSING PERSON
# =========================================================

class Person(Base):
    """
    A reported missing person case.

    The face_embedding field stores the person's
    128-dimensional SFace embedding as JSON.
    """

    __tablename__ = "persons"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    name = Column(
        String,
        nullable=False,
    )

    age = Column(
        Integer,
        nullable=True,
    )

    gender = Column(
        String,
        nullable=True,
    )

    description = Column(
        Text,
        nullable=True,
    )

    last_seen_location = Column(
        String,
        nullable=True,
    )

    last_seen_date = Column(
        String,
        nullable=True,
    )

    reporter_name = Column(
        String,
        nullable=True,
    )

    reporter_contact = Column(
        String,
        nullable=True,
    )

    status = Column(
        String,
        default="missing",
    )

    # Stored reference photograph
    image_path = Column(
        String,
        nullable=False,
    )

    # 128-dimensional SFace embedding
    # stored as JSON text
    face_embedding = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    sightings = relationship(
        "Sighting",
        back_populates="person",
        cascade="all, delete-orphan",
    )


# =========================================================
# SIGHTING
# =========================================================

class Sighting(Base):
    """
    A potential sighting detected during a search.

    A sighting can come from:
        - an uploaded image
        - a surveillance video

    For video searches, the additional fields store:
        - video name
        - camera ID
        - timestamp in video
        - camera/location metadata
    """

    __tablename__ = "sightings"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    # Person who potentially matched
    person_id = Column(
        Integer,
        ForeignKey("persons.id"),
        nullable=True,
    )

    # Best evidence frame / uploaded image
    image_path = Column(
        String,
        nullable=False,
    )

    # Similarity score from 0-100
    similarity_score = Column(
        Float,
        nullable=True,
    )

    # True when similarity crosses the configured threshold
    is_match = Column(
        Boolean,
        default=False,
    )

    # Human-readable location
    location = Column(
        String,
        nullable=True,
    )

    # Optional latitude / longitude for dashboard map
    latitude = Column(
        Float,
        nullable=True,
    )

    longitude = Column(
        Float,
        nullable=True,
    )

    # =====================================================
    # VIDEO SURVEILLANCE METADATA
    # =====================================================

    # Example:
    # CAM01
    camera_id = Column(
        String,
        nullable=True,
    )

    # Example:
    # CAM01.mp4
    video_name = Column(
        String,
        nullable=True,
    )

    # Timestamp of the strongest matching frame
    # measured in seconds from the beginning of the video
    timestamp_seconds = Column(
        Float,
        nullable=True,
    )

    # Optional start/end timestamps for a grouped sighting
    # Example: person detected from 12.4s to 18.7s
    sighting_start_seconds = Column(
        Float,
        nullable=True,
    )

    sighting_end_seconds = Column(
        Float,
        nullable=True,
    )

    # Name/contact of person or system that initiated search
    reported_by = Column(
        String,
        nullable=True,
    )

    # Additional information
    notes = Column(
        Text,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    person = relationship(
        "Person",
        back_populates="sightings",
    )
