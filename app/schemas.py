
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime


# =========================================================
# AUTH
# =========================================================

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: Optional[str] = "viewer"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# =========================================================
# PERSONS
# =========================================================

class PersonOut(BaseModel):
    id: int
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    description: Optional[str] = None

    last_seen_location: Optional[str] = None
    last_seen_date: Optional[str] = None

    reporter_name: Optional[str] = None
    reporter_contact: Optional[str] = None

    status: str

    image_path: str

    created_at: datetime

    class Config:
        from_attributes = True


class PersonUpdate(BaseModel):
    status: Optional[str] = None
    description: Optional[str] = None


# =========================================================
# IMAGE SEARCH
# =========================================================

class MatchResult(BaseModel):
    person: PersonOut
    similarity_score: float
    is_match: bool


class SearchResponse(BaseModel):
    query_image_path: str
    matches: List[MatchResult]
    best_match: Optional[MatchResult] = None
    message: str


# =========================================================
# SIGHTINGS
# =========================================================

class SightingOut(BaseModel):
    """
    Response model for a detected sighting.

    Supports both:
        - image-based searches
        - video surveillance searches
    """

    id: int

    person_id: Optional[int] = None

    # Best evidence image / frame
    image_path: str

    # Similarity score from 0-100
    similarity_score: Optional[float] = None

    # Whether the similarity crossed the threshold
    is_match: bool

    # Location metadata
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    # Video surveillance metadata
    camera_id: Optional[str] = None
    video_name: Optional[str] = None

    # Strongest matching frame timestamp
    timestamp_seconds: Optional[float] = None

    # Continuous sighting interval
    sighting_start_seconds: Optional[float] = None
    sighting_end_seconds: Optional[float] = None

    # Search/report information
    reported_by: Optional[str] = None
    notes: Optional[str] = None

    created_at: datetime

    class Config:
        from_attributes = True


# =========================================================
# VIDEO SEARCH
# =========================================================

class VideoSearchResponse(BaseModel):
    """
    Response returned after searching a surveillance video
    for a selected missing person.
    """

    person_id: int

    video_name: Optional[str] = None

    camera_id: Optional[str] = None

    location: Optional[str] = None

    latitude: Optional[float] = None

    longitude: Optional[float] = None

    # Number of frames examined
    frames_processed: int

    # Number of faces detected across sampled frames
    faces_detected: int

    # Number of confident matching detections
    matches_found: int

    # Strongest match
    best_similarity: Optional[float] = None

    best_timestamp_seconds: Optional[float] = None

    # Saved evidence frame
    evidence_image_path: Optional[str] = None

    # Database sightings created from the search
    sightings: List[SightingOut]

    message: str


# =========================================================
# STATS
# =========================================================

class StatsOut(BaseModel):
    total_cases: int
    found_cases: int
    missing_cases: int
    total_searches: int
    total_matches: int

class VideoSearchResponse(BaseModel):
    person_id: int
    video_name: str
    camera_id: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    frames_processed: int
    faces_detected: int
    matches_found: int

    best_similarity: Optional[float] = None
    best_timestamp_seconds: Optional[float] = None

    evidence_image_path: Optional[str] = None

    sightings: List[SightingOut] = []

    message: str