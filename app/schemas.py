from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime


# ---------- Auth ----------

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


# ---------- Persons ----------

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


# ---------- Search / Sightings ----------

class MatchResult(BaseModel):
    person: PersonOut
    similarity_score: float
    is_match: bool


class SearchResponse(BaseModel):
    query_image_path: str
    matches: List[MatchResult]
    best_match: Optional[MatchResult] = None
    message: str


class SightingOut(BaseModel):
    id: int
    person_id: Optional[int]
    image_path: str
    similarity_score: Optional[float]
    is_match: bool
    location: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class StatsOut(BaseModel):
    total_cases: int
    found_cases: int
    missing_cases: int
    total_searches: int
    total_matches: int
