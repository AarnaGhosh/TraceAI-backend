from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import models, schemas
from ..database import get_db

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("", response_model=schemas.StatsOut)
def get_stats(db: Session = Depends(get_db)):
    """Powers the live counters on the homepage (Cases Processed, Faster Recovery, etc.)."""
    total_cases = db.query(models.Person).count()
    found_cases = db.query(models.Person).filter(models.Person.status == "found").count()
    missing_cases = total_cases - found_cases
    total_searches = db.query(models.Sighting).count()
    total_matches = db.query(models.Sighting).filter(models.Sighting.is_match == True).count()  # noqa: E712

    return schemas.StatsOut(
        total_cases=total_cases,
        found_cases=found_cases,
        missing_cases=missing_cases,
        total_searches=total_searches,
        total_matches=total_matches,
    )
