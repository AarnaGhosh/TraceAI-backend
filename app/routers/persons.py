from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional, List

from .. import models, schemas, auth, utils
from ..database import get_db

router = APIRouter(prefix="/api/persons", tags=["persons"])


@router.post("", response_model=schemas.PersonOut, status_code=201)
async def report_missing_person(
    name: str = Form(...),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    last_seen_location: Optional[str] = Form(None),
    last_seen_date: Optional[str] = Form(None),
    reporter_name: Optional[str] = Form(None),
    reporter_contact: Optional[str] = Form(None),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Report a new missing person.
    Saves the photo and stores the case.
    """

    image_path = await utils.save_upload(image, "persons")

    person = models.Person(
        name=name,
        age=age,
        gender=gender,
        description=description,
        last_seen_location=last_seen_location,
        last_seen_date=last_seen_date,
        reporter_name=reporter_name,
        reporter_contact=reporter_contact,
        image_path=image_path,
        face_embedding=None,
    )

    db.add(person)
    db.commit()
    db.refresh(person)

    return person

@router.get("", response_model=List[schemas.PersonOut])
def list_persons(status: Optional[str] = None, db: Session = Depends(get_db)):
    """List all cases, optionally filtered by status=missing|found. Powers the Dashboard."""
    query = db.query(models.Person)
    if status:
        query = query.filter(models.Person.status == status)
    return query.order_by(models.Person.created_at.desc()).all()


@router.get("/{person_id}", response_model=schemas.PersonOut)
def get_person(person_id: int, db: Session = Depends(get_db)):
    person = db.query(models.Person).filter(models.Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    return person


@router.patch("/{person_id}", response_model=schemas.PersonOut)
def update_person(
    person_id: int,
    payload: schemas.PersonUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    """Mark a case as found / update details. Requires login (dashboard action)."""
    person = db.query(models.Person).filter(models.Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    if payload.status is not None:
        person.status = payload.status
    if payload.description is not None:
        person.description = payload.description

    db.commit()
    db.refresh(person)
    return person


@router.delete("/{person_id}", status_code=204)
def delete_person(
    person_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    person = db.query(models.Person).filter(models.Person.id == person_id).first()
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    db.delete(person)
    db.commit()
    return None
