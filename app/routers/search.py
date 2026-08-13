from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional
from starlette.concurrency import run_in_threadpool

from .. import models, schemas, utils
from ..database import get_db

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=schemas.SearchResponse)
async def search_by_image(
    image: UploadFile = File(...),
    location: Optional[str] = Form(None),
    reported_by: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    from .. import face_utils

    """
    Core matching endpoint: upload a photo and compare it
    against stored missing-person face embeddings.
    """

    sighting_image_path = await utils.save_upload(image, "sightings")

    try:
        query_embedding = await run_in_threadpool(
            face_utils.get_embedding,
            sighting_image_path
        )
    except face_utils.NoFaceDetectedError:
        raise HTTPException(
            status_code=422,
            detail="No face could be detected in the uploaded photo. "
                   "Please upload a clear, front-facing photo.",
        )

    candidates = (
        db.query(models.Person)
        .filter(models.Person.status == "missing")
        .all()
    )

    scored = []

    for person in candidates:
        if not person.face_embedding:
            continue

        stored_embedding = face_utils.embedding_from_json(
            person.face_embedding
        )

        score = face_utils.similarity_score_percent(
            query_embedding,
            stored_embedding
        )

        matched = face_utils.is_match(
            query_embedding,
            stored_embedding
        )

        scored.append(
            schemas.MatchResult(
                person=person,
                similarity_score=score,
                is_match=matched,
            )
        )

    scored.sort(
        key=lambda m: m.similarity_score,
        reverse=True
    )

    top_matches = scored[:5]

    best = (
        top_matches[0]
        if top_matches and top_matches[0].is_match
        else None
    )

    sighting = models.Sighting(
        person_id=best.person.id if best else None,
        image_path=sighting_image_path,
        similarity_score=(
            best.similarity_score
            if best
            else (
                top_matches[0].similarity_score
                if top_matches
                else None
            )
        ),
        is_match=best is not None,
        location=location,
        reported_by=reported_by,
        notes=notes,
    )

    db.add(sighting)
    db.commit()

    if best:
        message = (
            f"Potential match found: {best.person.name} "
            f"({best.similarity_score}% similarity)."
        )
    elif top_matches:
        message = (
            "No confident match found. Closest case shown below "
            "did not meet the match threshold."
        )
    else:
        message = (
            "No missing person cases in the database "
            "to compare against yet."
        )

    return schemas.SearchResponse(
        query_image_path=sighting_image_path,
        matches=top_matches,
        best_match=best,
        message=message,
    )