from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Optional

from starlette.concurrency import run_in_threadpool

from .. import models, schemas, utils, face_utils
from ..database import get_db


router = APIRouter(
    prefix="/api/search",
    tags=["search"]
)


@router.post(
    "",
    response_model=schemas.SearchResponse
)
async def search_by_image(
    image: UploadFile = File(...),
    location: Optional[str] = Form(None),
    reported_by: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Search for a missing person using a face image.

    The uploaded image is:
    1. Saved to Supabase Storage.
    2. Processed using lightweight OpenCV YuNet + SFace.
    3. Compared against stored missing-person embeddings.
    4. The top 5 matches are returned.
    """

    # ---------------------------------------------------------
    # 1. Save uploaded sighting image
    # ---------------------------------------------------------

    sighting_image_path = await utils.save_upload(
        image,
        "sightings"
    )

    # ---------------------------------------------------------
    # 2. Generate face embedding
    # ---------------------------------------------------------

    try:

        query_embedding = await run_in_threadpool(
            face_utils.get_embedding,
            sighting_image_path
        )

    except face_utils.NoFaceDetectedError:

        raise HTTPException(
            status_code=422,
            detail=(
                "No face could be detected in the uploaded photo. "
                "Please upload a clear, front-facing photo."
            ),
        )

    except face_utils.ImageLoadError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    except Exception as exc:

        # This makes unexpected AI/model errors visible
        # instead of causing a generic 502.
        raise HTTPException(
            status_code=500,
            detail=f"Face recognition failed: {str(exc)}",
        )

    # ---------------------------------------------------------
    # 3. Get all currently missing persons
    # ---------------------------------------------------------

    candidates = (
        db.query(models.Person)
        .filter(
            models.Person.status == "missing"
        )
        .all()
    )

    # ---------------------------------------------------------
    # 4. Compare query embedding with stored embeddings
    # ---------------------------------------------------------

    scored = []

    for person in candidates:

        # Skip cases that don't have an embedding
        if not person.face_embedding:
            continue

        try:

            stored_embedding = (
                face_utils.embedding_from_json(
                    person.face_embedding
                )
            )

            score = (
                face_utils.similarity_score_percent(
                    query_embedding,
                    stored_embedding
                )
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

        except Exception:
            # If one old/corrupted embedding is invalid,
            # don't make the entire search fail.
            continue

    # ---------------------------------------------------------
    # 5. Sort by similarity
    # ---------------------------------------------------------

    scored.sort(
        key=lambda match: match.similarity_score,
        reverse=True
    )

    top_matches = scored[:5]

    # ---------------------------------------------------------
    # 6. Determine best confident match
    # ---------------------------------------------------------

    best = (
        top_matches[0]
        if (
            top_matches
            and top_matches[0].is_match
        )
        else None
    )

    # ---------------------------------------------------------
    # 7. Save sighting/search event
    # ---------------------------------------------------------

    sighting = models.Sighting(
        person_id=(
            best.person.id
            if best
            else None
        ),

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

        is_match=(
            best is not None
        ),

        location=location,
        reported_by=reported_by,
        notes=notes,
    )

    db.add(sighting)
    db.commit()

    # ---------------------------------------------------------
    # 8. Create response message
    # ---------------------------------------------------------

    if best:

        message = (
            f"Potential match found: "
            f"{best.person.name} "
            f"({best.similarity_score}% similarity)."
        )

    elif top_matches:

        message = (
            "No confident match found. "
            "Closest case shown below did not "
            "meet the match threshold."
        )

    else:

        message = (
            "No missing person cases in the "
            "database to compare against yet."
        )

    # ---------------------------------------------------------
    # 9. Return response
    # ---------------------------------------------------------

    return schemas.SearchResponse(
        query_image_path=sighting_image_path,
        matches=top_matches,
        best_match=best,
        message=message,
    )