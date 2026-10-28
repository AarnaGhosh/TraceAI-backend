import os
import io
import cv2
import tempfile
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
)

from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers

from .. import models, schemas, utils
from ..database import get_db


router = APIRouter(
    prefix="/api/video",
    tags=["video-search"],
)


# =========================================================
# VIDEO PROCESSING
# =========================================================

def process_video(
    video_path: str,
    reference_embedding: list,
):
    """
    Search a surveillance video for the selected person.

    Pipeline:

        Surveillance Video
                ↓
        Sample video frames
                ↓
        YuNet detects ALL faces
                ↓
        SFace creates embedding for each face
                ↓
        Compare with selected person's SFace embedding
                ↓
        Record confident matches
                ↓
        Group continuous sightings
                ↓
        Save strongest evidence frame
    """

    from .. import face_utils

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open uploaded video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # -----------------------------------------------------
    # Sample approximately 2 frames per second
    # -----------------------------------------------------

    sample_interval = max(
        int(fps / 2),
        1,
    )

    frame_number = 0

    frames_processed = 0
    faces_detected = 0

    matches = []

    # Keep best similarity even if it does not cross
    # the threshold. This is useful for debugging.
    best_similarity_value = -1.0

    best_similarity_percent = None

    best_match_frame = None
    best_match_timestamp = None

    # -----------------------------------------------------
    # Read video
    # -----------------------------------------------------

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if frame_number % sample_interval == 0:

            frames_processed += 1

            timestamp = (
                frame_number / fps
            )

            # ---------------------------------------------
            # Detect ALL faces
            # ---------------------------------------------

            results = (
                face_utils.get_frame_embeddings(
                    frame
                )
            )

            faces_detected += len(results)

            # ---------------------------------------------
            # Compare every detected face
            # ---------------------------------------------

            for face_index, result in enumerate(
                results
            ):

                video_embedding = (
                    result["embedding"]
                )

                # -----------------------------------------
                # Cosine similarity
                # -----------------------------------------

                similarity = (
                    face_utils.cosine_similarity(
                        reference_embedding,
                        video_embedding,
                    )
                )

                similarity_percent = (
                    face_utils.similarity_score_percent(
                        reference_embedding,
                        video_embedding,
                    )
                )

                # -----------------------------------------
                # ALWAYS keep strongest similarity
                # -----------------------------------------

                if similarity > best_similarity_value:

                    best_similarity_value = similarity

                    best_similarity_percent = (
                        similarity_percent
                    )

                    best_match_frame = (
                        frame.copy()
                    )

                    best_match_timestamp = (
                        timestamp
                    )

                # -----------------------------------------
                # Threshold decision
                # -----------------------------------------

                matched = (
                    face_utils.is_match(
                        reference_embedding,
                        video_embedding,
                    )
                )

                if matched:

                    matches.append(
                        {
                            "timestamp": timestamp,
                            "face_index": face_index,
                            "similarity": similarity_percent,
                        }
                    )

        frame_number += 1

    cap.release()

    # =====================================================
    # GROUP CONTINUOUS MATCHES
    # =====================================================

    groups = []

    for match in matches:

        timestamp = match["timestamp"]

        similarity = match["similarity"]

        # First match
        if not groups:

            groups.append(
                {
                    "start": timestamp,
                    "end": timestamp,
                    "best_similarity": similarity,
                    "best_time": timestamp,
                }
            )

            continue

        last_group = groups[-1]

        # Matches within one second are considered
        # part of the same continuous sighting.

        if (
            timestamp
            - last_group["end"]
            <= 1.0
        ):

            last_group["end"] = timestamp

            if (
                similarity
                > last_group["best_similarity"]
            ):

                last_group[
                    "best_similarity"
                ] = similarity

                last_group[
                    "best_time"
                ] = timestamp

        else:

            groups.append(
                {
                    "start": timestamp,
                    "end": timestamp,
                    "best_similarity": similarity,
                    "best_time": timestamp,
                }
            )

    # =====================================================
    # RETURN RESULTS
    # =====================================================

    return {

        "total_frames":
            total_frames,

        "frames_processed":
            frames_processed,

        "faces_detected":
            faces_detected,

        "matches":
            matches,

        "groups":
            groups,

        "best_match_score":
            best_similarity_percent,

        "best_match_timestamp":
            best_match_timestamp,

        "best_match_frame":
            best_match_frame,
    }


# =========================================================
# VIDEO SEARCH ENDPOINT
# =========================================================

@router.post(
    "/search",
    response_model=schemas.VideoSearchResponse,
)
async def search_video(

    person_id: int = Form(...),

    video: UploadFile = File(...),

    camera_id: Optional[str] = Form(None),

    location: Optional[str] = Form(None),

    latitude: Optional[float] = Form(None),

    longitude: Optional[float] = Form(None),

    reported_by: Optional[str] = Form(None),

    notes: Optional[str] = Form(None),

    db: Session = Depends(get_db),

):

    """
    Search a surveillance video for a selected
    missing person.
    """

    from .. import face_utils

    # =====================================================
    # 1. VALIDATE VIDEO
    # =====================================================

    if not video.filename:

        raise HTTPException(
            status_code=400,
            detail="No video file was provided.",
        )

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    extension = os.path.splitext(
        video.filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Use MP4, AVI, MOV, MKV or WEBM."
            ),
        )

    # =====================================================
    # 2. FIND SELECTED PERSON
    # =====================================================

    person = (
        db.query(models.Person)
        .filter(
            models.Person.id == person_id
        )
        .first()
    )

    if person is None:

        raise HTTPException(
            status_code=404,
            detail="Person not found.",
        )

    # =====================================================
    # 3. MAKE SURE REFERENCE IMAGE EXISTS
    # =====================================================

    if not person.image_path:

        raise HTTPException(
            status_code=400,
            detail=(
                "This missing-person case does not "
                "have a reference image."
            ),
        )

    # =====================================================
    # 4. REGENERATE SFACE EMBEDDING
    #
    # IMPORTANT:
    #
    # Do NOT blindly trust the old database embedding.
    #
    # The current TraceAI system uses:
    #
    # YuNet → SFace → 128-D embedding
    #
    # We regenerate it from the person's actual
    # reference photograph so that it is guaranteed
    # to use the same SFace model as the surveillance
    # video.
    # =====================================================

    try:

        reference_embedding = (
            face_utils.get_embedding(
                person.image_path
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not generate a fresh face "
                "embedding from the person's reference "
                f"image: {str(exc)}"
            ),
        )

    # =====================================================
    # 5. VALIDATE EMBEDDING
    # =====================================================

    if not reference_embedding:

        raise HTTPException(
            status_code=400,
            detail=(
                "The person's reference image did not "
                "produce a valid face embedding."
            ),
        )

    if len(reference_embedding) != 128:

        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid SFace embedding size. "
                f"Expected 128 values but got "
                f"{len(reference_embedding)}."
            ),
        )

    # =====================================================
    # 6. SAVE FRESH EMBEDDING TO DATABASE
    # =====================================================

    try:

        person.face_embedding = (
            face_utils.embedding_to_json(
                reference_embedding
            )
        )

        db.commit()

        db.refresh(person)

    except Exception as exc:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Could not update the person's "
                f"face embedding: {str(exc)}"
            ),
        )

    # =====================================================
    # 7. SAVE VIDEO TEMPORARILY
    # =====================================================

    temp_video_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_video:

            temp_video_path = (
                temp_video.name
            )

            contents = await video.read()

            if not contents:

                raise HTTPException(
                    status_code=400,
                    detail="Uploaded video is empty.",
                )

            # 200 MB maximum
            max_video_size = (
                200 * 1024 * 1024
            )

            if len(contents) > max_video_size:

                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Video is too large. "
                        "Maximum allowed size is 200 MB."
                    ),
                )

            temp_video.write(contents)

        # =================================================
        # 8. RUN VIDEO PROCESSING
        # =================================================

        result = await run_in_threadpool(
            process_video,
            temp_video_path,
            reference_embedding,
        )

    finally:

        # =================================================
        # 9. DELETE TEMP VIDEO
        # =================================================

        if (
            temp_video_path
            and os.path.exists(
                temp_video_path
            )
        ):

            try:

                os.remove(
                    temp_video_path
                )

            except OSError:

                pass

    # =====================================================
    # 10. NO CONFIDENT MATCH
    # =====================================================

    if not result["groups"]:

        best_similarity = (
            result["best_match_score"]
        )

        best_timestamp = (
            result["best_match_timestamp"]
        )

        return schemas.VideoSearchResponse(

            person_id=person.id,

            video_name=video.filename,

            camera_id=camera_id,

            location=location,

            latitude=latitude,

            longitude=longitude,

            frames_processed=(
                result["frames_processed"]
            ),

            faces_detected=(
                result["faces_detected"]
            ),

            matches_found=0,

            best_similarity=best_similarity,

            best_timestamp_seconds=(
                best_timestamp
            ),

            evidence_image_path=None,

            sightings=[],

            message=(
                "Video processed successfully. "
                "No confident match was found. "
                f"Best similarity was "
                f"{best_similarity if best_similarity is not None else 'N/A'}%."
            ),
        )

    # =====================================================
    # 11. UPLOAD BEST EVIDENCE FRAME
    # =====================================================

    evidence_image_path = None

    best_frame = (
        result["best_match_frame"]
    )

    if best_frame is not None:

        success, encoded = cv2.imencode(
            ".jpg",
            best_frame,
        )

        if not success:

            raise HTTPException(
                status_code=500,
                detail=(
                    "Could not encode the "
                    "best evidence frame."
                ),
            )

        evidence_file = UploadFile(

            filename="video_evidence.jpg",

            file=io.BytesIO(
                encoded.tobytes()
            ),

            headers=Headers(
                {
                    "content-type":
                    "image/jpeg"
                }
            ),
        )

        evidence_image_path = (
            await utils.save_upload(
                evidence_file,
                "sightings",
            )
        )

    # =====================================================
    # 12. CREATE SIGHTING RECORDS
    # =====================================================

    sighting_objects = []

    for group in result["groups"]:

        sighting = models.Sighting(

            person_id=person.id,

            image_path=(
                evidence_image_path
            ),

            similarity_score=(
                group["best_similarity"]
            ),

            is_match=True,

            location=location,

            latitude=latitude,

            longitude=longitude,

            camera_id=camera_id,

            video_name=video.filename,

            timestamp_seconds=(
                group["best_time"]
            ),

            sighting_start_seconds=(
                group["start"]
            ),

            sighting_end_seconds=(
                group["end"]
            ),

            reported_by=reported_by,

            notes=notes,
        )

        db.add(sighting)

        sighting_objects.append(
            sighting
        )

    # =====================================================
    # 13. COMMIT
    # =====================================================

    db.commit()

    for sighting in sighting_objects:

        db.refresh(
            sighting
        )

    # =====================================================
    # 14. RETURN SUCCESS
    # =====================================================

    return schemas.VideoSearchResponse(

        person_id=person.id,

        video_name=video.filename,

        camera_id=camera_id,

        location=location,

        latitude=latitude,

        longitude=longitude,

        frames_processed=(
            result["frames_processed"]
        ),

        faces_detected=(
            result["faces_detected"]
        ),

        matches_found=len(
            result["groups"]
        ),

        best_similarity=(
            result["best_match_score"]
        ),

        best_timestamp_seconds=(
            result["best_match_timestamp"]
        ),

        evidence_image_path=(
            evidence_image_path
        ),

        sightings=[
            schemas.SightingOut.model_validate(
                sighting
            )
            for sighting in sighting_objects
        ],

        message=(
            f"Potential sighting found for "
            f"{person.name}. "
            f"{len(result['groups'])} "
            f"continuous sighting(s) detected."
        ),
    )
