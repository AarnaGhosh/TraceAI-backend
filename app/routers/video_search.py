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


# ---------------------------------------------------------
# Video processing
# ---------------------------------------------------------

def process_video(
    video_path: str,
    reference_embedding: list,
):
    """
    Process a surveillance video.

    Pipeline:

        Video
          ↓
        Sample frames
          ↓
        YuNet detects ALL faces
          ↓
        SFace generates embedding for each face
          ↓
        Compare with reference embedding
          ↓
        Record confident matches
          ↓
        Find strongest evidence frame
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

    best_match_score = 0.0
    best_match_frame = None
    best_match_face = None
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
            # Detect ALL faces + generate embeddings
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

                    # -------------------------------------
                    # Keep strongest evidence frame
                    # -------------------------------------

                    if (
                        similarity
                        > best_match_score
                    ):

                        best_match_score = similarity

                        best_match_frame = (
                            frame.copy()
                        )

                        best_match_face = result["face"]

                        best_match_timestamp = (
                            timestamp
                        )

        frame_number += 1

    cap.release()

    # -----------------------------------------------------
    # Group continuous matches into sightings
    # -----------------------------------------------------

    groups = []

    for match in matches:

        timestamp = match["timestamp"]
        similarity = match["similarity"]

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

        # A gap of <= 1 second means
        # this is probably the same sighting.
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

    return {
        "total_frames": total_frames,
        "frames_processed": frames_processed,
        "faces_detected": faces_detected,
        "matches": matches,
        "groups": groups,
        "best_match_score": (
            round(
                best_match_score * 100,
                2,
            )
            if best_match_frame is not None
            else None
        ),
        "best_match_timestamp": (
            best_match_timestamp
        ),
        "best_match_frame": (
            best_match_frame
        ),
        "best_match_face": (
            best_match_face
        ),
    }


# ---------------------------------------------------------
# Video search endpoint
# ---------------------------------------------------------

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

    # -----------------------------------------------------
    # 1. Validate video
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 2. Find missing person
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # 3. Check reference embedding
    # -----------------------------------------------------

    if not person.face_embedding:

        raise HTTPException(
            status_code=400,
            detail=(
                "This person does not have a stored "
                "face embedding. Please report the "
                "person again with a valid reference image."
            ),
        )

    # -----------------------------------------------------
    # 4. Convert stored JSON embedding
    # -----------------------------------------------------

    from .. import face_utils

    try:

        reference_embedding = (
            face_utils.embedding_from_json(
                person.face_embedding
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Stored face embedding could not "
                f"be loaded: {str(exc)}"
            ),
        )

    # -----------------------------------------------------
    # 5. Save uploaded video temporarily
    # -----------------------------------------------------

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

            # 200 MB limit
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

        # -------------------------------------------------
        # 6. Run CPU-heavy face processing
        # -------------------------------------------------

        result = await run_in_threadpool(
            process_video,
            temp_video_path,
            reference_embedding,
        )

    finally:

        # -------------------------------------------------
        # 7. Remove temporary video
        # -------------------------------------------------

        if (
            temp_video_path
            and os.path.exists(temp_video_path)
        ):

            try:
                os.remove(
                    temp_video_path
                )
            except OSError:
                pass

    # -----------------------------------------------------
    # 8. No confident matches
    # -----------------------------------------------------

    if not result["groups"]:

        return schemas.VideoSearchResponse(
            person_id=person.id,
            video_name=video.filename,
            camera_id=camera_id,
            location=location,
            latitude=latitude,
            longitude=longitude,
            frames_processed=result[
                "frames_processed"
            ],
            faces_detected=result[
                "faces_detected"
            ],
            matches_found=0,
            best_similarity=None,
            best_timestamp_seconds=None,
            evidence_image_path=None,
            sightings=[],
            message=(
                "Video processed successfully, "
                "but no confident match was found."
            ),
        )

    # -----------------------------------------------------
    # 9. Upload strongest evidence frame
    # -----------------------------------------------------

    evidence_image_path = None

    best_frame = result[
        "best_match_frame"
    ]

    if best_frame is not None:

        # Annotate the exact face that produced the strongest match.
        best_face = result.get("best_match_face")
        if best_face is not None and len(best_face) >= 4:
            try:
                x, y, w, h = [int(round(float(v))) for v in best_face[:4]]
                x = max(0, x); y = max(0, y); w = max(1, w); h = max(1, h)
                x2 = min(best_frame.shape[1] - 1, x + w)
                y2 = min(best_frame.shape[0] - 1, y + h)
                cv2.rectangle(best_frame, (x, y), (x2, y2), (39, 174, 96), 3)
                label = f"MATCH {result['best_match_score']:.2f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.65; thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                label_top = max(0, y - text_h - baseline - 10)
                label_bottom = label_top + text_h + baseline + 8
                label_right = min(best_frame.shape[1] - 1, x + text_w + 10)
                cv2.rectangle(best_frame, (x, label_top), (label_right, label_bottom), (39, 174, 96), -1)
                cv2.putText(best_frame, label, (x + 5, label_bottom - baseline - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            except Exception:
                pass

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

        # Create an UploadFile-like object
        # for the existing Supabase utility.
        evidence_file = UploadFile(
            filename="video_evidence.jpg",
            file=io.BytesIO(
                encoded.tobytes()
            ),
            headers=Headers({
                "content-type": "image/jpeg"
            }),
        )


        evidence_image_path = (
            await utils.save_upload(
                evidence_file,
                "sightings",
            )
        )

    # -----------------------------------------------------
    # 10. Create Sighting records
    # -----------------------------------------------------

    sighting_objects = []

    for group in result["groups"]:

        sighting = models.Sighting(

            person_id=person.id,

            image_path=evidence_image_path,

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

    # -----------------------------------------------------
    # 11. Mark the case as FOUND and store the surveillance
    #     location so the dashboard can update from the
    #     backend without relying on a second frontend PATCH.
    # -----------------------------------------------------

    person.status = "found"

    location_note = (
        "Surveillance potential sighting detected."
        f"\nLocation: {location or 'Not provided'}"
        f"\nCamera: {camera_id or 'Not specified'}"
        f"\nTimestamp: {result['best_match_timestamp']:.2f} seconds"
        f"\nSimilarity: {result['best_match_score']:.2f}%"
    )

    if (
        latitude is not None
        and longitude is not None
        and not (latitude == 0 and longitude == 0)
    ):
        location_note += (
            f"\nGPS Coordinates: {latitude}, {longitude}"
        )

    existing_description = person.description or ""

    if "Surveillance potential sighting detected." not in existing_description:
        person.description = (
            f"{existing_description}\n\n{location_note}"
            if existing_description
            else location_note
        )

    # -----------------------------------------------------
    # 12. Commit sightings + updated person status together
    # -----------------------------------------------------

    db.commit()

    # Refresh database objects
    for sighting in sighting_objects:
        db.refresh(sighting)

    # -----------------------------------------------------
    # 13. Return response
    # -----------------------------------------------------

    return schemas.VideoSearchResponse(

        person_id=person.id,

        video_name=video.filename,

        camera_id=camera_id,

        location=location,

        latitude=latitude,

        longitude=longitude,

        frames_processed=result[
            "frames_processed"
        ],

        faces_detected=result[
            "faces_detected"
        ],

        matches_found=len(
            result["groups"]
        ),

        best_similarity=result[
            "best_match_score"
        ],

        best_timestamp_seconds=result[
            "best_match_timestamp"
        ],

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