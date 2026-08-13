"""
Face detection, embedding and matching for TraceAI.

Uses DeepFace with the Facenet model and OpenCV face detector.

Images may be either:
- local file paths
- public Supabase Storage URLs

Embeddings are stored as JSON text in PostgreSQL.
"""

import json
import os
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

MATCH_THRESHOLD = float(
    os.getenv("FACE_MATCH_THRESHOLD", "0.55")
)

DeepFace = None


class NoFaceDetectedError(Exception):
    """Raised when no face can be detected."""
    pass


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded."""
    pass


def get_deepface():
    """
    Lazily load DeepFace.

    This prevents TensorFlow/DeepFace from being loaded when
    the application starts unless face matching is actually used.
    """
    global DeepFace

    if DeepFace is None:
        from deepface import DeepFace as DF
        DeepFace = DF

    return DeepFace


def load_image(image_source: str) -> np.ndarray:
    """
    Load an image from either a local path or a public URL.

    The image is kept in memory only for the duration of processing.
    Nothing is permanently written to Render's disk.
    """

    if not image_source:
        raise ImageLoadError("No image path was provided.")

    parsed = urlparse(image_source)

    # Supabase/public HTTP URL
    if parsed.scheme in ("http", "https"):
        try:
            response = requests.get(
                image_source,
                timeout=20,
            )
            response.raise_for_status()

            content = response.content

            if not content:
                raise ImageLoadError("Downloaded image is empty.")

            image_array = np.frombuffer(
                content,
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR,
            )

            if image is None:
                raise ImageLoadError(
                    "Downloaded file could not be decoded as an image."
                )

            return image

        except requests.RequestException as exc:
            raise ImageLoadError(
                f"Could not download image: {exc}"
            ) from exc

    # Local file
    if not os.path.exists(image_source):
        raise ImageLoadError(
            f"Image file not found: {image_source}"
        )

    image = cv2.imread(image_source)

    if image is None:
        raise ImageLoadError(
            "Local file could not be decoded as an image."
        )

    return image


def get_embedding(image_source: str) -> list:
    """
    Generate a 128-dimensional Facenet embedding.

    Accepts either a local image path or a public image URL.
    """

    image = load_image(image_source)

    try:
        deepface = get_deepface()

        result = deepface.represent(
            img_path=image,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )

    except ValueError as exc:
        raise NoFaceDetectedError(str(exc)) from exc

    except Exception:
        raise

    if not result:
        raise NoFaceDetectedError(
            "No face embedding was generated."
        )

    embedding = result[0].get("embedding")

    if not embedding:
        raise NoFaceDetectedError(
            "No face embedding was generated."
        )

    return embedding


def embedding_to_json(embedding: list) -> str:
    """Convert embedding to JSON for database storage."""
    return json.dumps(embedding)


def embedding_from_json(embedding_json: str) -> np.ndarray:
    """Convert JSON embedding back into a NumPy array."""
    return np.array(
        json.loads(embedding_json),
        dtype=np.float32,
    )


def cosine_similarity(vec_a, vec_b) -> float:
    """Calculate cosine similarity between two embeddings."""

    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom == 0:
        return 0.0

    return float(
        np.dot(a, b) / denom
    )


def similarity_score_percent(vec_a, vec_b) -> float:
    """
    Convert cosine similarity into a 0-100 score
    for displaying in the dashboard.
    """

    similarity = cosine_similarity(
        vec_a,
        vec_b,
    )

    similarity = max(
        0.0,
        min(1.0, similarity),
    )

    return round(
        similarity * 100,
        2,
    )


def is_match(
    vec_a,
    vec_b,
    threshold: float = MATCH_THRESHOLD,
) -> bool:
    """Return True when similarity reaches the configured threshold."""

    return (
        cosine_similarity(vec_a, vec_b)
        >= threshold
    )