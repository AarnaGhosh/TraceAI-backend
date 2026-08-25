"""
Lightweight face detection + recognition for TraceAI.

Uses:
- OpenCV YuNet for face detection
- OpenCV SFace for face embeddings
- Trained 500-person face database for recognition

No TensorFlow or DeepFace required.
"""

import json
import os
import pickle
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


# ---------------------------------------------------------
# Model paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

YUNET_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face_detection_yunet_2023mar.onnx",
)

SFACE_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "face_recognition_sface_2021dec.onnx",
)

FACE_DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "face_database.pkl",
)


# ---------------------------------------------------------
# Matching threshold
# ---------------------------------------------------------

MATCH_THRESHOLD = float(
    os.getenv("FACE_MATCH_THRESHOLD", "0.363")
)


# ---------------------------------------------------------
# Exceptions
# ---------------------------------------------------------

class NoFaceDetectedError(Exception):
    """Raised when no face can be detected."""
    pass


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded."""
    pass


# ---------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------

_face_detector = None
_face_recognizer = None
_face_database = None


# ---------------------------------------------------------
# Load face database
# ---------------------------------------------------------

def get_face_database():
    """
    Load the trained 500-person face embedding database.
    """

    global _face_database

    if _face_database is None:

        if not os.path.exists(FACE_DATABASE_PATH):
            raise FileNotFoundError(
                f"Face database not found: {FACE_DATABASE_PATH}"
            )

        with open(FACE_DATABASE_PATH, "rb") as file:
            _face_database = pickle.load(file)

        print(
            f"Loaded face database with "
            f"{len(_face_database)} persons."
        )

    return _face_database


# ---------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------

def get_models():
    """
    Load YuNet and SFace only when face processing is needed.
    """

    global _face_detector
    global _face_recognizer

    if _face_detector is None:

        if not os.path.exists(YUNET_MODEL):
            raise FileNotFoundError(
                f"YuNet model not found: {YUNET_MODEL}"
            )

        _face_detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL,
            "",
            (320, 320),
            0.6,
            0.3,
            5000,
        )

    if _face_recognizer is None:

        if not os.path.exists(SFACE_MODEL):
            raise FileNotFoundError(
                f"SFace model not found: {SFACE_MODEL}"
            )

        _face_recognizer = cv2.FaceRecognizerSF.create(
            SFACE_MODEL,
            "",
        )

    return _face_detector, _face_recognizer


# ---------------------------------------------------------
# Image loading
# ---------------------------------------------------------

def load_image(image_source: str) -> np.ndarray:
    """
    Load an image from:
    - local file path
    - public HTTP/HTTPS URL
    """

    if not image_source:
        raise ImageLoadError(
            "No image path was provided."
        )

    parsed = urlparse(image_source)

    # Public URL
    if parsed.scheme in ("http", "https"):

        try:

            response = requests.get(
                image_source,
                timeout=20,
            )

            response.raise_for_status()

            content = response.content

            if not content:
                raise ImageLoadError(
                    "Downloaded image is empty."
                )

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
                    "Downloaded file could not be decoded."
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
            "Local file could not be decoded."
        )

    return image


# ---------------------------------------------------------
# Face detection
# ---------------------------------------------------------

def detect_face(image: np.ndarray):
    """
    Detect the largest face in an image.

    Returns:
        face box + landmarks

    Raises:
        NoFaceDetectedError
    """

    detector, _ = get_models()

    height, width = image.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:

        raise NoFaceDetectedError(
            "No face could be detected."
        )

    # Select the largest detected face
    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    return face


# ---------------------------------------------------------
# Face embedding
# ---------------------------------------------------------

def get_embedding(image_source: str) -> list:
    """
    Detect the largest face and generate
    an SFace embedding.
    """

    image = load_image(image_source)

    _, recognizer = get_models()

    face = detect_face(image)

    aligned_face = recognizer.alignCrop(
        image,
        face,
    )

    embedding = recognizer.feature(
        aligned_face
    )

    if embedding is None:

        raise NoFaceDetectedError(
            "Could not generate face embedding."
        )

    return embedding.flatten().astype(
        np.float32
    ).tolist()


# ---------------------------------------------------------
# JSON conversion
# ---------------------------------------------------------

def embedding_to_json(
    embedding: list
) -> str:

    return json.dumps(embedding)


def embedding_from_json(
    embedding_json: str
) -> np.ndarray:

    return np.array(
        json.loads(embedding_json),
        dtype=np.float32,
    )


# ---------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------

def cosine_similarity(
    vec_a,
    vec_b,
) -> float:

    a = np.asarray(
        vec_a,
        dtype=np.float32,
    )

    b = np.asarray(
        vec_b,
        dtype=np.float32,
    )

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom == 0:
        return 0.0

    return float(
        np.dot(a, b) / denom
    )


def similarity_score_percent(
    vec_a,
    vec_b,
) -> float:

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


# ---------------------------------------------------------
# Check if two embeddings match
# ---------------------------------------------------------

def is_match(
    vec_a,
    vec_b,
    threshold: float = MATCH_THRESHOLD,
) -> bool:

    return (
        cosine_similarity(
            vec_a,
            vec_b,
        )
        >= threshold
    )


# ---------------------------------------------------------
# Find best person match from 500-person database
# ---------------------------------------------------------

def find_best_match(
    embedding: list,
):
    """
    Compare a face embedding against all persons
    in the trained face database.

    Returns:
        person_id
        similarity
        confidence
        matched
    """

    database = get_face_database()

    best_person_id = None
    best_similarity = -1.0

    query_embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    # Compare against every stored embedding
    for person_id, embeddings in database.items():

        for stored_embedding in embeddings:

            similarity = cosine_similarity(
                query_embedding,
                stored_embedding,
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_person_id = str(person_id)

    # Convert similarity to percentage
    confidence_percent = round(
        max(
            0.0,
            min(1.0, best_similarity)
        ) * 100,
        2,
    )

    # Check threshold
    matched = best_similarity >= MATCH_THRESHOLD

    return {
        "person_id": best_person_id,
        "similarity": round(
            float(best_similarity),
            4,
        ),
        "confidence": confidence_percent,
        "matched": matched,
    }


# ---------------------------------------------------------
# Recognize person directly from image
# ---------------------------------------------------------

def recognize_person(
    image_source: str,
):
    """
    Complete recognition pipeline:

    Image
        ↓
    YuNet face detection
        ↓
    SFace embedding
        ↓
    Compare with 500-person database
        ↓
    Return best match
    """

    embedding = get_embedding(
        image_source
    )

    result = find_best_match(
        embedding
    )

    return result