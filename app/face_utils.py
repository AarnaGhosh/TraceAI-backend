"""
Face detection + embedding + matching, powered by DeepFace (Facenet model).

Why Facenet + opencv detector:
- Facenet weights are small (~90MB) and download once, then cache — good for
  free-tier hosting with limited disk/bandwidth.
- The opencv detector backend needs no extra model download (unlike retinaface/mtcnn),
  keeping cold-start time low.

All embeddings are stored as JSON text in the database so we never need a
separate vector DB for a project this size.
"""
import json
import numpy as np

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "opencv"

DeepFace = None


def get_deepface():
    global DeepFace

    if DeepFace is None:
        from deepface import DeepFace as DF
        DeepFace = DF

    return DeepFace
# Empirically reasonable cosine-similarity cutoff for Facenet embeddings.
# Similarity is 0..1 here (1 = identical). Tune this after testing with real photos.
MATCH_THRESHOLD = 0.55


class NoFaceDetectedError(Exception):
    """Raised when DeepFace cannot find a face in the given image."""
    pass


def get_embedding(image_path: str) -> list:
    """
    Extract a face embedding vector from an image file.
    Raises NoFaceDetectedError if no face is found.
    """
    try:
        deepface = get_deepface()
        result = deepface.represent(
        img_path=image_path,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
    )
    except ValueError as e:
        # DeepFace raises ValueError when face detection fails
        raise NoFaceDetectedError(str(e))

    # DeepFace.represent returns a list (one entry per detected face); take the first/largest.
    embedding = result[0]["embedding"]
    return embedding


def embedding_to_json(embedding: list) -> str:
    return json.dumps(embedding)


def embedding_from_json(embedding_json: str) -> np.ndarray:
    return np.array(json.loads(embedding_json))


def cosine_similarity(vec_a, vec_b) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def similarity_score_percent(vec_a, vec_b) -> float:
    """Cosine similarity mapped to a friendly 0-100 percentage."""
    sim = cosine_similarity(vec_a, vec_b)
    # cosine similarity for face embeddings realistically ranges ~ -0.2 .. 1.0
    # clip and rescale so the UI shows a sane 0-100 percentage
    sim = max(0.0, min(1.0, sim))
    return round(sim * 100, 2)


def is_match(vec_a, vec_b, threshold: float = MATCH_THRESHOLD) -> bool:
    return cosine_similarity(vec_a, vec_b) >= threshold
