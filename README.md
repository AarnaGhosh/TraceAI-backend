# TraceAI Backend

TraceAI is an AI-powered missing-person safety platform.

## Tech Stack

- FastAPI
- OpenCV YuNet for face detection
- OpenCV SFace for face recognition
- NumPy for embedding comparison
- PostgreSQL / Supabase
- SQLAlchemy

## Face Matching

When a missing-person case is reported, TraceAI:

1. Receives the person's photo.
2. Detects the face using OpenCV YuNet.
3. Generates a 128-dimensional face embedding using OpenCV SFace.
4. Stores the embedding in the database.

When a search photo is uploaded:

1. The face is detected.
2. An SFace embedding is generated.
3. The embedding is compared with stored missing-person embeddings.
4. The closest matches are returned with similarity scores.

The system uses lightweight ONNX models for face detection and recognition, making it suitable for limited-memory hosting.

## Models

The following models are included in the `models/` directory:

- `face_detection_yunet_2023mar.onnx`
- `face_recognition_sface_2021dec.onnx`