import os
import uuid
from fastapi import UploadFile

UPLOAD_ROOT = os.getenv("UPLOAD_ROOT", "uploads")


async def save_upload(file: UploadFile, subfolder: str) -> str:
    """
    Saves an UploadFile to disk under uploads/<subfolder>/<uuid>.<ext>
    Returns the relative path (safe to store in DB and serve via /uploads static mount).
    """
    folder = os.path.join(UPLOAD_ROOT, subfolder)
    os.makedirs(folder, exist_ok=True)

    ext = os.path.splitext(file.filename or "")[1].lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"

    filename = f"{uuid.uuid4().hex}{ext}"
    full_path = os.path.join(folder, filename)

    contents = await file.read()
    with open(full_path, "wb") as f:
        f.write(contents)

    return full_path.replace("\\", "/")
