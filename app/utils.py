import os
import uuid
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "person-images")


if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL and SUPABASE_KEY environment variables are required"
    )


supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)


async def save_upload(file: UploadFile, subfolder: str) -> str:
    """
    Uploads an image to Supabase Storage.

    Files are stored inside:
        person-images/persons/
        person-images/sightings/

    Returns the public URL of the uploaded image.
    """

    allowed_types = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }

    content_type = file.content_type or "image/jpeg"

    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are allowed."
        )

    extension = allowed_types[content_type]

    filename = f"{uuid.uuid4().hex}{extension}"

    storage_path = f"{subfolder}/{filename}"

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    max_size = 10 * 1024 * 1024

    if len(contents) > max_size:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum allowed size is 10 MB."
        )

    try:
        supabase.storage.from_(SUPABASE_BUCKET).upload(
            storage_path,
            contents,
            {
                "content-type": content_type,
                "upsert": False,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload image to Supabase Storage: {str(e)}"
        )

    public_url = supabase.storage.from_(
        SUPABASE_BUCKET
    ).get_public_url(storage_path)

    return public_url