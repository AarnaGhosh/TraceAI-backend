# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860
# and prefers a non-root user. This also runs fine locally or on Render/Fly
# if you just change the exposed port / start command.
FROM python:3.10-slim

# System deps needed by opencv + deepface
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create a non-root user (required by Hugging Face Spaces) and give it
# ownership of the app dir so it can write uploads/, the SQLite file,
# and DeepFace's downloaded model weights cache.
RUN useradd -m -u 1000 appuser && \
    mkdir -p uploads/persons uploads/sightings && \
    chown -R appuser:appuser /code
USER appuser

# DeepFace/keras cache goes here by default (~/.deepface) - point it inside /code
# so it's writable regardless of platform home-dir quirks.
ENV DEEPFACE_HOME=/code

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
