FROM python:3.12-slim

# OpenCV runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p uploads/persons uploads/sightings && \
    chown -R appuser:appuser /code

USER appuser

# Render provides the PORT environment variable.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]