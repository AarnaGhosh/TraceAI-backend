# TraceAI Backend

Backend for **SafeTrace AI** — the missing person reporting & AI face-matching platform
(frontend: https://ai-missing-person-safety-app.vercel.app/).

Built with **FastAPI** + **DeepFace (Facenet)** for face matching + **SQLite** (swappable to Postgres).

This replaces the old Railway backend. It's been tested end-to-end: reporting a case,
uploading a photo to search, matching/non-matching results, login, and marking a case as found
all work correctly (verified during development — see test transcript if you want proof).

---

## How the matching actually works

1. When someone reports a missing person, their photo is run through **DeepFace** (Facenet model)
   to produce a 128-number "face embedding" — a mathematical fingerprint of the face. This is
   stored in the database alongside the case details (not the raw pixels being re-analyzed each time).
2. When someone uploads a "sighting" photo to search, the same embedding is extracted from it.
3. The sighting's embedding is compared against every stored missing-person embedding using
   **cosine similarity** — a score from 0-100%. Anything ≥ 55% is flagged `is_match: true`
   (this threshold is a constant in `app/face_utils.py` — tune it if you get too many
   false positives/negatives during testing).
4. The top 5 closest matches are returned, ranked by score.

No API keys, no external services, no cost — everything runs in the container.

---

## Project structure

```
app/
  main.py            # FastAPI app, CORS, static file serving, router wiring
  database.py         # SQLAlchemy engine/session (SQLite by default)
  models.py            # Person, Sighting, User tables
  schemas.py            # Pydantic request/response models
  face_utils.py          # DeepFace embedding extraction + cosine similarity matching
  auth.py                 # JWT auth (login-protected routes for the dashboard)
  utils.py                  # File upload helper
  routers/
    persons.py               # Report / list / update / delete missing-person cases
    search.py                  # THE core feature — upload a photo, get ranked matches
    stats.py                     # Powers the homepage counters
    auth_router.py                 # /register, /login, /me
Dockerfile                         # Configured for Hugging Face Spaces (port 7860)
requirements.txt
.env.example
```

---

## Deploy to Hugging Face Spaces (recommended, free)

**Why HF Spaces and not Render/Railway free tier:** DeepFace + TensorFlow need real memory headroom.
Render's free tier caps at 512MB RAM, which is not enough — the app will crash on the first face
detection. Hugging Face Spaces' free CPU tier gives you far more headroom and is built for exactly
this kind of ML workload.

### Steps

1. Go to https://huggingface.co/new-space
2. Name it (e.g. `traceai-backend`), choose **Docker** as the Space SDK, pick **Public** or **Private**.
3. Once created, clone the Space repo it gives you, or use the "Files" web UI to upload every file
   from this folder (keep the folder structure — `app/` must stay a folder).
4. In the Space's **Settings → Variables and secrets**, add:
   - `SECRET_KEY` — any long random string (generate one with `openssl rand -hex 32`)
   - `FRONTEND_ORIGINS` — `https://ai-missing-person-safety-app.vercel.app` (add more origins comma-separated if needed, e.g. `http://localhost:3000` for local frontend dev)
   - (Optional) `DATABASE_URL` if you want Postgres instead of the default SQLite
5. Push/upload — the Space will build the Dockerfile automatically. First build takes a few minutes
   (installing TensorFlow). Watch the "Logs" tab.
6. Once it says "Running", your API is live at:
   `https://<your-username>-<space-name>.hf.space`
7. Test it: visit `https://<your-username>-<space-name>.hf.space/docs` — you should see the
   interactive Swagger UI with every endpoint listed.

**Note on free-tier sleep:** Free Spaces sleep after inactivity and take ~30-60s to wake on the
next request (DeepFace model reload). That's fine for a PBL demo; just click "Run Analysis" once
to wake it before your actual demo/presentation.

### Alternative: Render (works, just size down the model)
If you'd rather use Render's free tier, switch `MODEL_NAME` in `app/face_utils.py` from `"Facenet"`
to a lighter option, and know it may still be tight on 512MB. Fly.io's free allowance (256MB-1GB
depending on plan) has the same constraint. HF Spaces is genuinely the path of least resistance here.

---

## Connecting your frontend

Update your frontend's API base URL to point at your deployed backend
(e.g. as a `NEXT_PUBLIC_API_URL` or `VITE_API_URL` env var, wherever it currently points to Railway).

| Frontend needs to... | Call |
|---|---|
| Submit the "Report Missing Person" form | `POST /api/persons` (multipart form: `name`, `age`, `gender`, `description`, `last_seen_location`, `last_seen_date`, `reporter_name`, `reporter_contact`, `image`) |
| Show the Dashboard list | `GET /api/persons` (optional `?status=missing` or `?status=found`) |
| Show homepage stat counters | `GET /api/stats` |
| "AI Analysis" — upload a photo to search | `POST /api/search` (multipart form: `image`, optional `location`, `reported_by`, `notes`) |
| Login | `POST /api/auth/login` (JSON: `email`, `password`) → returns `access_token` |
| Mark a case as found (dashboard action) | `PATCH /api/persons/{id}` with `Authorization: Bearer <token>` header |
| Display a stored photo | `GET /uploads/<image_path returned by the API>` |

Full interactive docs (auto-generated, always in sync with the code) are at `/docs` on
whatever URL you deploy to.

---

## Running locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # edit SECRET_KEY at minimum
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs to try every endpoint interactively.

First request that touches face matching will download the Facenet model weights (~90MB,
one-time, cached after that).

---

## For your SE Lab / SRS documentation

If you need this described for your IEEE 830 SRS docs: this is a **RESTful API backend**
following a **layered architecture** (routers → business logic in face_utils/auth →
SQLAlchemy ORM → SQLite persistence), using **JWT-based authentication** for protected
endpoints and **DeepFace's Facenet CNN model** for facial feature extraction with
**cosine similarity** as the matching metric. Happy to generate an actual architecture
diagram or update your SRS doc with this backend's endpoints/data model if that's useful —
just say the word.
