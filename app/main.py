import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import Base, engine
from .routers import search,persons,stats, auth_router

# Create tables on startup (fine for SQLite / a PBL-scale project;
# swap for Alembic migrations if this ever needs to survive schema changes in prod).
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="TraceAI API",
    description="Backend for TraceAI - AI-powered missing person safety platform. "
                 "Handles case reporting, face-matching search, and dashboard stats.",
    version="1.0.0",
)

# CORS: allow your deployed frontend + local dev. Update FRONTEND_ORIGINS env var
# on your host (comma-separated) once you know your final frontend URL.
origins_env = os.getenv(
    "FRONTEND_ORIGINS",
    "https://ai-missing-person-safety-app.vercel.app,http://localhost:3000,http://localhost:5173",
)
origins = [o.strip() for o in origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve uploaded photos directly, e.g. GET /uploads/persons/<file>.jpg
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.include_router(auth_router.router)
app.include_router(persons.router)
app.include_router(stats.router)
app.include_router(search.router)


@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "TraceAI API"}
