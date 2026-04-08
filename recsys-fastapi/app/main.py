from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import logs
from app.api import recommendations
from app.db.session import Base, engine, DATABASE_URL
from app.db import models  # noqa: F401
from app.services.artefacts import load_artefacts
from app.services.recommender import recommender_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # For local dev/tests this makes the service usable without running Alembic.
    # In production, prefer migrations.
    if DATABASE_URL.startswith("sqlite"):
        # SQLite files used in tests/dev don't support schema migrations here.
        # Recreate schema so model changes (e.g., new columns) don't break.
        Base.metadata.drop_all(bind=engine)

    Base.metadata.create_all(bind=engine)

    # Load RecSys artefacts once (embeddings + Faiss index).
    # If loading fails (missing deps/files), the API still works with heuristic recs.
    try:
        artefacts = load_artefacts()
        # Keep a reference on app.state for endpoints that may want it.
        app.state.recommender_artefacts = artefacts

        # Tests may inject their own artefacts before the app starts; don't override.
        if getattr(recommender_service, "_artefacts", None) is None:
            recommender_service.set_artefacts(artefacts)
    except Exception:
        pass
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(logs.router)
app.include_router(recommendations.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the User Action Logging API!"}