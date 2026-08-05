"""STAR SARA — AI Executive Assistant Platform, by STAR Technologies."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import routes, voice_ws
from .config import settings
from .services.stt import recognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Warm the transcription model in the background so the first utterance of
    # the session isn't the one that pays for the model load.
    warmup = asyncio.create_task(recognizer.warmup())
    yield
    warmup.cancel()


app = FastAPI(title="STAR SARA API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(routes.router)
app.include_router(voice_ws.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "assistant": settings.assistant_name}
