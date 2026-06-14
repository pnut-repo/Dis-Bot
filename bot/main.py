"""
main.py — FastAPI entry point for Render deployment.

Responsibilities:
    1. Start the Discord bot as an asyncio task on the SAME event loop as Uvicorn.
       This avoids "Event loop is closed" errors from threading.
    2. Start APScheduler (midnight pipeline) in a background thread.
    3. Serve the REST API for the frontend.
    4. Expose GET /health for UptimeRobot to keep Render awake.

Start command (render.yaml):
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load .env when running locally (no-op in production where env vars are set natively)
load_dotenv()

# Local imports — all relative to the bot/ directory (Render rootDir: bot)
from bot.collector import client as discord_client
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: startup + shutdown ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    On startup:
        - Creates the Discord bot task on the shared asyncio event loop.
        - Starts APScheduler in a background thread.

    On shutdown:
        - Closes the Discord WebSocket cleanly.
        - Cancels the bot task.
    """
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        logger.warning(
            "DISCORD_TOKEN not set — Discord bot will NOT start. "
            "This is expected in local dev without a token."
        )
        bot_task = None
    else:
        # Start Discord bot as an async task on the SAME event loop as FastAPI.
        # NEVER use threading.Thread — discord.py requires asyncio.
        bot_task = asyncio.create_task(discord_client.start(token))
        logger.info("Discord bot task created on shared event loop.")

    # Start the midnight pipeline scheduler (sync APScheduler in background thread)
    # Only import here to keep startup fast if scheduler is not needed
    try:
        from pipeline.orchestrator import start_scheduler
        scheduler = start_scheduler()
        logger.info("APScheduler started — midnight pipeline at 00:05 UTC.")
    except ImportError:
        scheduler = None
        logger.warning("pipeline.orchestrator not found — scheduler not started.")

    yield  # FastAPI serves requests

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    if scheduler:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler stopped.")

    if bot_task:
        await discord_client.close()
        bot_task.cancel()
        try:
            await bot_task
        except asyncio.CancelledError:
            pass
        logger.info("Discord bot shut down cleanly.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Discord Sentiment Bot API",
    description="REST API for the Discord sentiment analysis dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Tighten to your Netlify domain in production:
    # allow_origins=["https://your-app.netlify.app"]
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["infra"])
def health():
    """
    UptimeRobot pings this endpoint every 5 minutes to keep Render awake.
    Render free tier sleeps after 15 minutes of HTTP inactivity.
    """
    return {"status": "ok"}
