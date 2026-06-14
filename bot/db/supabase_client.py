"""
db/supabase_client.py — Supabase database helper layer.

Provides typed insert/fetch functions for all 4 tables.
Uses a singleton client to avoid re-creating connections on every call.
The service role key bypasses Row Level Security (RLS) — only used on Render, never frontend.
"""

import os
import logging
from supabase import create_client, Client

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set as environment variables."
            )
        _client = create_client(url, key)
        logger.info("Supabase client initialized.")
    return _client


# ── messages ─────────────────────────────────────────────────────────────────

def insert_message(payload: dict) -> None:
    """
    Insert a single Discord message into the messages table.

    Payload shape (mirrors Supabase messages schema):
        message_id, user_id, username, display_name, content,
        created_at, has_attachment, is_reply, reply_to_id
    """
    get_client().table("messages").insert(payload).execute()


def fetch_messages_for_date(date_str: str) -> list[dict]:
    """
    Return all messages for a given UTC date (YYYY-MM-DD), ordered by created_at.
    Used by the midnight pipeline to fetch yesterday's messages.
    """
    result = (
        get_client()
        .table("messages")
        .select("message_id, user_id, username, display_name, content, created_at, is_reply")
        .gte("created_at", f"{date_str}T00:00:00+00:00")
        .lte("created_at", f"{date_str}T23:59:59+00:00")
        .order("created_at")
        .execute()
    )
    return result.data


def bulk_update_message_topics(updates: list[dict]) -> None:
    """
    Write topic_id, sentiment_label, sentiment_score back to messages after ML pipeline.

    updates = [
        {"message_id": "...", "topic_id": 3, "sentiment_label": "positive", "sentiment_score": 0.91},
        ...
    ]
    Uses upsert so it's safe to re-run.
    """
    get_client().table("messages").upsert(updates).execute()


# ── daily_reports ─────────────────────────────────────────────────────────────

def upsert_daily_report(payload: dict) -> None:
    """
    Write (or overwrite) the daily report for a given date.
    ON CONFLICT report_date ensures the pipeline is idempotent.
    """
    get_client().table("daily_reports").upsert(payload, on_conflict="report_date").execute()


def fetch_available_report_dates() -> list[str]:
    """Return the 14 most recent dates that have reports, newest first."""
    result = (
        get_client()
        .table("daily_reports")
        .select("report_date")
        .order("report_date", desc=True)
        .limit(14)
        .execute()
    )
    return [r["report_date"] for r in result.data]


def fetch_report_by_date(date_str: str) -> dict | None:
    """Return the full daily_reports row for the given date, or None if missing."""
    result = (
        get_client()
        .table("daily_reports")
        .select("*")
        .eq("report_date", date_str)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


# ── topic_stats ───────────────────────────────────────────────────────────────

def insert_topic_stats(rows: list[dict]) -> None:
    """Insert all topic rows for a given report_date."""
    get_client().table("topic_stats").insert(rows).execute()


def fetch_topics_by_date(date_str: str) -> list[dict]:
    """Return all topics for a date, ordered by engagement rank."""
    return (
        get_client()
        .table("topic_stats")
        .select("*")
        .eq("report_date", date_str)
        .order("topic_rank")
        .execute()
        .data
    )


# ── user_daily_stats ──────────────────────────────────────────────────────────

def insert_user_daily_stats(rows: list[dict]) -> None:
    """
    Upsert per-user daily stats. ON CONFLICT (report_date, user_id) is the
    unique key — safe to re-run for the same date.
    """
    get_client().table("user_daily_stats").upsert(rows, on_conflict="report_date,user_id").execute()
