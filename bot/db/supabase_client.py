"""
db/supabase_client.py — Supabase database helper layer.

Provides typed insert/fetch functions for all tables:
  - messages          (3-day retention)
  - daily_reports     (30-day retention)
  - topic_stats       (30-day retention)
  - user_daily_stats  (30-day retention)
  - daily_analytics   (30-day retention — expanded chart data for frontend)
  - audit_log         (indefinite)

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
    Uses upsert with ignore_duplicates so duplicates are silently skipped.
    """
    get_client().table("messages").upsert(
        payload, on_conflict="message_id", ignore_duplicates=True
    ).execute()


def fetch_messages_for_date(date_str: str) -> list[dict]:
    """
    Return ALL messages for a given UTC date (YYYY-MM-DD), ordered by created_at.
    Paginates through all pages using .range() (PostgREST default limit = 1000).
    """
    client = get_client()
    all_messages = []
    page_size = 1000
    offset = 0

    while True:
        result = (
            client
            .table("messages")
            .select("message_id, user_id, username, display_name, content, created_at, is_reply, reply_to_id")
            .gte("created_at", f"{date_str}T00:00:00+00:00")
            .lte("created_at", f"{date_str}T23:59:59+00:00")
            .order("created_at")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        page = result.data
        all_messages.extend(page)

        if len(page) < page_size:
            break

        offset += page_size

    pages = (offset // page_size) + 1
    logger.info(f"Fetched {len(all_messages)} messages for {date_str} ({pages} pages)")
    return all_messages


def bulk_update_message_topics(updates: list[dict], batch_size: int = 100) -> None:
    """
    Write topic_id, sentiment_label, sentiment_score back to messages.
    Uses batched upsert: 100 messages per HTTP request.
    """
    if not updates:
        return

    client = get_client()
    n_batches = 0

    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]
        client.table("messages").upsert(
            batch, on_conflict="message_id"
        ).execute()
        n_batches += 1

    logger.info(
        f"Updated {len(updates)} messages in {n_batches} batches "
        f"(batch_size={batch_size})"
    )


# ── daily_reports ─────────────────────────────────────────────────────────────

def upsert_daily_report(payload: dict) -> None:
    """Write (or overwrite) the daily report for a given date."""
    get_client().table("daily_reports").upsert(payload, on_conflict="report_date").execute()


def fetch_available_report_dates() -> list[str]:
    """Return the 30 most recent dates that have reports, newest first."""
    result = (
        get_client()
        .table("daily_reports")
        .select("report_date")
        .order("report_date", desc=True)
        .limit(30)
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
    """
    Replace all topic rows for a given report_date.
    Deletes existing rows before inserting fresh ones (idempotent).
    """
    if not rows:
        return
    report_date = rows[0]["report_date"]
    client = get_client()
    client.table("topic_stats").delete().eq("report_date", report_date).execute()
    client.table("topic_stats").insert(rows).execute()
    logger.info(f"Replaced topic_stats for {report_date} ({len(rows)} topics)")


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
    """Upsert per-user daily stats. ON CONFLICT (report_date, user_id)."""
    get_client().table("user_daily_stats").upsert(rows, on_conflict="report_date,user_id").execute()


def fetch_user_stats_by_date(date_str: str) -> list[dict]:
    """Return all user stats for a date, ordered by message_count descending."""
    return (
        get_client()
        .table("user_daily_stats")
        .select("*")
        .eq("report_date", date_str)
        .order("message_count", desc=True)
        .execute()
        .data
    )


# ── daily_analytics (NEW — 30-day retention) ─────────────────────────────────

def upsert_daily_analytics(payload: dict) -> None:
    """
    Write (or overwrite) the daily analytics for a given date.
    This table stores expanded chart data including per-user analytics
    for the user dropdown feature. Retained for 30 days.
    """
    get_client().table("daily_analytics").upsert(payload, on_conflict="report_date").execute()


def fetch_analytics_by_date(date_str: str) -> dict | None:
    """Return the daily_analytics row for a given date, or None."""
    result = (
        get_client()
        .table("daily_analytics")
        .select("*")
        .eq("report_date", date_str)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


# ── audit_log ─────────────────────────────────────────────────────────────────

def insert_audit_log(payload: dict) -> None:
    """Insert a single audit event into the audit_log table."""
    get_client().table("audit_log").insert(payload).execute()
