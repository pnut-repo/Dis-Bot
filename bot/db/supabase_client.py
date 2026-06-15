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

    Uses upsert with ignore_duplicates so that if the bot restarts and
    Discord replays recent messages, duplicates are silently skipped
    instead of raising a 23505 unique-constraint error.

    Payload shape (mirrors Supabase messages schema):
        message_id, user_id, username, display_name, content,
        created_at, has_attachment, is_reply, reply_to_id
    """
    get_client().table("messages").upsert(
        payload, on_conflict="message_id", ignore_duplicates=True
    ).execute()


def fetch_messages_for_date(date_str: str) -> list[dict]:
    """
    Return ALL messages for a given UTC date (YYYY-MM-DD), ordered by created_at.

    Supabase PostgREST has a default limit of 1000 rows per SELECT.
    This function paginates through all pages using .range() to ensure
    no messages are silently dropped on busy days (3000+ messages).
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
            break  # Last page — fewer rows than page_size means no more data

        offset += page_size

    pages = (offset // page_size) + 1
    logger.info(f"Fetched {len(all_messages)} messages for {date_str} ({pages} pages)")
    return all_messages


def bulk_update_message_topics(updates: list[dict], batch_size: int = 100) -> None:
    """
    Write topic_id, sentiment_label, sentiment_score back to messages after ML pipeline.

    Uses batched upsert: 100 messages per HTTP request instead of 1 per request.
    Each upsert payload includes ALL original message columns plus the ML-output
    columns, so NOT NULL constraints are satisfied.

    Args:
        updates: List of FULL message dicts (original Supabase row merged with
                 ML output columns: topic_id, sentiment_label, sentiment_score).
        batch_size: Number of rows per upsert request (default 100).
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
    """
    Replace all topic rows for a given report_date.
    Deletes any existing rows for the date before inserting fresh ones,
    so re-triggering the pipeline never creates duplicate topic entries.
    """
    if not rows:
        return
    report_date = rows[0]["report_date"]
    client = get_client()
    # Delete previous rows for this date (idempotent — safe if none exist)
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
    """
    Upsert per-user daily stats. ON CONFLICT (report_date, user_id) is the
    unique key — safe to re-run for the same date.
    """
    get_client().table("user_daily_stats").upsert(rows, on_conflict="report_date,user_id").execute()


# ── audit_log ─────────────────────────────────────────────────────────────────

def insert_audit_log(payload: dict) -> None:
    """
    Insert a single audit event into the audit_log table.

    Payload shape:
        clerk_user_id, email, display_name, nickname, event_type,
        event_meta (JSONB), ip_address, user_agent
    """
    get_client().table("audit_log").insert(payload).execute()

