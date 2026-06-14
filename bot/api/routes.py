"""
api/routes.py — REST API endpoints served to the Netlify frontend.

All endpoints are read-only GETs. The frontend calls these via VITE_API_BASE_URL.
Data comes from Supabase via db/supabase_client.py.

This module has zero business logic — it only queries Supabase and returns JSON.
All computation happens in the midnight pipeline.

Endpoints:
    GET /api/reports/dates          → list of available dates
    GET /api/reports/latest         → most recent report
    GET /api/reports/{report_date}  → full report for a date
    GET /api/topics/{report_date}   → topic rankings for a date
"""

import logging
from fastapi import APIRouter, HTTPException

from db.supabase_client import (
    fetch_available_report_dates,
    fetch_report_by_date,
    fetch_topics_by_date,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Report Dates ──────────────────────────────────────────────────────────────

@router.get("/reports/dates")
def get_report_dates():
    """
    Returns a list of dates (YYYY-MM-DD strings) that have reports.
    Used by the frontend date picker. Ordered newest-first. Max 14.
    """
    try:
        dates = fetch_available_report_dates()
        return {"dates": dates}
    except Exception as e:
        logger.error(f"Failed to fetch report dates: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch report dates")


# ── Latest Report (convenience) ──────────────────────────────────────────────
# Declared BEFORE /{report_date} so FastAPI doesn't match "latest" as a date.

@router.get("/reports/latest")
def get_latest_report():
    """
    Returns the most recent available report.
    Convenience endpoint so the frontend can load immediately without
    knowing the latest date.
    """
    try:
        dates = fetch_available_report_dates()
    except Exception as e:
        logger.error(f"Failed to fetch report dates: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    if not dates:
        raise HTTPException(status_code=404, detail="No reports available yet")

    latest_date = dates[0]  # Newest first
    report = fetch_report_by_date(latest_date)
    if not report:
        raise HTTPException(status_code=404, detail="Report data missing")

    return report


# ── Full Report for a Date ────────────────────────────────────────────────────

@router.get("/reports/{report_date}")
def get_report(report_date: str):
    """
    Returns the full daily report for a given date.

    Response shape:
    {
        "report_date": "2026-06-14",
        "total_messages": 8012,
        "total_users": 143,
        "total_topics": 12,
        "overall_sentiment": {"positive": 0.52, ...},
        "narrative_md": "# Daily Report...",
        "chart_data": {
            "hourly_volume": [...],
            "sentiment_overview": {...},
            "sentiment_by_hour": [...],
            "topic_engagement": [...],
            "user_activity": [...]
        },
        "pipeline_duration_seconds": 367
    }
    """
    try:
        report = fetch_report_by_date(report_date)
    except Exception as e:
        logger.error(f"Failed to fetch report for {report_date}: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    if not report:
        raise HTTPException(status_code=404, detail=f"No report found for {report_date}")

    return report


# ── Topics for a Date ─────────────────────────────────────────────────────────

@router.get("/topics/{report_date}")
def get_topics(report_date: str):
    """
    Returns all topics for a given date, ordered by engagement rank.

    Response shape:
    {
        "report_date": "2026-06-14",
        "topics": [
            {
                "topic_rank": 1,
                "topic_name": "python · async · loop",
                "message_count": 412,
                "unique_users": 38,
                "engagement_score": 0.87,
                "sentiment_dist": {"positive": 0.72, ...},
                ...
            }
        ]
    }
    """
    try:
        topics = fetch_topics_by_date(report_date)
    except Exception as e:
        logger.error(f"Failed to fetch topics for {report_date}: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    return {"report_date": report_date, "topics": topics}
