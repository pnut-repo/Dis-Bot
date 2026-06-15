"""
api/routes.py — REST API endpoints served to the Netlify frontend.

All report endpoints are read-only GETs. Data comes from Supabase.
The audit endpoint is a POST that logs user activity from authenticated
Clerk sessions.

Endpoints:
    GET  /api/reports/dates          → list of available dates
    GET  /api/reports/latest         → most recent report
    GET  /api/reports/{report_date}  → full report for a date
    GET  /api/topics/{report_date}   → topic rankings for a date
    GET  /api/users/{report_date}    → all users' stats for a date
    GET  /api/analytics/{report_date}→ expanded analytics for a date
    POST /api/audit                  → log user activity event
    POST /api/pipeline/run           → manually trigger the daily pipeline
"""

import json
import logging
import os
import hashlib
import threading

import jwt
import requests
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from db.supabase_client import (
    fetch_available_report_dates,
    fetch_report_by_date,
    fetch_topics_by_date,
    fetch_user_stats_by_date,
    fetch_analytics_by_date,
    insert_audit_log,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Clerk JWKS for JWT verification ───────────────────────────────────────────

_jwks_client = None


def _get_jwks_client():
    """
    Lazy-load the Clerk JWKS client. Clerk publishes its public keys at:
    https://<clerk-frontend-api>/.well-known/jwks.json

    The Frontend API domain is derived from the publishable key:
    pk_test_<base64-encoded-domain> → decode → "advanced-chamois-10.clerk.accounts.dev"
    """
    global _jwks_client
    if _jwks_client is None:
        pk = os.getenv("CLERK_PUBLISHABLE_KEY", "")
        if not pk:
            raise RuntimeError("CLERK_PUBLISHABLE_KEY is not set")

        # Extract the Clerk Frontend API domain from the publishable key
        # pk_test_<base64> or pk_live_<base64> → decode the base64 part
        import base64
        encoded_part = pk.split("_")[-1]
        # Add padding if needed
        padded = encoded_part + "=" * (4 - len(encoded_part) % 4)
        clerk_domain = base64.b64decode(padded).decode("utf-8").rstrip("$")

        jwks_url = f"https://{clerk_domain}/.well-known/jwks.json"
        _jwks_client = jwt.PyJWKClient(jwks_url)
        logger.info(f"Clerk JWKS client initialized from {jwks_url}")
    return _jwks_client


def verify_clerk_token(authorization: str) -> dict:
    """
    Verify a Clerk JWT from the Authorization header.
    Returns the decoded token claims (sub, email, name, etc.).
    Raises HTTPException 401 on failure.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.removeprefix("Bearer ").strip()

    try:
        jwks_client = _get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False},  # Clerk doesn't always set aud
        )
        return claims
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid Clerk JWT: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"JWT verification error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


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
    """
    try:
        topics = fetch_topics_by_date(report_date)
    except Exception as e:
        logger.error(f"Failed to fetch topics for {report_date}: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    return {"report_date": report_date, "topics": topics}


# ── User Stats ────────────────────────────────────────────────────────────────

@router.get("/users/{report_date}")
def get_user_stats(report_date: str):
    """
    Returns all users' daily stats for a given date.
    Used by the user dropdown in the dashboard analytics tab.
    """
    try:
        users = fetch_user_stats_by_date(report_date)
    except Exception as e:
        logger.error(f"Failed to fetch user stats for {report_date}: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    return {"report_date": report_date, "users": users}


# ── Analytics ─────────────────────────────────────────────────────────────────

@router.get("/analytics/{report_date}")
def get_analytics(report_date: str):
    """
    Returns expanded analytics data for a given date.
    Includes per-user hourly activity, sentiment breakdowns, and topic details.
    Retained for 30 days (even after messages are purged at 3 days).
    """
    try:
        analytics = fetch_analytics_by_date(report_date)
    except Exception as e:
        logger.error(f"Failed to fetch analytics for {report_date}: {e}")
        raise HTTPException(status_code=500, detail="Database error")

    if not analytics:
        raise HTTPException(status_code=404, detail=f"No analytics for {report_date}")

    return analytics


# ── Audit Logging ─────────────────────────────────────────────────────────────

VALID_EVENT_TYPES = {
    "login",
    "logout",
    "view_report",
    "view_topics",
    "change_date",
    "click_topic_card",
}


class AuditEvent(BaseModel):
    event_type: str
    event_meta: dict = {}


@router.post("/audit")
def log_audit_event(event: AuditEvent, request: Request):
    """
    Logs a user activity event to the audit_log Supabase table.

    The Clerk JWT is verified from the Authorization header. User email,
    ID, and name are extracted from the token claims — the frontend
    cannot forge these.

    Request body:
        { "event_type": "view_report", "event_meta": {"report_date": "2026-06-14"} }
    """
    # Verify the Clerk JWT
    auth_header = request.headers.get("authorization", "")
    claims = verify_clerk_token(auth_header)

    # Validate event type
    if event.event_type not in VALID_EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event_type. Must be one of: {', '.join(sorted(VALID_EVENT_TYPES))}"
        )

    # Extract user info from Clerk JWT claims
    # Clerk JWTs have: sub (user ID), email, name, username
    clerk_user_id = claims.get("sub", "unknown")

    # Clerk stores user metadata differently — email might be in claims directly
    # or in the session claims. We also get it from the frontend event_meta as fallback.
    email = (
        claims.get("email")
        or claims.get("primary_email")
        or event.event_meta.get("email", "unknown")
    )
    display_name = (
        claims.get("name")
        or claims.get("full_name")
        or event.event_meta.get("name", "")
    )
    nickname = (
        claims.get("username")
        or event.event_meta.get("nickname", "")
    )

    # Build audit payload
    raw_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")
    raw_meta = json.dumps(event.event_meta) if event.event_meta else "{}"
    
    payload = {
        "clerk_user_id": clerk_user_id,
        "email": hashlib.sha256(email.encode("utf-8")).hexdigest(),
        "display_name": display_name,
        "nickname": nickname,
        "event_type": event.event_type,
        "event_meta": json.dumps({"hashed": hashlib.sha256(raw_meta.encode("utf-8")).hexdigest()}),
        "ip_address": hashlib.sha256(raw_ip.encode("utf-8")).hexdigest(),
        "user_agent": request.headers.get("user-agent", "unknown"),
    }

    try:
        insert_audit_log(payload)
        logger.info(f"[audit] {event.event_type} by {email} ({clerk_user_id})")
    except Exception as e:
        logger.error(f"Failed to insert audit event: {e}")
        raise HTTPException(status_code=500, detail="Failed to log event")

    return {"status": "logged"}


# ── Manual Pipeline Trigger ───────────────────────────────────────────────────

@router.post("/pipeline/run")
def trigger_pipeline(request: Request):
    """
    Manually trigger the daily pipeline. Protected by a shared secret
    (PIPELINE_SECRET env var) to prevent unauthorized triggers.

    Usage:
        curl -X POST https://your-app.onrender.com/api/pipeline/run \
             -H "X-Pipeline-Secret: your-secret-here"
    """
    expected = os.getenv("PIPELINE_SECRET", "")
    provided = request.headers.get("x-pipeline-secret", "")

    if not expected or provided != expected:
        raise HTTPException(status_code=403, detail="Forbidden")

    from pipeline.orchestrator import run_daily_pipeline

    thread = threading.Thread(target=run_daily_pipeline, daemon=True)
    thread.start()

    logger.info("[manual] Pipeline triggered via /api/pipeline/run")
    return {"status": "started", "message": "Pipeline is running in the background. Check logs for progress."}
