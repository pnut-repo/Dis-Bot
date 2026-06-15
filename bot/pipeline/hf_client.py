"""
pipeline/hf_client.py — Render-side client for the HuggingFace Space.

Responsibilities:
    1. Wake up the HF Space before sending the main payload (it can sleep after
       48-72h of inactivity; cold start takes 2-4 minutes).
    2. Send all of a day's messages in ONE POST to /analyze.
    3. Return the full pipeline result (topics, sentiments, labels) to the orchestrator.
    4. Fall back gracefully (neutral sentiment, all-noise labels) if the Space
       fails — the daily pipeline should never crash the bot entirely.

This module is synchronous (uses `requests`). The midnight orchestrator calls
it from a background thread via APScheduler, so blocking is fine.
"""

import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

HF_SPACE_URL     = os.getenv("HF_SPACE_URL", "").rstrip("/")
HF_SPACE_API_KEY = os.getenv("HF_SPACE_API_KEY", "")


def _auth_headers() -> dict:
    """Build Authorization header if HF_SPACE_API_KEY is set."""
    if HF_SPACE_API_KEY:
        return {"Authorization": f"Bearer {HF_SPACE_API_KEY}"}
    return {}


def wake_up_space(max_retries: int = 10, delay: int = 30) -> bool:
    """
    Polls /health until the Space responds 200 or retries are exhausted.

    HF Spaces sleep after 48-72h of inactivity. Cold start (Docker pull +
    ~520 MB model load) takes 2-4 minutes. 10 × 30s = 5-minute budget,
    which covers worst-case cold starts comfortably.

    Args:
        max_retries: How many times to poll before giving up.
        delay:       Seconds between each retry.

    Returns:
        True if the Space is awake, False if all retries failed.
    """
    if not HF_SPACE_URL:
        logger.error("HF_SPACE_URL is not set — cannot wake up Space.")
        return False

    logger.info(f"Waking up HF Space at {HF_SPACE_URL}...")

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(f"{HF_SPACE_URL}/health", timeout=60, headers=_auth_headers())
            if r.status_code == 200:
                data = r.json()
                logger.info(
                    f"HF Space is awake (attempt {attempt}) — "
                    f"pipeline: {data.get('pipeline', 'unknown')}"
                )
                return True
            logger.warning(
                f"Wake-up attempt {attempt}/{max_retries}: status={r.status_code}"
            )
        except requests.exceptions.Timeout:
            logger.warning(f"Wake-up attempt {attempt}/{max_retries}: timeout (60s)")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Wake-up attempt {attempt}/{max_retries}: connection error — {e}")
        except Exception as e:
            logger.warning(f"Wake-up attempt {attempt}/{max_retries}: {e}")

        if attempt < max_retries:
            logger.info(f"Space not ready — waiting {delay}s before retry...")
            time.sleep(delay)

    logger.error(
        f"HF Space did not wake up after {max_retries} attempts "
        f"({max_retries * delay // 60} minutes). Falling back to neutral/noise labels."
    )
    return False


def analyze_messages(messages: list[dict]) -> dict:
    """
    Sends ALL messages for a given day to the HF Space in a single POST request.

    The HF Space runs the full pipeline:
        Preprocessing → Context Enrichment → Embeddings → UMAP →
        HDBSCAN → Outlier Reassignment → c-TF-IDF → Sentiment Aggregation

    Args:
        messages: List of Supabase message dicts (must have message_id,
                  content, user_id, created_at, and optionally
                  referenced_message_id for reply chain enrichment).

    Returns:
        Full pipeline result dict with keys:
            - topics:                 [{topic_id, keywords, sentiment, ...}]
            - topic_labels:           [int, ...] per-message cluster labels
            - per_message_sentiment:  [{id, label, score}, ...]
            - n_topics:               int
            - uncategorized_count:    int
            - day_summary:            {tension_score, sentiment_distribution, ...}
            - processing_time_seconds: float

        On failure: returns a fallback dict with neutral sentiment and all-noise labels.
    """
    if not messages:
        return _fallback_result([])

    # Build the exact payload shape that HF Space /analyze expects
    payload = {
        "messages": [
            {
                "id":           m["message_id"],
                "content":      m["content"],
                "user_id":      m["user_id"],
                "timestamp":    m["created_at"],
                "reference_id": m.get("reply_to_id"),  # Reply chain for context enrichment
            }
            for m in messages
        ],
        # Use defaults for UMAP/HDBSCAN params — can be overridden via env vars later
    }

    n = len(messages)
    logger.info(f"Sending {n} messages to HF Space for full pipeline analysis...")

    try:
        # 600s = 10 minutes. Generous timeout for 8k messages × ~5min processing.
        resp = requests.post(
            f"{HF_SPACE_URL}/analyze",
            json=payload,
            timeout=600,
            headers=_auth_headers(),
        )
        resp.raise_for_status()
        data = resp.json()

        logger.info(
            f"HF Space analysis complete — "
            f"{data['n_topics']} topics, {data['uncategorized_count']} uncategorized, "
            f"{data['processing_time_seconds']}s"
        )

        return data

    except requests.exceptions.Timeout:
        logger.error(
            f"HF Space request timed out after 600s for {n} messages. "
            f"Returning fallback neutral/noise labels."
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"HF Space returned HTTP error: {e}")
    except Exception as e:
        logger.error(f"HF Space request failed unexpectedly: {e}")

    return _fallback_result(messages)


def _fallback_result(messages: list[dict]) -> dict:
    """
    Build a fallback result when the HF Space is unreachable.
    Returns neutral sentiment and all-noise topic labels so the
    rest of the orchestrator can still write a basic report.
    """
    n = len(messages)
    return {
        "count": n,
        "processing_time_seconds": 0,
        "n_topics": 0,
        "uncategorized_count": n,
        "uncategorized_pct": 100.0,
        "topics": [],
        "topic_labels": [-1] * n,
        "per_message_sentiment": [
            {"id": m["message_id"], "label": "neutral", "score": 0.5}
            for m in messages
        ],
        "day_summary": {
            "total_topics": 0,
            "dominant_topic_keywords": [],
            "most_negative_topic_keywords": [],
            "day_tension_score": 0.0,
            "day_sentiment_distribution": {"positive": 0, "neutral": n, "negative": 0},
            "overall_dominant_sentiment": "neutral",
        },
    }
