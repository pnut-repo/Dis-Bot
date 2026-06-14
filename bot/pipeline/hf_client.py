"""
pipeline/hf_client.py — Render-side client for the HuggingFace Space.

Responsibilities:
    1. Wake up the HF Space before sending the main payload (it can sleep after
       48-72h of inactivity; cold start takes 2-4 minutes).
    2. Send all of a day's messages in ONE POST to /analyze.
    3. Return (sentiments, topic_labels) to the orchestrator.
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
                    f"models: {data.get('models', 'unknown')}"
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


def analyze_messages(
    messages: list[dict],
    hdbscan_params: dict | None = None,
) -> tuple[list[dict], list[int]]:
    """
    Sends ALL messages for a given day to the HF Space in a single POST request.

    The HF Space handles internal batching for inference and runs HDBSCAN on
    all embeddings. Embeddings are computed and clustered on the HF Space —
    they are never sent back to Render (keeps response ~800 KB vs ~25 MB).

    Args:
        messages:       List of Supabase message dicts (must have message_id,
                        content, user_id, created_at).
        hdbscan_params: Optional override for HDBSCAN hyperparameters.

    Returns:
        sentiments:   [{id, label, score}, ...] in the same order as messages.
        topic_labels: [int, ...] — HDBSCAN cluster labels (-1 = noise).
    """
    if not messages:
        return [], []

    if hdbscan_params is None:
        hdbscan_params = {
            "min_cluster_size": 8,
            "min_samples": 3,
            "cluster_selection_epsilon": 0.3,
        }

    # Build the exact payload shape that HF Space /analyze expects
    payload = {
        "messages": [
            {
                "id":        m["message_id"],
                "content":   m["content"],
                "user_id":   m["user_id"],
                "timestamp": m["created_at"],
            }
            for m in messages
        ],
        "hdbscan_params": hdbscan_params,
    }

    n = len(messages)
    logger.info(
        f"Sending {n} messages to HF Space for sentiment + embedding + HDBSCAN..."
    )

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
            f"{data['n_topics']} topics, {data['n_noise']} noise, "
            f"{data['processing_time_seconds']}s"
        )

        sentiments   = data["sentiments"]    # [{id, label, score}, ...] × n
        topic_labels = data["topic_labels"]  # [int, ...] × n

        return sentiments, topic_labels

    except requests.exceptions.Timeout:
        logger.error(
            f"HF Space request timed out after 600s for {n} messages. "
            f"Returning fallback neutral/noise labels."
        )
    except requests.exceptions.HTTPError as e:
        logger.error(f"HF Space returned HTTP error: {e}")
    except Exception as e:
        logger.error(f"HF Space request failed unexpectedly: {e}")

    # Fallback: never crash the pipeline. Return neutral sentiment and all-noise
    # topic labels so the rest of the orchestrator can still write a basic report.
    fallback_sentiments = [
        {"id": m["message_id"], "label": "neutral", "score": 0.5}
        for m in messages
    ]
    fallback_labels = [-1] * n
    return fallback_sentiments, fallback_labels
