"""
pipeline/topic_analyzer.py — Topic metadata assembly from HF Space results
============================================================================
The HF Space now runs the full ML pipeline (UMAP, HDBSCAN, c-TF-IDF, sentiment
aggregation). This module receives the already-computed topic data and builds
rich per-cluster metadata for the orchestrator:

    - Uses HF Space's c-TF-IDF keywords → human-readable topic name
    - Earliest sender → initiator
    - Participant list + counts
    - Duration (first → last message timestamp)
    - Sentiment distribution (from HF Space's per-topic aggregation)
    - Reply density, positive fraction (used by engagement scorer)

No local TF-IDF needed — the HF Space handles keyword extraction via c-TF-IDF.
"""

import logging
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


def extract_topic_metadata(
    messages: list[dict],
    topic_labels: list[int],
    sentiments: dict,
    hf_topics: list[dict],
) -> list[dict]:
    """
    Builds rich metadata for each topic cluster, combining HF Space's ML results
    (keywords, sentiment aggregation) with message-level metadata (participants,
    timing, reply density).

    Args:
        messages:     List of Supabase message dicts (content, user_id, etc.)
                      — same order as topic_labels.
        topic_labels: Cluster labels from HF Space (-1 = noise, excluded).
        sentiments:   {message_id: {id, label, score}} lookup dict.
        hf_topics:    Topic objects from HF Space response — each has
                      {topic_id, keywords, keyword_scores, sentiment,
                       message_count, representative_messages}.

    Returns:
        List of topic metadata dicts, one per cluster.
    """
    # ── Group messages by topic, skip noise ──────────────────────────────────
    topic_groups: dict[int, list[dict]] = defaultdict(list)
    for msg, label in zip(messages, topic_labels):
        if label != -1:
            topic_groups[label].append(msg)

    if not topic_groups:
        logger.warning("No topics found (all noise). Check HDBSCAN parameters.")
        return []

    # Build lookup for HF Space topic data by topic_id
    hf_topic_by_id = {t["topic_id"]: t for t in hf_topics}

    # ── Build per-topic metadata ──────────────────────────────────────────────
    results = []
    for topic_id, msgs in topic_groups.items():
        msgs = sorted(msgs, key=lambda m: m["created_at"])

        # ── Keywords from HF Space c-TF-IDF ──────────────────────────────────
        hf_topic = hf_topic_by_id.get(topic_id, {})
        keywords = hf_topic.get("keywords", [])

        # Human-readable topic name: top 3 keywords joined with ·
        topic_name = " · ".join(keywords[:3]) if keywords else f"topic_{topic_id}"

        # ── Time metrics ──────────────────────────────────────────────────────
        first_ts = msgs[0]["created_at"]
        last_ts  = msgs[-1]["created_at"]
        try:
            first_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            last_dt  = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            duration_minutes = (last_dt - first_dt).total_seconds() / 60
        except Exception:
            duration_minutes = 0

        hour_counts = Counter(
            datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")).hour
            for m in msgs
        )
        peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else 0

        # ── Participants ──────────────────────────────────────────────────────
        unique_users = len(set(m["user_id"] for m in msgs))

        user_info: dict[str, dict] = defaultdict(
            lambda: {"username": "", "message_count": 0}
        )
        for m in msgs:
            user_info[m["user_id"]]["username"]      = m["username"]
            user_info[m["user_id"]]["message_count"] += 1

        top_participants = sorted(
            [
                {
                    "user_id":       uid,
                    "username":      d["username"],
                    "message_count": d["message_count"],
                }
                for uid, d in user_info.items()
            ],
            key=lambda x: x["message_count"],
            reverse=True,
        )

        # ── Sentiment distribution ────────────────────────────────────────────
        # Use HF Space's pre-aggregated sentiment if available, else compute locally
        hf_sentiment = hf_topic.get("sentiment", {})
        if hf_sentiment:
            sentiment_dist = {
                "positive": round(hf_sentiment.get("pct_positive", 0) / 100, 3),
                "neutral":  round(hf_sentiment.get("pct_neutral", 0) / 100, 3),
                "negative": round(hf_sentiment.get("pct_negative", 0) / 100, 3),
            }
            tension_score = hf_sentiment.get("tension_score", 0)
            needs_moderation = hf_sentiment.get("needs_moderation_review", False)
            dominant_sentiment = hf_sentiment.get("dominant_sentiment", "neutral")
        else:
            # Fallback: compute locally from per-message sentiments
            topic_sents = [
                sentiments.get(m["message_id"], {"label": "neutral"})["label"]
                for m in msgs
            ]
            n_s = len(topic_sents)
            sentiment_dist = {
                "positive": round(topic_sents.count("positive") / n_s, 3),
                "neutral":  round(topic_sents.count("neutral")  / n_s, 3),
                "negative": round(topic_sents.count("negative") / n_s, 3),
            }
            tension_score = 0
            needs_moderation = False
            dominant_sentiment = max(sentiment_dist, key=sentiment_dist.get)

        # ── Engagement sub-signals (used by compute_engagement_scores) ─────
        reply_count       = sum(1 for m in msgs if m.get("is_reply"))
        reply_density     = reply_count / len(msgs)
        positive_fraction = sentiment_dist["positive"]

        results.append({
            "topic_id":             topic_id,
            "topic_name":           topic_name,
            "topic_keywords":       keywords,
            "message_count":        len(msgs),
            "message_ids":          [m["message_id"] for m in msgs],
            "unique_users":         unique_users,
            "duration_minutes":     duration_minutes,
            "peak_hour":            peak_hour,
            "sentiment_dist":       sentiment_dist,
            "tension_score":        tension_score,
            "needs_moderation":     needs_moderation,
            "dominant_sentiment":   dominant_sentiment,
            "reply_density":        reply_density,
            "positive_fraction":    positive_fraction,
            "initiator_user_id":    msgs[0]["user_id"],
            "initiator_username":   msgs[0]["username"],
            "top_participants":     top_participants,
            "first_message_at":     first_ts,
            "last_message_at":      last_ts,
            "representative_messages": hf_topic.get("representative_messages", []),
        })

    return results
