"""
pipeline/topic_analyzer.py — TF-IDF topic naming + metadata assembly
======================================================================
HDBSCAN clustering runs on the HF Space. This module receives the already-
computed `topic_labels` list and builds rich per-cluster metadata:

    - TF-IDF keywords → human-readable topic name
    - Earliest sender → initiator
    - Participant list + counts
    - Duration (first → last message timestamp)
    - Sentiment distribution
    - Reply density, positive fraction (used by engagement scorer)
"""

import logging
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


def extract_topic_metadata(
    messages: list[dict],
    topic_labels: list[int],
    sentiments: dict,
) -> list[dict]:
    """
    Builds a rich metadata dict for each HDBSCAN cluster.

    Args:
        messages:     List of Supabase message dicts (content, user_id, etc.)
                      — same order as topic_labels.
        topic_labels: HDBSCAN labels from HF Space (-1 = noise, excluded).
        sentiments:   {message_id: {id, label, score}} lookup dict.

    Returns:
        List of topic metadata dicts, one per cluster.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # ── Group messages by topic, skip noise ──────────────────────────────────
    topic_groups: dict[int, list[dict]] = defaultdict(list)
    for msg, label in zip(messages, topic_labels):
        if label != -1:
            topic_groups[label].append(msg)

    if not topic_groups:
        logger.warning("No topics found (all noise). Check HDBSCAN min_cluster_size.")
        return []

    # ── TF-IDF across all topic documents ────────────────────────────────────
    # Each "document" is the concatenated text of all messages in a cluster.
    all_topic_ids = list(topic_groups.keys())
    all_docs = [
        " ".join(m["content"] for m in topic_groups[tid])
        for tid in all_topic_ids
    ]

    # Discord-specific stopwords (common filler words that carry no topic signal)
    discord_stopwords = [
        "the", "is", "it", "in", "of", "to", "a", "and", "you", "i", "me", "my",
        "we", "they", "that", "this", "was", "are", "be", "have", "has", "do",
        "will", "just", "for", "on", "at", "with", "he", "she", "but", "not",
        "or", "so", "if", "as", "an", "by", "no",
        "lol", "lmao", "yeah", "yep", "ok", "okay", "hey", "oh", "ah", "umm",
        "haha", "omg", "like", "get", "got", "can", "don't", "its", "im",
        "ur", "u", "r",
    ]

    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),   # unigrams and bigrams
        min_df=2,              # term must appear in at least 2 topic docs
        stop_words=discord_stopwords,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        logger.warning(f"TF-IDF failed: {e} — falling back to topic_N names")
        feature_names = []
        tfidf_matrix  = None

    # ── Build per-topic metadata ──────────────────────────────────────────────
    results = []
    for idx, topic_id in enumerate(all_topic_ids):
        msgs = sorted(topic_groups[topic_id], key=lambda m: m["created_at"])

        # Keywords from TF-IDF row (top 5 terms by score)
        if tfidf_matrix is not None:
            row = tfidf_matrix[idx].toarray().flatten()
            top_indices = row.argsort()[::-1][:5]
            keywords = [feature_names[j] for j in top_indices if row[j] > 0]
        else:
            keywords = []

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
        topic_sents = [
            sentiments.get(m["message_id"], {"label": "neutral"})["label"]
            for m in msgs
        ]
        n = len(topic_sents)
        sentiment_dist = {
            "positive": round(topic_sents.count("positive") / n, 3),
            "neutral":  round(topic_sents.count("neutral")  / n, 3),
            "negative": round(topic_sents.count("negative") / n, 3),
        }

        # ── Engagement sub-signals (used by compute_engagement_scores) ─────
        reply_count       = sum(1 for m in msgs if m.get("is_reply"))
        reply_density     = reply_count / len(msgs)
        positive_fraction = sentiment_dist["positive"]

        results.append({
            "topic_id":           topic_id,
            "topic_name":         topic_name,
            "topic_keywords":     keywords,
            "message_count":      len(msgs),
            "message_ids":        [m["message_id"] for m in msgs],
            "unique_users":       unique_users,
            "duration_minutes":   duration_minutes,
            "peak_hour":          peak_hour,
            "sentiment_dist":     sentiment_dist,
            "reply_density":      reply_density,
            "positive_fraction":  positive_fraction,
            "initiator_user_id":  msgs[0]["user_id"],
            "initiator_username": msgs[0]["username"],
            "top_participants":   top_participants,
            "first_message_at":   first_ts,
            "last_message_at":    last_ts,
        })

    return results
