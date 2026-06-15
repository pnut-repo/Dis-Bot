"""
pipeline/orchestrator.py — Midnight Pipeline Orchestrator
==========================================================
The central brain of the system. Scheduled at 00:05 UTC every day via APScheduler.

Pipeline steps:
    0. Determine target date (yesterday UTC)
    1. Fetch messages from Supabase
    2. Wake up HuggingFace Space (handles cold starts up to 5 min)
    3. Send all messages to HF Space → get sentiment labels + HDBSCAN topic labels
    4. Write sentiment + topic labels back to Supabase messages table
    5. Run TF-IDF topic naming + metadata assembly (on Render, no heavy ML)
    6. Compute engagement scores + rank topics
    7. Compute per-user daily stats
    8. Build structured chart data JSON for the frontend
    9. Build summary JSON for Groq
    10. Generate Groq narrative report
    11. Write daily_report, topic_stats, user_daily_stats to Supabase

The 14-day message purge is NOT here — it runs via pg_cron at 00:05 UTC (skill-02).
"""

import logging
import traceback
import time
from datetime import date, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from db.supabase_client import (
    fetch_messages_for_date,
    bulk_update_message_topics,
    upsert_daily_report,
    insert_topic_stats,
    insert_user_daily_stats,
)
from pipeline.hf_client import wake_up_space, analyze_messages
from pipeline.topic_analyzer import extract_topic_metadata
from pipeline.groq_reporter import generate_narrative_report

logger = logging.getLogger(__name__)


def run_daily_pipeline():
    """
    Full midnight pipeline. Called by APScheduler at 00:05 UTC.
    Analyzes the previous UTC day's messages.
    """
    pipeline_start = time.time()

    # ── Step 0: Determine target date ─────────────────────────────────────────
    target_date = (date.today() - timedelta(days=1)).isoformat()
    logger.info(f"═══ Starting daily pipeline for {target_date} ═══")

    try:
        # ── Step 1: Fetch messages ─────────────────────────────────────────────
        logger.info("Step 1: Fetching messages from Supabase")
        messages = fetch_messages_for_date(target_date)

        if len(messages) < 10:
            logger.warning(
                f"Only {len(messages)} messages found for {target_date}. "
                f"Skipping pipeline (min threshold: 10)."
            )
            return

        logger.info(f"Fetched {len(messages)} messages")

        # ── Step 2: Wake up HF Space ───────────────────────────────────────────
        logger.info("Step 2: Waking up HuggingFace Space")
        if not wake_up_space():   # 10 retries × 30s = 5 min budget
            logger.error("HF Space failed to wake up after 10 attempts. Aborting pipeline.")
            return

        # ── Step 3: ML Analysis + Clustering (single HF Space request) ────────
        logger.info("Step 3: Sending messages to HF Space for sentiment + HDBSCAN")

        # Only send messages with text content — ML can't analyze empty messages.
        # We keep ALL messages for downstream volume/user stats.
        text_messages = [m for m in messages if m["content"].strip()]

        sentiments, topic_labels = analyze_messages(text_messages)
        # sentiments:   [{id, label, score}, ...] — same order as text_messages
        # topic_labels: [int, ...]                — HDBSCAN labels (-1 = noise)

        # O(1) lookup dict: message_id → {id, label, score}
        sent_by_id = {s["id"]: s for s in sentiments}

        n_topics = len(set(l for l in topic_labels if l != -1))
        logger.info(
            f"HF Space complete: {len(sentiments)} sentiments, {n_topics} topics"
        )

        # ── Step 4: Write sentiment + topic labels back to Supabase ───────────
        logger.info("Step 4: Updating message topics + sentiments in Supabase")
        updates = []
        for i, msg in enumerate(text_messages):
            s = sent_by_id.get(msg["message_id"], {"label": "neutral", "score": 0.5})
            updates.append({
                "message_id":      msg["message_id"],
                "topic_id":        int(topic_labels[i]),
                "sentiment_label": s["label"],
                "sentiment_score": s["score"],
            })
        bulk_update_message_topics(updates)
        logger.info("Message topics + sentiments saved")

        # ── Step 5: TF-IDF topic naming + metadata (Render, no heavy ML) ──────
        logger.info("Step 5: Extracting topic metadata (TF-IDF keywords)")
        topic_meta_list = extract_topic_metadata(
            messages=text_messages,
            topic_labels=topic_labels,
            sentiments=sent_by_id,
        )
        logger.info(f"Extracted metadata for {len(topic_meta_list)} topics")

        # ── Step 6: Engagement scores + ranking ───────────────────────────────
        logger.info("Step 6: Computing engagement scores")
        topic_meta_list = compute_engagement_scores(topic_meta_list)
        topic_meta_list.sort(key=lambda t: t["engagement_score"], reverse=True)
        for rank, topic in enumerate(topic_meta_list, start=1):
            topic["topic_rank"]  = rank
            topic["report_date"] = target_date

        # ── Step 7: Per-user daily stats ──────────────────────────────────────
        logger.info("Step 7: Computing user daily stats")
        user_stats = compute_user_stats(messages, topic_meta_list, sent_by_id)

        # ── Step 8: Chart data JSON for frontend ──────────────────────────────
        logger.info("Step 8: Building chart data JSON for frontend")
        chart_data = build_chart_data(messages, topic_meta_list, sent_by_id)
        logger.info("Chart data ready")

        # ── Step 9: Summary JSON for Groq ─────────────────────────────────────
        logger.info("Step 9: Building Groq summary JSON")
        all_sentiments = [
            sent_by_id.get(m["message_id"], {"label": "neutral", "score": 0.5})
            for m in text_messages
        ]
        n_sent = len(all_sentiments) or 1
        overall_pos = sum(1 for s in all_sentiments if s["label"] == "positive") / n_sent
        overall_neg = sum(1 for s in all_sentiments if s["label"] == "negative") / n_sent
        overall_neu = 1.0 - overall_pos - overall_neg

        summary_json = {
            "date":           target_date,
            "total_messages": len(messages),
            "text_messages":  len(text_messages),
            "total_users":    len(set(m["user_id"] for m in messages)),
            "total_topics":   len(topic_meta_list),
            "overall_sentiment": {
                "positive": round(overall_pos, 3),
                "neutral":  round(overall_neu, 3),
                "negative": round(overall_neg, 3),
            },
            "top_topics": [
                {
                    "rank":             t["topic_rank"],
                    "name":             t["topic_name"],
                    "initiator":        t["initiator_username"],
                    "message_count":    t["message_count"],
                    "unique_users":     t["unique_users"],
                    "duration_minutes": round(t["duration_minutes"], 1),
                    "engagement_score": round(t["engagement_score"], 3),
                    "sentiment":        t["sentiment_dist"],
                    "peak_hour":        t["peak_hour"],
                    "top_participants": [p["username"] for p in t["top_participants"][:5]],
                }
                for t in topic_meta_list[:15]
            ],
            "most_active_users": sorted(
                user_stats, key=lambda u: u["message_count"], reverse=True
            )[:10],
        }

        # ── Step 10: Groq narrative report ─────────────────────────────────────
        logger.info("Step 10: Generating Groq narrative report")
        narrative_md = generate_narrative_report(summary_json)
        logger.info("Groq report generated")

        # ── Step 11: Write to Supabase ─────────────────────────────────────────
        logger.info("Step 11: Writing results to Supabase")
        pipeline_duration = int(time.time() - pipeline_start)

        upsert_daily_report({
            "report_date":               target_date,
            "total_messages":            len(messages),
            "total_users":               summary_json["total_users"],
            "total_topics":              len(topic_meta_list),
            "overall_sentiment":         summary_json["overall_sentiment"],
            "narrative_md":              narrative_md,
            "chart_data":                chart_data,
            "summary_json":              summary_json,
            "pipeline_duration_seconds": pipeline_duration,
        })

        topic_rows = build_topic_rows(topic_meta_list)
        if topic_rows:
            insert_topic_stats(topic_rows)
        else:
            logger.warning("No topic rows to insert (all noise).")

        user_daily_rows = [{**u, "report_date": target_date} for u in user_stats]
        if user_daily_rows:
            insert_user_daily_stats(user_daily_rows)
        else:
            logger.warning("No user daily stats to insert.")

        # The 14-day purge is handled by pg_cron — no delete call needed here.

        logger.info(f"═══ Pipeline complete for {target_date} in {pipeline_duration}s ═══")

    except Exception as e:
        logger.error(f"Pipeline FAILED for {target_date}: {e}")
        logger.error(traceback.format_exc())


# ── Chart Data Builder ────────────────────────────────────────────────────────

def build_chart_data(
    messages: list[dict],
    topic_meta_list: list[dict],
    sent_by_id: dict,
) -> dict:
    """
    Builds structured JSON for the frontend's Recharts/Chart.js components.
    No images, no base64 — the frontend renders everything from this JSON.
    Response is ~15 KB vs the old matplotlib/Pillow approach (~2 MB).
    """
    from collections import Counter, defaultdict
    from datetime import datetime

    def parse_hour(ts: str) -> int:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).hour

    # 1. Hourly message volume (24 slots, 0-filled)
    hour_counts = Counter(parse_hour(m["created_at"]) for m in messages)
    hourly_volume = [{"hour": h, "count": hour_counts.get(h, 0)} for h in range(24)]

    # 2. Overall sentiment distribution
    all_labels = [
        sent_by_id.get(m["message_id"], {"label": "neutral"})["label"]
        for m in messages
    ]
    n = len(all_labels) or 1
    sentiment_overview = {
        "positive": round(all_labels.count("positive") / n, 3),
        "neutral":  round(all_labels.count("neutral")  / n, 3),
        "negative": round(all_labels.count("negative") / n, 3),
    }

    # 3. Sentiment breakdown per hour (for stacked area / bar chart)
    hour_buckets: dict[int, dict] = defaultdict(
        lambda: {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
    )
    for m in messages:
        h = parse_hour(m["created_at"])
        label = sent_by_id.get(m["message_id"], {"label": "neutral"})["label"]
        hour_buckets[h][label] += 1
        hour_buckets[h]["total"] += 1

    sentiment_by_hour = []
    for h in range(24):
        d = hour_buckets[h]
        total = d["total"] or 1
        sentiment_by_hour.append({
            "hour":     h,
            "positive": round(d["positive"] / total, 3),
            "neutral":  round(d["neutral"]  / total, 3),
            "negative": round(d["negative"] / total, 3),
            "count":    d["total"],
        })

    # 4. Topic engagement ranking (top 20 for bar chart)
    topic_engagement = [
        {
            "name":             t["topic_name"],
            "engagement_score": round(t["engagement_score"], 3),
            "message_count":    t["message_count"],
            "unique_users":     t["unique_users"],
            "duration_minutes": round(t["duration_minutes"], 1),
            "sentiment":        t["sentiment_dist"],
        }
        for t in topic_meta_list[:20]
    ]

    # 5. Top 15 users by message count (for horizontal bar chart)
    user_counts = Counter(m["username"] for m in messages)
    user_activity = [
        {"username": username, "message_count": count}
        for username, count in user_counts.most_common(15)
    ]

    return {
        "hourly_volume":      hourly_volume,       # [{hour, count}] × 24
        "sentiment_overview": sentiment_overview,  # {positive, neutral, negative}
        "sentiment_by_hour":  sentiment_by_hour,   # [{hour, pos, neu, neg, count}] × 24
        "topic_engagement":   topic_engagement,    # [{name, score, messages, …}] × ≤20
        "user_activity":      user_activity,       # [{username, message_count}] × ≤15
    }


# ── Engagement Score ──────────────────────────────────────────────────────────

def compute_engagement_scores(topics: list[dict]) -> list[dict]:
    """
    Min-max normalises 5 signals and computes a weighted engagement score.

    Weights (must sum to 1.0):
        unique_users      0.35  — breadth of participation
        message_count     0.25  — raw volume
        duration_minutes  0.20  — conversation longevity
        reply_density     0.15  — depth of back-and-forth
        positive_fraction 0.05  — sentiment boost (minor signal)
    """
    if not topics:
        return topics

    def minmax(values: list[float]) -> list[float]:
        mn, mx = min(values), max(values)
        if mx == mn:
            return [0.5] * len(values)
        return [(v - mn) / (mx - mn) for v in values]

    fields = ["unique_users", "message_count", "duration_minutes",
              "reply_density", "positive_fraction"]

    field_vals = {f: [t.get(f, 0) for t in topics] for f in fields}
    normalized  = {f: minmax(field_vals[f]) for f in fields}

    weights = {
        "unique_users":      0.35,
        "message_count":     0.25,
        "duration_minutes":  0.20,
        "reply_density":     0.15,
        "positive_fraction": 0.05,
    }

    for i, topic in enumerate(topics):
        topic["engagement_score"] = sum(
            normalized[f][i] * w for f, w in weights.items()
        )

    return topics


# ── User Stats ────────────────────────────────────────────────────────────────

def compute_user_stats(
    messages: list[dict],
    topic_meta_list: list[dict],
    sent_by_id: dict,
) -> list[dict]:
    from collections import defaultdict
    from datetime import datetime

    user_data: dict = defaultdict(lambda: {
        "message_count":    0,
        "topics_initiated": 0,
        "topics_joined":    set(),
        "sentiment_scores": [],
        "active_hours":     set(),
        "username":         "",
        "display_name":     "",
    })

    # Map message_id → topic_id for O(1) lookup
    msg_topic_map = {}
    for t in topic_meta_list:
        for msg_id in t.get("message_ids", []):
            msg_topic_map[msg_id] = t["topic_id"]

    for m in messages:
        uid  = m["user_id"]
        hour = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")).hour

        user_data[uid]["message_count"]   += 1
        user_data[uid]["username"]         = m["username"]
        user_data[uid]["display_name"]     = m.get("display_name", m["username"])
        user_data[uid]["active_hours"].add(hour)

        s = sent_by_id.get(m["message_id"])
        if s:
            user_data[uid]["sentiment_scores"].append(s["score"])

        topic_id = msg_topic_map.get(m["message_id"])
        if topic_id is not None:
            user_data[uid]["topics_joined"].add(topic_id)

    for t in topic_meta_list:
        uid = t["initiator_user_id"]
        if uid in user_data:
            user_data[uid]["topics_initiated"] += 1

    rows = []
    for uid, d in user_data.items():
        avg_sent = (
            sum(d["sentiment_scores"]) / len(d["sentiment_scores"])
            if d["sentiment_scores"] else None
        )
        rows.append({
            "user_id":          uid,
            "username":         d["username"],
            "display_name":     d["display_name"],
            "message_count":    d["message_count"],
            "topics_initiated": d["topics_initiated"],
            "topics_joined":    len(d["topics_joined"]),
            "avg_sentiment":    round(avg_sent, 4) if avg_sent else None,
            "active_hours":     sorted(d["active_hours"]),
        })

    return rows


# ── Build topic rows for Supabase ─────────────────────────────────────────────

def build_topic_rows(topic_meta_list: list[dict]) -> list[dict]:
    return [
        {
            "report_date":        t["report_date"],
            "topic_rank":         t["topic_rank"],
            "topic_id":           t["topic_id"],
            "topic_name":         t["topic_name"],
            "topic_keywords":     t["topic_keywords"],
            "initiator_user_id":  t["initiator_user_id"],
            "initiator_username": t["initiator_username"],
            "message_count":      t["message_count"],
            "unique_users":       t["unique_users"],
            "duration_minutes":   t["duration_minutes"],
            "peak_hour":          t["peak_hour"],
            "engagement_score":   t["engagement_score"],
            "sentiment_dist":     t["sentiment_dist"],
            "top_participants":   t["top_participants"][:10],
            "first_message_at":   t["first_message_at"],
            "last_message_at":    t["last_message_at"],
        }
        for t in topic_meta_list
    ]


# ── APScheduler Setup ─────────────────────────────────────────────────────────

def start_scheduler():
    """
    Creates and starts the APScheduler background scheduler.
    Called once at FastAPI startup (inside lifespan).

    misfire_grace_time=3600: If Render restarts at midnight and misses the
    00:05 window, APScheduler will still fire the job within the next hour.
    """
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        run_daily_pipeline,
        trigger=CronTrigger(hour=0, minute=5),
        id="daily_pipeline",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.start()
    logger.info("APScheduler started — daily pipeline fires at 00:05 UTC")
    return scheduler
