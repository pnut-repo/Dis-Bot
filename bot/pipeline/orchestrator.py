"""
pipeline/orchestrator.py — Midnight Pipeline Orchestrator (Groq-Only v3)
==========================================================================
The central brain of the system. Scheduled at 00:05 UTC every day via APScheduler.

Architecture: Two-stage Groq LLM pipeline (no HuggingFace Space).

Pipeline steps:
    0. Determine target date (yesterday UTC)
    1. Fetch ALL messages from Supabase (paginated, full day 00:00–23:59)
    2. Groq API 1 (llama-4-scout): Batch topic analysis + sentiment
       - Processes ~1000 messages per batch
       - Maintains topic consistency across batches
       - 65s delay between batches for rate limit
    3. Write sentiment + topic labels back to Supabase messages table
    4. Build topic metadata from LLM results
    5. Compute engagement scores + rank topics
    6. Compute per-user daily stats
    7. Build chart data JSON for frontend (including per-user analytics)
    8. Build summary JSON for Groq API 2
    9. Groq API 2 (llama-3.3-70b): Generate narrative report
   10. Write daily_report, topic_stats, user_daily_stats, daily_analytics to Supabase
"""

import logging
import traceback
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from db.supabase_client import (
    fetch_messages_for_date,
    bulk_update_message_topics,
    upsert_daily_report,
    insert_topic_stats,
    insert_user_daily_stats,
    upsert_daily_analytics,
)
from pipeline.groq_topic_engine import analyze_messages
from pipeline.groq_reporter import generate_narrative_report

logger = logging.getLogger(__name__)


def run_daily_pipeline():
    """
    Full midnight pipeline. Called by APScheduler at 00:05 UTC.
    Analyzes the previous UTC day's messages using two Groq LLM APIs.
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

        # ── Step 2: Groq API 1 — Topic & Sentiment Analysis ──────────────────
        logger.info("Step 2: Groq API 1 — Batch topic & sentiment analysis")
        groq_result = analyze_messages(messages)

        msg_results = groq_result.get("messages", [])
        llm_topics = groq_result.get("topics", [])

        # O(1) lookup: message_id → {topic_id, sentiment, score}
        result_by_id = {m["id"]: m for m in msg_results}

        n_topics = groq_result.get("n_topics", 0)
        logger.info(
            f"Groq API 1 complete: {len(msg_results)} messages analyzed, "
            f"{n_topics} topics, {groq_result.get('uncategorized_count', 0)} uncategorized, "
            f"{groq_result.get('processing_time_seconds', 0)}s"
        )

        # ── Step 3: Write sentiment + topic labels back to Supabase ───────────
        logger.info("Step 3: Updating message topics + sentiments in Supabase")
        updates = []
        for msg in messages:
            r = result_by_id.get(msg["message_id"], {
                "topic_id": -1, "sentiment": "neutral", "score": 0.5,
            })
            updates.append({
                **msg,  # original Supabase row
                "topic_id":        int(r.get("topic_id", -1)),
                "sentiment_label": r.get("sentiment", "neutral"),
                "sentiment_score": float(r.get("score", 0.5)),
            })
        bulk_update_message_topics(updates)
        logger.info("Message topics + sentiments saved")

        # ── Step 4: Build topic metadata from LLM results ─────────────────────
        logger.info("Step 4: Building topic metadata")
        topic_meta_list = _build_topic_metadata(
            messages=messages,
            result_by_id=result_by_id,
            llm_topics=llm_topics,
        )
        logger.info(f"Built metadata for {len(topic_meta_list)} topics")

        # ── Step 5: Engagement scores + ranking ───────────────────────────────
        logger.info("Step 5: Computing engagement scores")
        topic_meta_list = compute_engagement_scores(topic_meta_list)
        topic_meta_list.sort(key=lambda t: t["engagement_score"], reverse=True)
        for rank, topic in enumerate(topic_meta_list, start=1):
            topic["topic_rank"] = rank
            topic["report_date"] = target_date

        # ── Step 6: Per-user daily stats ──────────────────────────────────────
        logger.info("Step 6: Computing user daily stats")
        user_stats = compute_user_stats(messages, topic_meta_list, result_by_id)

        # ── Step 7: Chart data JSON for frontend ──────────────────────────────
        logger.info("Step 7: Building chart data JSON for frontend")
        chart_data = build_chart_data(messages, topic_meta_list, result_by_id)
        logger.info("Chart data ready")

        # ── Step 8: Build summary JSON for Groq API 2 ────────────────────────
        logger.info("Step 8: Building Groq summary JSON")
        batch_reports = groq_result.get("batch_reports", [])
        summary_json = _build_summary_json(
            target_date, messages, topic_meta_list, user_stats, result_by_id, batch_reports
        )

        # ── Step 9: Groq API 2 — Narrative Report ─────────────────────────────
        logger.info("Step 9: Generating narrative report (llama-3.3-70b)")
        narrative_md = generate_narrative_report(summary_json)
        logger.info("Groq report generated")

        # ── Step 10: Write to Supabase ─────────────────────────────────────────
        logger.info("Step 10: Writing results to Supabase")
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

        # Write expanded analytics data (30-day retention)
        upsert_daily_analytics({
            "report_date": target_date,
            "hourly_data": chart_data.get("hourly_volume", []),
            "sentiment_hourly": chart_data.get("sentiment_by_hour", []),
            "user_analytics": chart_data.get("user_detail", {}),
            "topic_analytics": chart_data.get("topic_engagement", []),
            "summary_stats": {
                "total_messages": len(messages),
                "total_users": summary_json["total_users"],
                "total_topics": len(topic_meta_list),
                "overall_sentiment": summary_json["overall_sentiment"],
            },
        })

        logger.info(f"═══ Pipeline complete for {target_date} in {pipeline_duration}s ═══")

    except Exception as e:
        logger.error(f"Pipeline FAILED for {target_date}: {e}")
        logger.error(traceback.format_exc())


# ── Topic Metadata Builder ────────────────────────────────────────────────────

def _build_topic_metadata(
    messages: list[dict],
    result_by_id: dict,
    llm_topics: list[dict],
) -> list[dict]:
    """
    Build topic metadata by combining LLM topic analysis with raw message data.

    The LLM provides: topic names, keywords, insights
    This function adds: message counts, participants, timing, sentiment distribution
    """
    # Group messages by topic_id
    topic_msgs: dict[int, list[dict]] = defaultdict(list)
    for msg in messages:
        r = result_by_id.get(msg["message_id"], {})
        tid = r.get("topic_id", -1)
        if tid >= 0:
            topic_msgs[tid].append(msg)

    # Build LLM topic lookup
    llm_topic_map = {t["topic_id"]: t for t in llm_topics if t.get("topic_id", -1) >= 0}

    topics = []
    for tid, msgs in topic_msgs.items():
        if not msgs:
            continue

        llm_info = llm_topic_map.get(tid, {})
        msgs.sort(key=lambda m: m["created_at"])

        # Participants
        user_counter = Counter(m["username"] for m in msgs)
        top_participants = [
            {"username": u, "message_count": c}
            for u, c in user_counter.most_common(10)
        ]

        # Timing
        first_ts = msgs[0]["created_at"]
        last_ts = msgs[-1]["created_at"]
        try:
            first_dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
            last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            duration = (last_dt - first_dt).total_seconds() / 60
        except (ValueError, TypeError):
            duration = 0

        # Peak hour
        hours = []
        for m in msgs:
            try:
                hours.append(datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")).hour)
            except (ValueError, TypeError):
                pass
        peak_hour = Counter(hours).most_common(1)[0][0] if hours else 0

        # Sentiment distribution
        sentiments = [
            result_by_id.get(m["message_id"], {}).get("sentiment", "neutral")
            for m in msgs
        ]
        sent_counter = Counter(sentiments)
        n_sent = len(sentiments) or 1
        sentiment_dist = {
            s: round(sent_counter.get(s, 0) / n_sent, 3)
            for s in ["excited", "happy", "curious", "neutral", "frustrated", "angry", "sad", "confused"]
        }

        # Tension score
        tension = sentiment_dist["angry"] + sentiment_dist["frustrated"] + 0.5 * sentiment_dist["sad"] + 0.3 * sentiment_dist["neutral"]

        # Reply density
        reply_count = sum(1 for m in msgs if m.get("is_reply") or m.get("reply_to_id"))
        reply_density = reply_count / len(msgs)

        # Initiator (first message in topic)
        initiator = msgs[0]

        topics.append({
            "topic_id":            tid,
            "topic_name":          llm_info.get("name", f"Topic {tid}"),
            "topic_keywords":      llm_info.get("keywords", []),
            "groq_insight":        llm_info.get("insight", ""),
            "message_count":       len(msgs),
            "unique_users":        len(user_counter),
            "top_participants":    top_participants,
            "initiator_user_id":   initiator["user_id"],
            "initiator_username":  initiator["username"],
            "first_message_at":    first_ts,
            "last_message_at":     last_ts,
            "duration_minutes":    duration,
            "peak_hour":           peak_hour,
            "sentiment_dist":      sentiment_dist,
            "tension_score":       round(tension, 4),
            "needs_moderation":    tension > 0.40,
            "reply_density":       round(reply_density, 3),
            "positive_fraction":   round(sentiment_dist["excited"] + sentiment_dist["happy"], 3),
            "engagement_score":    0,  # Computed in Step 5
            "message_ids":         [m["message_id"] for m in msgs],
        })

    return topics


# ── Summary JSON Builder ─────────────────────────────────────────────────────

def _build_summary_json(
    target_date: str,
    messages: list[dict],
    topic_meta_list: list[dict],
    user_stats: list[dict],
    result_by_id: dict,
    batch_reports: list[dict] = None,
) -> dict:
    """Build the structured summary JSON sent to Groq API 2 for narrative generation."""
    # Overall sentiment
    all_sentiments = [
        result_by_id.get(m["message_id"], {}).get("sentiment", "neutral")
        for m in messages
    ]
    n_sent = len(all_sentiments) or 1
    sent_counter = Counter(all_sentiments)
    overall_sentiment = {
        s: round(sent_counter.get(s, 0) / n_sent, 3)
        for s in ["excited", "happy", "curious", "neutral", "frustrated", "angry", "sad", "confused"]
    }

    return {
        "date":           target_date,
        "total_messages": len(messages),
        "total_users":    len(set(m["user_id"] for m in messages)),
        "total_topics":   len(topic_meta_list),
        "overall_sentiment": overall_sentiment,
        "batch_reports": (batch_reports or [])[-10:],
        "top_topics": [
            {
                "rank":             t["topic_rank"],
                "name":             t["topic_name"],
                "keywords":         t["topic_keywords"][:5],
                "initiator":        t["initiator_username"],
                "message_count":    t["message_count"],
                "unique_users":     t["unique_users"],
                "duration_minutes": round(t["duration_minutes"], 1),
                "engagement_score": round(t["engagement_score"], 3),
                "sentiment":        t["sentiment_dist"],
                "tension_score":    t.get("tension_score", 0),
                "needs_moderation": t.get("needs_moderation", False),
                "peak_hour":        t["peak_hour"],
                "top_participants": [p["username"] for p in t["top_participants"][:5]],
                "groq_insight":     t.get("groq_insight", ""),
            }
            for t in topic_meta_list[:10]
        ],
        "most_active_users": sorted(
            user_stats, key=lambda u: u["message_count"], reverse=True
        )[:10],
    }


# ── Chart Data Builder ────────────────────────────────────────────────────────

def build_chart_data(
    messages: list[dict],
    topic_meta_list: list[dict],
    result_by_id: dict,
) -> dict:
    """
    Builds structured JSON for the frontend's Recharts components.
    Includes per-user detailed analytics for the user dropdown feature.
    """
    def parse_hour(ts: str) -> int:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).hour

    # 1. Hourly message volume (24 slots, 0-filled)
    hour_counts = Counter(parse_hour(m["created_at"]) for m in messages)
    hourly_volume = [{"hour": h, "count": hour_counts.get(h, 0)} for h in range(24)]

    # 2. Overall sentiment distribution
    all_labels = [
        result_by_id.get(m["message_id"], {}).get("sentiment", "neutral")
        for m in messages
    ]
    n = len(all_labels) or 1
    label_counts = Counter(all_labels)
    sentiment_overview = {
        s: round(label_counts.get(s, 0) / n, 3)
        for s in ["excited", "happy", "curious", "neutral", "frustrated", "angry", "sad", "confused"]
    }

    # 3. Sentiment breakdown per hour
    hour_buckets: dict[int, dict] = defaultdict(
        lambda: {"excited": 0, "happy": 0, "curious": 0, "neutral": 0, "frustrated": 0, "angry": 0, "sad": 0, "confused": 0, "total": 0}
    )
    for m in messages:
        h = parse_hour(m["created_at"])
        label = result_by_id.get(m["message_id"], {}).get("sentiment", "neutral")
        if label not in hour_buckets[h]:
            label = "neutral"
        hour_buckets[h][label] += 1
        hour_buckets[h]["total"] += 1

    sentiment_by_hour = []
    for h in range(24):
        d = hour_buckets[h]
        total = d["total"] or 1
        hour_data = {"hour": h, "count": d["total"]}
        for s in ["excited", "happy", "curious", "neutral", "frustrated", "angry", "sad", "confused"]:
            hour_data[s] = round(d[s] / total, 3)
        sentiment_by_hour.append(hour_data)

    # 4. Topic engagement ranking (top 20)
    topic_engagement = [
        {
            "name":             t["topic_name"],
            "keywords":         t["topic_keywords"][:5],
            "engagement_score": round(t["engagement_score"], 3),
            "message_count":    t["message_count"],
            "unique_users":     t["unique_users"],
            "duration_minutes": round(t["duration_minutes"], 1),
            "sentiment":        t["sentiment_dist"],
            "tension_score":    t.get("tension_score", 0),
            "needs_moderation": t.get("needs_moderation", False),
        }
        for t in topic_meta_list[:20]
    ]

    # 5. Top 15 users by message count
    user_counts = Counter(m["username"] for m in messages)
    user_activity = [
        {"username": username, "message_count": count}
        for username, count in user_counts.most_common(15)
    ]

    # 6. Per-user detailed analytics (for user dropdown)
    user_detail = _build_user_detail(messages, result_by_id, topic_meta_list)

    return {
        "hourly_volume":      hourly_volume,
        "sentiment_overview": sentiment_overview,
        "sentiment_by_hour":  sentiment_by_hour,
        "topic_engagement":   topic_engagement,
        "user_activity":      user_activity,
        "user_detail":        user_detail,
    }


def _build_user_detail(
    messages: list[dict],
    result_by_id: dict,
    topic_meta_list: list[dict],
) -> dict:
    """
    Build per-user detailed analytics for the user dropdown feature.

    Returns: {
        "username1": {
            "hourly_activity": [{hour: 0, count: 5}, ...],
            "sentiment": {positive: 0.3, neutral: 0.5, negative: 0.2},
            "topics": ["Gaming Discussion", "Bug Reports"],
            "message_count": 45,
            "active_hours": [10, 11, 14, 15, 20, 21],
        },
        ...
    }
    """
    # Build topic_id → topic_name lookup
    topic_names = {t["topic_id"]: t["topic_name"] for t in topic_meta_list}

    user_data: dict = defaultdict(lambda: {
        "hourly": Counter(),
        "sentiments": [],
        "topic_ids": set(),
        "message_count": 0,
    })

    for m in messages:
        username = m["username"]
        try:
            hour = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")).hour
        except (ValueError, TypeError):
            hour = 0

        r = result_by_id.get(m["message_id"], {})
        sentiment = r.get("sentiment", "neutral")
        topic_id = r.get("topic_id", -1)

        user_data[username]["hourly"][hour] += 1
        user_data[username]["sentiments"].append(sentiment)
        user_data[username]["message_count"] += 1
        if topic_id >= 0:
            user_data[username]["topic_ids"].add(topic_id)

    result = {}
    for username, data in user_data.items():
        sents = data["sentiments"]
        n = len(sents) or 1
        result[username] = {
            "hourly_activity": [
                {"hour": h, "count": data["hourly"].get(h, 0)}
                for h in range(24)
            ],
            "sentiment": {
                s: round(sents.count(s) / n, 3)
                for s in ["excited", "happy", "curious", "neutral", "frustrated", "angry", "sad", "confused"]
            },
            "topics": [
                topic_names.get(tid, f"Topic {tid}")
                for tid in sorted(data["topic_ids"])
            ],
            "message_count": data["message_count"],
            "active_hours": sorted(data["hourly"].keys()),
        }

    return result


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
    result_by_id: dict,
) -> list[dict]:
    """Compute per-user daily stats for Supabase user_daily_stats table."""
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
        uid = m["user_id"]
        try:
            hour = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")).hour
        except (ValueError, TypeError):
            hour = 0

        user_data[uid]["message_count"]   += 1
        user_data[uid]["username"]         = m["username"]
        user_data[uid]["display_name"]     = m.get("display_name", m["username"])
        user_data[uid]["active_hours"].add(hour)

        r = result_by_id.get(m["message_id"])
        if r:
            try:
                score = float(r.get("score", 0.5))
            except (ValueError, TypeError):
                score = 0.5
            user_data[uid]["sentiment_scores"].append(score)

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
            "tension_score":      t.get("tension_score", 0),
            "needs_moderation":   t.get("needs_moderation", False),
            "top_participants":   t["top_participants"][:10],
            "first_message_at":   t["first_message_at"],
            "last_message_at":    t["last_message_at"],
            "groq_insight":       t.get("groq_insight", ""),
        }
        for t in topic_meta_list
    ]


# ── APScheduler Setup ─────────────────────────────────────────────────────────

def start_scheduler():
    """
    Creates and starts the APScheduler background scheduler.
    Called once at FastAPI startup (inside lifespan).
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
