"""
pipeline/groq_reporter.py — Groq Narrative Report Generator
=============================================================
Two responsibilities:

1. **Per-topic insights** (NEW): For each topic cluster, sends actual
   messages + reply chains + keywords to Groq. Returns a 2-3 sentence
   contextual insight per topic. Batches 3 topics per Groq call to stay
   within the free-tier 30 RPM / 30k TPM limits.

2. **Daily narrative report**: Takes the structured summary JSON (now
   enriched with per-topic insights) and generates a Markdown daily digest.

Model: meta-llama/llama-4-scout-17b-16e-instruct
Usage: ~6 calls/day (5 topic batches + 1 narrative) ≪ 14,400 RPD free tier.
"""

import os
import json
import time
import logging
from collections import defaultdict

from groq import Groq

logger = logging.getLogger(__name__)

# ── Groq client (singleton) ───────────────────────────────────────────────────

_client: Groq | None = None

MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS  = 3000   # Output cap — report should be ~2,000 tokens
TEMPERATURE = 0.7    # Slightly creative prose, grounded by structured data


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set")
        _client = Groq(api_key=api_key)
    return _client


# ══════════════════════════════════════════════════════════════════════════════
# 1. Per-Topic Insight Generation
# ══════════════════════════════════════════════════════════════════════════════

TOPIC_SYSTEM_PROMPT = """You are a Discord community analyst. For each topic below, write a concise 2-3 sentence insight.

**Rules:**
1. Describe what the conversation was actually about (not just listing keywords).
2. Note the emotional tone — agreement, debate, excitement, frustration, etc.
3. Mention who drove the discussion if obvious from the messages.
4. Be factual — only reference content from the provided messages.
5. Keep each topic insight to exactly 2-3 sentences.
6. Return your response as a JSON object mapping topic_id (as string) to the insight text.

**Response format (strict JSON, no markdown fences):**
{"0": "insight for topic 0...", "3": "insight for topic 3..."}"""


def _build_topic_block(
    topic_meta: dict,
    topic_messages: list[dict],
    msg_by_id: dict,
    max_messages: int = 30,
) -> str:
    """
    Build a structured text block for one topic, suitable for the Groq prompt.

    Includes:
      - Topic keywords and sentiment distribution
      - Up to max_messages in chronological order
      - Reply chain notation: "bob (→alice):" shows bob replying to alice
    """
    tid = topic_meta["topic_id"]
    kw  = ", ".join(topic_meta.get("topic_keywords", [])[:6])
    sd  = topic_meta.get("sentiment_dist", {})
    pos = round(sd.get("positive", 0) * 100)
    neu = round(sd.get("neutral", 0) * 100)
    neg = round(sd.get("negative", 0) * 100)
    mc  = topic_meta.get("message_count", len(topic_messages))

    lines = [
        f'Topic {tid}: "{topic_meta.get("topic_name", f"topic_{tid}")}"',
        f"Keywords: {kw}",
        f"Sentiment: {pos}% positive, {neu}% neutral, {neg}% negative",
        f"Messages ({min(max_messages, len(topic_messages))} of {mc}):",
    ]

    # Sort chronologically and take up to max_messages
    sorted_msgs = sorted(topic_messages, key=lambda m: m["created_at"])[:max_messages]

    for m in sorted_msgs:
        ts = m["created_at"]
        # Extract HH:MM from ISO timestamp
        try:
            hhmm = ts[11:16]
        except (IndexError, TypeError):
            hhmm = "??:??"

        username = m.get("username", "unknown")
        content  = m.get("content", "").strip()
        if not content:
            continue

        # Truncate very long messages to save tokens
        if len(content) > 200:
            content = content[:197] + "..."

        # Show reply chain: "bob (→alice):" if this message is a reply
        reply_to_id = m.get("reply_to_id")
        if reply_to_id and reply_to_id in msg_by_id:
            parent = msg_by_id[reply_to_id]
            parent_name = parent.get("username", "unknown")
            lines.append(f"[{hhmm}] {username} (→{parent_name}): {content}")
        else:
            lines.append(f"[{hhmm}] {username}: {content}")

    return "\n".join(lines)


def generate_topic_insights(
    topic_meta_list: list[dict],
    messages: list[dict],
    topic_labels: list[int],
    max_topics: int = 15,
    topics_per_batch: int = 3,
    delay_between_calls: float = 3.0,
) -> dict[int, str]:
    """
    Send actual messages + reply chains to Groq for per-topic contextual insights.

    Groups messages by topic, builds structured prompt blocks with reply-chain
    notation (→parent), and batches 3 topics per Groq call to stay within
    the free-tier rate limits (30 RPM, 30k TPM).

    Args:
        topic_meta_list:  Ranked topic metadata from extract_topic_metadata().
        messages:         All text messages for the day (same order as topic_labels).
        topic_labels:     HDBSCAN cluster labels per message (-1 = noise).
        max_topics:       Process at most this many topics (by engagement rank).
        topics_per_batch: Topics per Groq API call (3-4 to stay within TPM).
        delay_between_calls: Seconds to wait between Groq calls (rate limiting).

    Returns:
        Dict mapping topic_id → insight text string.
        On failure: returns empty dict — never raises.
    """
    if not topic_meta_list or not messages:
        return {}

    # ── Group messages by topic ──────────────────────────────────────────────
    topic_messages: dict[int, list[dict]] = defaultdict(list)
    for msg, label in zip(messages, topic_labels):
        if label != -1:
            topic_messages[label].append(msg)

    # Build O(1) lookup for reply chain resolution
    msg_by_id = {m["message_id"]: m for m in messages}

    # Only process top N topics by engagement rank
    topics_to_process = topic_meta_list[:max_topics]
    logger.info(
        f"Generating Groq insights for {len(topics_to_process)} topics "
        f"({topics_per_batch} per batch, {delay_between_calls}s delay)"
    )

    # ── Batch topics into Groq calls ─────────────────────────────────────────
    all_insights: dict[int, str] = {}

    for batch_start in range(0, len(topics_to_process), topics_per_batch):
        batch = topics_to_process[batch_start : batch_start + topics_per_batch]

        # Build prompt with message blocks for each topic in this batch
        blocks = []
        for tmeta in batch:
            tid = tmeta["topic_id"]
            msgs = topic_messages.get(tid, [])
            if msgs:
                blocks.append(_build_topic_block(tmeta, msgs, msg_by_id))

        if not blocks:
            continue

        user_prompt = (
            "Analyze the following Discord conversation topics and provide "
            "a 2-3 sentence contextual insight for each.\n\n"
            + "\n\n---\n\n".join(blocks)
        )

        try:
            client = get_groq_client()
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": TOPIC_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=1500,     # ~500 tokens per topic × 3 topics
                temperature=0.5,     # More factual for per-topic analysis
            )

            raw = completion.choices[0].message.content.strip()

            # Parse JSON response — Groq returns {"0": "...", "3": "..."}
            # Strip markdown fences if model wraps them
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            parsed = json.loads(raw)
            for tid_str, insight in parsed.items():
                try:
                    tid_int = int(tid_str)
                    all_insights[tid_int] = insight.strip()
                except (ValueError, AttributeError):
                    continue

            batch_ids = [t["topic_id"] for t in batch]
            logger.info(f"Groq insights received for topics: {batch_ids}")

        except json.JSONDecodeError as e:
            batch_ids = [t["topic_id"] for t in batch]
            logger.warning(
                f"Groq returned non-JSON for topics {batch_ids}: {e}. "
                f"Raw response: {raw[:200]}"
            )
        except Exception as e:
            batch_ids = [t["topic_id"] for t in batch]
            logger.error(f"Groq insight call failed for topics {batch_ids}: {e}")

        # Rate limit: wait between calls to stay within 30k TPM
        if batch_start + topics_per_batch < len(topics_to_process):
            time.sleep(delay_between_calls)

    logger.info(
        f"Topic insights complete: {len(all_insights)}/{len(topics_to_process)} topics"
    )
    return all_insights


# ══════════════════════════════════════════════════════════════════════════════
# 2. Daily Narrative Report
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a Discord community analyst. You write a concise, insightful daily digest for server moderators.

**Rules:**
1. Write in Markdown format with clear headings.
2. Start with a one-paragraph executive summary (total messages, users, overall mood).
3. Highlight the top 3–5 topics by engagement — for each, mention the topic name, who started it, how many people joined, and the sentiment.
4. If per-topic insights are provided, weave them into your topic summaries for richer context.
5. Note any interesting patterns: quiet hours, sentiment spikes, unusually active users.
6. End with a brief "Community Health" assessment (1–2 sentences).
7. Keep the total report under 500 words.
8. Be factual — only reference data from the provided JSON. Do not invent usernames, topics, or numbers.
9. Use a friendly but professional tone."""


def build_user_prompt(summary_json: dict) -> str:
    """
    Converts the orchestrator's summary dict into the Groq user prompt.

    The summary JSON contains:
      - date, total_messages, text_messages, total_users, total_topics
      - overall_sentiment {positive, neutral, negative}
      - top_topics [{rank, name, initiator, message_count, unique_users,
                     groq_insight (NEW), ...}]
      - most_active_users [{username, message_count}]
    """
    return (
        f"Here is the structured analysis for **{summary_json['date']}**.\n\n"
        f"Write a daily digest report based on this data:\n\n"
        f"```json\n{json.dumps(summary_json, indent=2)}\n```"
    )


def generate_narrative_report(summary_json: dict) -> str:
    """
    Sends the summary JSON to Groq Llama 4 Scout and returns a Markdown report.

    Args:
        summary_json: Structured dict from orchestrator Step 9.

    Returns:
        Markdown string (~2,000 tokens / ~500 words).
        On any failure: returns a fallback stub report — never raises.
    """
    try:
        client      = get_groq_client()
        user_prompt = build_user_prompt(summary_json)

        logger.info(f"Sending summary to Groq ({MODEL})")

        completion = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        narrative = completion.choices[0].message.content
        logger.info(f"Groq report generated: {len(narrative)} chars")
        return narrative

    except Exception as e:
        logger.error(f"Groq report generation failed: {e}")
        # Fallback: write a stub report so the pipeline never crashes.
        return (
            f"# Daily Report — {summary_json.get('date', 'Unknown')}\n\n"
            f"*Report generation failed. Raw stats: "
            f"{summary_json.get('total_messages', '?')} messages from "
            f"{summary_json.get('total_users', '?')} users.*"
        )
