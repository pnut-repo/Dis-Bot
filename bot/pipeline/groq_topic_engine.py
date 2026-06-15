"""
pipeline/groq_topic_engine.py — Groq-based Topic & Sentiment Engine
=====================================================================
Replaces the HuggingFace Space ML pipeline entirely. Uses Groq's
llama-4-scout-17b-16e-instruct to analyze Discord messages in batches.

The LLM reads actual conversations and outputs:
  - Topic groupings (which messages belong to which topic)
  - Topic names, keywords, and contextual insights
  - Per-message sentiment (positive / neutral / negative + score)

Token Budget (free tier):
  - Model: llama-4-scout-17b-16e-instruct
  - Limit: 30k TPM, 30 RPM, 14,400 RPD
  - Budget per batch: ~25k tokens (5k safety margin)
  - ~1,000 messages per batch × ~20 tokens/msg = 20k + 2k prompt + 3k output = 25k
  - Wait 65s between batches to never touch 30k ceiling.

Error handling:
  - On 429 (rate limit): exponential backoff up to 3 retries
  - On JSON parse failure: retry once with stricter prompt
  - On total failure: return neutral sentiment + no topics (pipeline continues)
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
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Fixed batch size: 100 messages per Groq call.
# ~9k input tokens + ~4k output tokens = ~13k total, well under 30k TPM.
MESSAGES_PER_BATCH = 100
WAIT_BETWEEN_BATCHES = 65  # seconds — ensures we stay under 30k TPM


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY_SCOUT")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY_SCOUT environment variable not set")
        _client = Groq(api_key=api_key)
    return _client


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Discord message analyzer. You receive batches of messages from a single day and must:

1. **Group messages into conversation topics.** Messages about the same subject get the same topic_id.
2. **Assign sentiment** to each message: "positive", "neutral", or "negative" with a confidence score 0.0-1.0.
3. **Name each topic** with a short descriptive title (3-6 words).
4. **Extract keywords** for each topic (top 5 most relevant words).
5. **Write an insight** for each topic: a 2-3 sentence summary of what was discussed, the emotional tone, and who drove the discussion.

**Critical rules:**
- This is BATCH processing. You may receive known topics from previous batches. Assign messages to existing topics if they match, or create new topics with NEW topic_ids (use the next available integer).
- topic_id must be a non-negative integer starting from 0.
- Messages that don't fit any topic should get topic_id: -1 (uncategorized).
- Short messages like "lol", "ok", "yeah" should be assigned to the topic of the message they reply to (check reply_to field), or -1 if no reply context.
- Be consistent: if batch 1 had "Gaming Discussion" as topic 0, batch 2 should reuse topic 0 for similar gaming messages.

**Response format (STRICT JSON, no markdown fences, no extra text):**
{
  "messages": [
    {"id": "msg_id_here", "topic_id": 0, "sentiment": "positive", "score": 0.85},
    {"id": "msg_id_here", "topic_id": -1, "sentiment": "neutral", "score": 0.5}
  ],
  "topics": [
    {
      "topic_id": 0,
      "name": "Game Balance Debate",
      "keywords": ["nerf", "pvp", "patch", "meta", "broken"],
      "insight": "Players debated the latest balance patch. The tone was mostly frustrated, with several users calling the changes unfair. User_xyz led the discussion with 12 messages."
    }
  ]
}"""


def _build_batch_prompt(
    batch_messages: list[dict],
    known_topics: list[dict],
    batch_number: int,
    total_batches: int,
) -> str:
    """
    Build the user prompt for a single batch.

    Each message is formatted as:
      {"id": "...", "user": "username", "time": "HH:MM", "text": "...", "reply_to": "parent_username or null"}

    Known topics from previous batches are included so the LLM can
    reuse existing topic_ids for consistency.
    """
    # Build compact message list (save tokens)
    compact_msgs = []
    # Pre-build username lookup for reply_to resolution
    id_to_username = {m["message_id"]: m["username"] for m in batch_messages}

    for m in batch_messages:
        text = m.get("content", "").strip()
        if not text:
            text = "[empty]"
        # Truncate very long messages to save tokens
        if len(text) > 300:
            text = text[:297] + "..."

        ts = m.get("created_at", "")
        try:
            hhmm = ts[11:16]
        except (IndexError, TypeError):
            hhmm = "00:00"

        entry = {
            "id": m["message_id"],
            "user": m["username"],
            "time": hhmm,
            "text": text,
        }

        # Add reply context if available
        reply_to_id = m.get("reply_to_id")
        if reply_to_id:
            parent_name = id_to_username.get(reply_to_id)
            if parent_name:
                entry["reply_to"] = parent_name

        compact_msgs.append(entry)

    parts = [f"Batch {batch_number}/{total_batches}. Analyze these {len(compact_msgs)} Discord messages:"]

    if known_topics:
        parts.append(
            "\n**KNOWN TOPICS from previous batches (reuse these topic_ids if messages match):**\n"
            + json.dumps(known_topics, indent=None)
        )

    parts.append("\n**Messages:**\n" + json.dumps(compact_msgs, indent=None))

    return "\n".join(parts)


def _parse_response(raw: str) -> dict:
    """
    Parse the LLM's JSON response, handling common formatting issues.
    Returns {"messages": [...], "topics": [...]} or raises ValueError.
    """
    # Strip markdown fences if the model wraps them
    text = raw.strip()
    if text.startswith("```"):
        # Remove ```json or ``` prefix
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    parsed = json.loads(text)

    if "messages" not in parsed or "topics" not in parsed:
        raise ValueError(f"Missing 'messages' or 'topics' key in response")

    return parsed


def _call_groq_with_retry(
    system: str,
    user_prompt: str,
    max_retries: int = 3,
) -> str:
    """
    Call Groq with exponential backoff on 429 / transient errors.
    Returns raw response text.
    """
    client = _get_client()
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=8192,
                temperature=0.3,  # Low temp for consistent structured output
            )
            return completion.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate" in error_str.lower()

            if attempt == max_retries:
                raise

            wait = (2 ** attempt) * 30 if is_rate_limit else 5
            logger.warning(
                f"Groq call failed (attempt {attempt}/{max_retries}): {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError("Unreachable")


# ── Main Entry Point ─────────────────────────────────────────────────────────

def analyze_messages(messages: list[dict]) -> dict:
    """
    Analyze a full day's messages using Groq LLM in batches.

    This replaces the entire HuggingFace Space pipeline:
      - Preprocessing, topic detection, sentiment analysis, keyword extraction

    Args:
        messages: All messages for the day (from Supabase).
                  Must have: message_id, content, username, created_at, reply_to_id

    Returns:
        {
            "messages": [{"id": "...", "topic_id": 0, "sentiment": "pos", "score": 0.8}, ...],
            "topics": [{"topic_id": 0, "name": "...", "keywords": [...], "insight": "..."}, ...],
            "processing_time_seconds": float,
        }

        On total failure: returns fallback with neutral sentiment and no topics.
    """
    if not messages:
        return _fallback_result([])

    t_start = time.time()
    n = len(messages)

    # Filter to messages with content
    text_messages = [m for m in messages if m.get("content", "").strip()]
    text_messages.sort(key=lambda m: m["created_at"])

    logger.info(f"Groq topic engine: {n} total messages, {len(text_messages)} with text")

    # ── Split into batches ────────────────────────────────────────────────────
    batches = []
    for i in range(0, len(text_messages), MESSAGES_PER_BATCH):
        batches.append(text_messages[i : i + MESSAGES_PER_BATCH])

    total_batches = len(batches)
    logger.info(
        f"Processing in {total_batches} batches "
        f"({MESSAGES_PER_BATCH} msgs/batch, {WAIT_BETWEEN_BATCHES}s between)"
    )

    # ── Process each batch ────────────────────────────────────────────────────
    all_msg_results: list[dict] = []
    all_topics: list[dict] = []
    known_topics: list[dict] = []  # Accumulates across batches for consistency

    for batch_idx, batch in enumerate(batches):
        batch_num = batch_idx + 1
        logger.info(
            f"Batch {batch_num}/{total_batches}: "
            f"{len(batch)} messages"
        )

        try:
            user_prompt = _build_batch_prompt(
                batch, known_topics, batch_num, total_batches
            )

            raw = _call_groq_with_retry(SYSTEM_PROMPT, user_prompt)
            parsed = _parse_response(raw)

            # Collect message-level results
            batch_msgs = parsed.get("messages", [])
            all_msg_results.extend(batch_msgs)

            # Merge topics: update existing or add new
            batch_topics = parsed.get("topics", [])
            existing_ids = {t["topic_id"] for t in all_topics}
            for bt in batch_topics:
                tid = bt.get("topic_id")
                if tid in existing_ids:
                    # Update existing topic (merge keywords, update insight)
                    for et in all_topics:
                        if et["topic_id"] == tid:
                            # Merge keywords (deduplicate)
                            old_kw = set(et.get("keywords", []))
                            new_kw = bt.get("keywords", [])
                            et["keywords"] = list(old_kw | set(new_kw))[:10]
                            # Use latest insight (later batches have more context)
                            if bt.get("insight"):
                                et["insight"] = bt["insight"]
                            # Update name if provided
                            if bt.get("name"):
                                et["name"] = bt["name"]
                            break
                else:
                    all_topics.append(bt)
                    existing_ids.add(tid)

            # Update known_topics for next batch (compact: just id + name + keywords)
            known_topics = [
                {
                    "topic_id": t["topic_id"],
                    "name": t.get("name", f"Topic {t['topic_id']}"),
                    "keywords": t.get("keywords", [])[:5],
                }
                for t in all_topics
            ]

            logger.info(
                f"Batch {batch_num} complete: "
                f"{len(batch_msgs)} messages analyzed, "
                f"{len(batch_topics)} topics in batch, "
                f"{len(all_topics)} total topics so far"
            )

        except json.JSONDecodeError as e:
            logger.error(
                f"Batch {batch_num}: JSON parse failed: {e}. "
                f"Assigning neutral sentiment to {len(batch)} messages."
            )
            # Fallback: assign neutral to all messages in this batch
            for m in batch:
                all_msg_results.append({
                    "id": m["message_id"],
                    "topic_id": -1,
                    "sentiment": "neutral",
                    "score": 0.5,
                })

        except Exception as e:
            logger.error(
                f"Batch {batch_num}: Groq call failed after retries: {e}. "
                f"Assigning neutral sentiment to {len(batch)} messages."
            )
            for m in batch:
                all_msg_results.append({
                    "id": m["message_id"],
                    "topic_id": -1,
                    "sentiment": "neutral",
                    "score": 0.5,
                })

        # Rate limit: wait between batches
        if batch_num < total_batches:
            logger.info(f"Rate limit: waiting {WAIT_BETWEEN_BATCHES}s before next batch...")
            time.sleep(WAIT_BETWEEN_BATCHES)

    # ── Build final result ────────────────────────────────────────────────────
    processing_time = round(time.time() - t_start, 1)

    # Add messages that had no text (empty content) as neutral/uncategorized
    analyzed_ids = {m["id"] for m in all_msg_results}
    for m in messages:
        if m["message_id"] not in analyzed_ids:
            all_msg_results.append({
                "id": m["message_id"],
                "topic_id": -1,
                "sentiment": "neutral",
                "score": 0.5,
            })

    # Sort topics by topic_id for consistency
    all_topics.sort(key=lambda t: t.get("topic_id", 999))

    n_topics = len([t for t in all_topics if t.get("topic_id", -1) >= 0])
    n_uncategorized = sum(1 for m in all_msg_results if m.get("topic_id", -1) == -1)

    logger.info(
        f"Groq topic engine complete in {processing_time}s — "
        f"{n_topics} topics, {n_uncategorized} uncategorized, "
        f"{len(all_msg_results)} messages processed"
    )

    return {
        "messages": all_msg_results,
        "topics": all_topics,
        "n_topics": n_topics,
        "uncategorized_count": n_uncategorized,
        "processing_time_seconds": processing_time,
    }


def _fallback_result(messages: list[dict]) -> dict:
    """Fallback when Groq is completely unreachable."""
    return {
        "messages": [
            {"id": m["message_id"], "topic_id": -1, "sentiment": "neutral", "score": 0.5}
            for m in messages
        ],
        "topics": [],
        "n_topics": 0,
        "uncategorized_count": len(messages),
        "processing_time_seconds": 0,
    }
