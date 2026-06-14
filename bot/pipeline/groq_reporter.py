"""
pipeline/groq_reporter.py — Groq Narrative Report Generator
=============================================================
Takes the structured summary JSON (built by orchestrator Step 9) and sends
it to Groq's Llama 4 Scout model. Returns a Markdown narrative report
(~2,000 tokens / ~500 words) that the frontend displays as the Daily Digest.

This module owns all prompt engineering.
The orchestrator builds the data; this module formats, calls, and returns.

Model: meta-llama/llama-4-scout-17b-16e-instruct
Usage: 1–2 requests/day × ~8,000 tokens ≪ 500,000 token/day free tier limit.
"""

import os
import json
import logging
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


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Discord community analyst. You write a concise, insightful daily digest for server moderators.

**Rules:**
1. Write in Markdown format with clear headings.
2. Start with a one-paragraph executive summary (total messages, users, overall mood).
3. Highlight the top 3–5 topics by engagement — for each, mention the topic name, who started it, how many people joined, and the sentiment.
4. Note any interesting patterns: quiet hours, sentiment spikes, unusually active users.
5. End with a brief "Community Health" assessment (1–2 sentences).
6. Keep the total report under 500 words.
7. Be factual — only reference data from the provided JSON. Do not invent usernames, topics, or numbers.
8. Use a friendly but professional tone."""


# ── User Prompt Builder ───────────────────────────────────────────────────────

def build_user_prompt(summary_json: dict) -> str:
    """
    Converts the orchestrator's summary dict into the Groq user prompt.

    The summary JSON contains:
      - date, total_messages, text_messages, total_users, total_topics
      - overall_sentiment {positive, neutral, negative}
      - top_topics [{rank, name, initiator, message_count, unique_users, ...}]
      - most_active_users [{username, message_count}]

    No raw Discord messages ever reach Groq — only aggregate statistics.
    This protects user privacy and keeps the prompt compact (~3–4k tokens).
    """
    return (
        f"Here is the structured analysis for **{summary_json['date']}**.\n\n"
        f"Write a daily digest report based on this data:\n\n"
        f"```json\n{json.dumps(summary_json, indent=2)}\n```"
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────

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
        # The orchestrator does not need try/except around this call.
        return (
            f"# Daily Report — {summary_json.get('date', 'Unknown')}\n\n"
            f"*Report generation failed. Raw stats: "
            f"{summary_json.get('total_messages', '?')} messages from "
            f"{summary_json.get('total_users', '?')} users.*"
        )
