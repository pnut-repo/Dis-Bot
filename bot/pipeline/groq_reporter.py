"""
pipeline/groq_reporter.py — Groq Narrative Report Generator (API 2)
=====================================================================
Uses llama-3.3-70b-versatile to generate a polished daily narrative report.

This is the SECOND Groq API in the two-stage pipeline:
  API 1 (groq_topic_engine.py): llama-4-scout → topic/sentiment analysis
  API 2 (this file):            llama-3.3-70b → narrative report generation

The 70b model produces more coherent, well-structured prose than the scout model.

Token Budget (free tier):
  - Model: llama-3.3-70b-versatile
  - Limit: 12k TPM, 30 RPM
  - Input: ~5k tokens (summary JSON + system prompt)
  - Output: ~3k tokens (narrative report)
  - Total: ~8k tokens — comfortably within 12k TPM
  - Called once per day.
"""

import os
import json
import logging

from groq import Groq

logger = logging.getLogger(__name__)

# ── Groq client (singleton) ───────────────────────────────────────────────────

_client: Groq | None = None

MODEL       = "llama-3.3-70b-versatile"
MAX_TOKENS  = 3000   # Output cap — report is ~2,000 tokens
TEMPERATURE = 0.7    # Slightly creative prose, grounded by structured data


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY_VERSATILE")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY_VERSATILE environment variable not set")
        _client = Groq(api_key=api_key)
    return _client


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Discord community analyst. You write a concise, insightful daily digest for server moderators.

**Rules:**
1. Write in Markdown format with clear headings: Executive Summary, Topic Deep-Dives, Temporal Pattern, and Community Pulse.
2. **Executive Summary**: One sentence covering key numbers and the dominant mood.
3. **Topic Deep-Dives**: Highlight the top 3-5 topics by engagement. Use the per-topic AI insights and batch mini-reports to answer: What was said? What was the emotional arc? Who were the key actors? What were the unresolved threads?
4. **Temporal Pattern**: Use the batch_reports to describe when things heated up, when sentiment spikes occurred, and when there were dead hours.
5. **Community Pulse**: End with a 2-sentence holistic health check of the community.
6. **Be concrete and factual**: Avoid vague language. Reference actual topic names, usernames, and sentiment patterns from the data. Do not invent details.
7. Keep the total report under 500 words.
8. Use a friendly but professional tone."""


# ── Report Generation ─────────────────────────────────────────────────────────

def build_user_prompt(summary_json: dict) -> str:
    """
    Converts the orchestrator's summary dict into the Groq user prompt.
    """
    return (
        f"Here is the structured analysis for **{summary_json['date']}**.\n\n"
        f"Write a daily digest report based on this data:\n\n"
        f"```json\n{json.dumps(summary_json, indent=2)}\n```"
    )


def generate_narrative_report(summary_json: dict) -> str:
    """
    Sends the summary JSON to Groq llama-3.3-70b and returns a Markdown report.

    Args:
        summary_json: Structured dict from orchestrator (includes topic insights).

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
