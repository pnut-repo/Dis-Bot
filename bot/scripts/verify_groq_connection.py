"""
scripts/verify_groq_connection.py
===================================
Quick live test: sends a minimal summary JSON to Groq and prints the response.
Run from the bot/ directory:
    python scripts/verify_groq_connection.py
"""

import os, sys, json
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY or GROQ_API_KEY == "gsk_placeholder":
    print("❌  GROQ_API_KEY is missing or still a placeholder in .env")
    sys.exit(1)

# Minimal summary JSON — same shape as orchestrator Step 9
sample_summary = {
    "date": "2026-06-13",
    "total_messages": 142,
    "text_messages": 138,
    "total_users": 12,
    "total_topics": 4,
    "overall_sentiment": {"positive": 0.42, "neutral": 0.45, "negative": 0.13},
    "top_topics": [
        {
            "rank": 1, "name": "game · valorant · ranked",
            "initiator": "dowedoes7443", "message_count": 47,
            "unique_users": 8, "duration_minutes": 93.4,
            "engagement_score": 0.91, "peak_hour": 21,
            "sentiment": {"positive": 0.55, "neutral": 0.38, "negative": 0.07},
            "top_participants": ["dowedoes7443", "pnut_repo", "ace7"],
        },
        {
            "rank": 2, "name": "movie · weekend · watch",
            "initiator": "pnut_repo", "message_count": 31,
            "unique_users": 5, "duration_minutes": 44.1,
            "engagement_score": 0.62, "peak_hour": 19,
            "sentiment": {"positive": 0.35, "neutral": 0.52, "negative": 0.13},
            "top_participants": ["pnut_repo", "ace7"],
        },
    ],
    "most_active_users": [
        {"username": "dowedoes7443", "message_count": 58},
        {"username": "pnut_repo",    "message_count": 34},
        {"username": "ace7",         "message_count": 21},
    ],
}

print("\n" + "═" * 60)
print("  Groq Connectivity Verification")
print("═" * 60)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.groq_reporter import generate_narrative_report

print(f"\n  Model : meta-llama/llama-4-scout-17b-16e-instruct")
print(f"  Input : {len(json.dumps(sample_summary))} chars of summary JSON")
print(f"\n  Calling Groq API...\n")

report = generate_narrative_report(sample_summary)

if report.startswith("# Daily Report") and "failed" in report:
    print("  ❌  Groq call failed — see error above")
    sys.exit(1)

print("─" * 60)
print(report)
print("─" * 60)
print(f"\n  ✅  Groq is working! Report: {len(report)} chars")
print("═" * 60 + "\n")
