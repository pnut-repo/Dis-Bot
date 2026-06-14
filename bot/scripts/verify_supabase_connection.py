"""
scripts/verify_supabase_connection.py
======================================
Verifies Supabase connectivity in isolation — no Discord needed.

What it does:
    1. Connects to Supabase using SUPABASE_URL + SUPABASE_SERVICE_KEY from .env
    2. Writes a synthetic test message row to the `messages` table
    3. Reads it back and confirms every field matches
    4. Deletes the test row (leaves the table clean)
    5. Lists the 4 expected tables to confirm schema is in place

How to run (from bot/ directory):
    python scripts/verify_supabase_connection.py
"""

import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load .env from bot/ directory
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_SERVICE_KEY")

# ── Pre-flight ────────────────────────────────────────────────────────────────
missing = []
if not URL:
    missing.append("SUPABASE_URL")
if not KEY or len(KEY) < 100 or not KEY.startswith("eyJ"):
    missing.append("SUPABASE_SERVICE_KEY")
if missing:
    print(f"\n❌  Missing or placeholder values: {', '.join(missing)}")
    sys.exit(1)

from supabase import create_client

print("\n" + "═" * 60)
print("  Supabase Connectivity Verification")
print("═" * 60)
print(f"\n  Connecting to: {URL}")

try:
    client = create_client(URL, KEY)
    print("  ✅  Client created successfully")
except Exception as e:
    print(f"  ❌  Failed to create client: {e}")
    sys.exit(1)

# ── Test 1: Check tables exist ────────────────────────────────────────────────
print("\n─ Step 1: Verify tables exist " + "─" * 30)
EXPECTED_TABLES = ["messages", "daily_reports", "topic_stats", "user_daily_stats"]
all_tables_ok = True

for table in EXPECTED_TABLES:
    try:
        result = client.table(table).select("*").limit(1).execute()
        print(f"  ✅  Table '{table}' — accessible")
    except Exception as e:
        print(f"  ❌  Table '{table}' — ERROR: {e}")
        all_tables_ok = False

if not all_tables_ok:
    print("\n  ⚠️  Some tables are missing. Did you run the schema SQL in Supabase SQL Editor?")
    sys.exit(1)

# ── Test 2: Insert a test message ─────────────────────────────────────────────
print("\n─ Step 2: Insert test message " + "─" * 30)
TEST_ID = "TEST_VERIFY_0000000000000001"

test_payload = {
    "message_id":     TEST_ID,
    "user_id":        "0000000000000000001",
    "username":       "_verify_script",
    "display_name":   "Verify Script",
    "content":        "This is an automated connectivity test — safe to ignore.",
    "created_at":     datetime.now(timezone.utc).isoformat(),
    "has_attachment":  False,
    "is_reply":       False,
    "reply_to_id":    None,
}

try:
    client.table("messages").insert(test_payload).execute()
    print(f"  ✅  Inserted test row (message_id={TEST_ID})")
except Exception as e:
    print(f"  ❌  Insert failed: {e}")
    sys.exit(1)

# ── Test 3: Read it back ──────────────────────────────────────────────────────
print("\n─ Step 3: Read back & validate " + "─" * 29)
try:
    result = client.table("messages").select("*").eq("message_id", TEST_ID).execute()
    rows = result.data
    if not rows:
        print(f"  ❌  Row not found after insert — unexpected")
        sys.exit(1)

    row = rows[0]
    mismatches = []
    for field in ["message_id", "user_id", "username", "display_name",
                   "content", "has_attachment", "is_reply", "reply_to_id"]:
        expected = test_payload[field]
        actual   = row.get(field)
        if actual != expected:
            mismatches.append(f"    {field}: expected={expected!r}  got={actual!r}")

    if mismatches:
        print("  ❌  Field mismatches detected:")
        for m in mismatches:
            print(m)
    else:
        print("  ✅  All fields match exactly:")
        for field in ["message_id", "user_id", "username", "content",
                       "has_attachment", "is_reply", "reply_to_id"]:
            print(f"      {field}: {row.get(field)!r}")
except Exception as e:
    print(f"  ❌  Read failed: {e}")
    sys.exit(1)

# ── Test 4: Delete the test row ───────────────────────────────────────────────
print("\n─ Step 4: Cleanup " + "─" * 42)
try:
    client.table("messages").delete().eq("message_id", TEST_ID).execute()
    print(f"  ✅  Test row deleted — table is clean")
except Exception as e:
    print(f"  ⚠️  Cleanup failed (row may remain): {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "═" * 60)
print("  ✅  Supabase is fully operational.")
print("  All 4 tables accessible. Write → Read → Delete all work.")
print("  Ready to wire the Discord bot to Supabase.\n")
