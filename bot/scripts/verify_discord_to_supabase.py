"""
scripts/verify_discord_to_supabase.py
======================================
End-to-end test: Discord message → Supabase messages table.

What it does:
    1. Connects to Discord (same as verify_discord_connection.py)
    2. On each message received, builds the payload AND inserts it into Supabase
    3. Immediately reads the row back from Supabase to confirm it landed
    4. Prints both the payload and the DB read result side-by-side

This is the full collector loop — exactly what main.py will do in production,
but with verbose logging so you can see every step.

How to run (from bot/ directory):
    python scripts/verify_discord_to_supabase.py
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timezone

import discord
from dotenv import load_dotenv

# Load .env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

TOKEN      = os.getenv("DISCORD_TOKEN")
GUILD_ID   = os.getenv("DISCORD_GUILD_ID")
CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")
SUPA_URL   = os.getenv("SUPABASE_URL")
SUPA_KEY   = os.getenv("SUPABASE_SERVICE_KEY")

# Pre-flight
missing = []
for name, val in [("DISCORD_TOKEN", TOKEN), ("DISCORD_GUILD_ID", GUILD_ID),
                   ("DISCORD_CHANNEL_ID", CHANNEL_ID)]:
    if not val:
        missing.append(name)
if not SUPA_URL:
    missing.append("SUPABASE_URL")
if not SUPA_KEY or len(SUPA_KEY) < 100:
    missing.append("SUPABASE_SERVICE_KEY")

if missing:
    print(f"\n❌  Missing env vars: {', '.join(missing)}")
    sys.exit(1)

GUILD_ID   = int(GUILD_ID)
CHANNEL_ID = int(CHANNEL_ID)

# Supabase client
from supabase import create_client
supabase = create_client(SUPA_URL, SUPA_KEY)

logging.basicConfig(level=logging.WARNING, format="%(message)s")

# Discord client
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)

message_count = 0


@client.event
async def on_ready():
    print("\n" + "═" * 60)
    print(f"  ✅  Bot: {client.user}")
    guild = client.get_guild(GUILD_ID)
    channel = guild.get_channel(CHANNEL_ID) if guild else None

    if not guild:
        print(f"  ❌  Guild {GUILD_ID} not found"); await client.close(); return
    if not channel:
        print(f"  ❌  Channel {CHANNEL_ID} not found"); await client.close(); return

    print(f"  ✅  Guild: {guild.name!r}  |  Channel: #{channel.name}")
    print(f"  ✅  Supabase: {SUPA_URL}")
    print(f"\n  💾  Messages will be WRITTEN to Supabase and READ back.")
    print(f"      Send a message in #{channel.name}. Press Ctrl+C to stop.\n")
    print("═" * 60)


@client.event
async def on_message(message: discord.Message):
    global message_count

    if message.author.bot:
        return
    if not message.guild or message.guild.id != GUILD_ID:
        return
    if message.channel.id != CHANNEL_ID:
        return

    message_count += 1

    # Build payload (identical to bot/collector.py)
    payload = {
        "message_id":     str(message.id),
        "user_id":        str(message.author.id),
        "username":       message.author.name,
        "display_name":   message.author.display_name,
        "content":        message.content[:2000],
        "created_at":     message.created_at.isoformat(),
        "has_attachment":  len(message.attachments) > 0,
        "is_reply":       message.reference is not None,
        "reply_to_id":    str(message.reference.message_id) if message.reference else None,
    }

    print(f"\n  📨  Message #{message_count} from @{payload['username']}")
    print(f"      Content: {payload['content']!r}")

    # ── WRITE to Supabase ─────────────────────────────────────────────────
    try:
        supabase.table("messages").insert(payload).execute()
        print(f"  💾  WRITE OK — inserted message_id={payload['message_id']}")
    except Exception as e:
        print(f"  ❌  WRITE FAILED: {e}")
        return

    # ── READ back from Supabase ───────────────────────────────────────────
    try:
        result = (
            supabase.table("messages")
            .select("message_id, user_id, username, content, has_attachment, is_reply")
            .eq("message_id", payload["message_id"])
            .execute()
        )
        rows = result.data
        if rows:
            row = rows[0]
            print(f"  📖  READ OK — verified in DB:")
            print(f"      message_id   : {row['message_id']}")
            print(f"      username     : {row['username']}")
            print(f"      content      : {row['content']!r}")
            print(f"      has_attachment: {row['has_attachment']}")
            print(f"      is_reply     : {row['is_reply']}")

            # Validate
            if row["content"] == payload["content"] and row["username"] == payload["username"]:
                print(f"  ✅  MATCH — Discord → Supabase round-trip verified!")
            else:
                print(f"  ⚠️  MISMATCH detected between payload and DB row")
        else:
            print(f"  ❌  READ returned 0 rows — row not found after insert")
    except Exception as e:
        print(f"  ❌  READ FAILED: {e}")

    print("─" * 60)


async def main():
    try:
        print("\n  Connecting to Discord + Supabase...")
        await client.start(TOKEN)
    except discord.LoginFailure:
        print("\n  ❌  Discord login failed — check DISCORD_TOKEN")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n  Stopped. Stored {message_count} message(s) in Supabase.")
        await client.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n  Stopped. Stored {message_count} message(s) in Supabase.\n")
