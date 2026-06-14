"""
scripts/verify_discord_connection.py
=====================================
Standalone script to verify Discord connectivity WITHOUT Supabase or any other
dependency. Run this FIRST to confirm your bot token, guild ID, and channel ID
are correct and that the bot can read messages from your server.

What it does:
    1. Connects to Discord using your bot token.
    2. Confirms it can see your target guild.
    3. Confirms it can see your target channel.
    4. Waits for messages in that channel and prints each one to the terminal.
    5. Press Ctrl+C to stop.

How to run (from bot/ directory):
    python scripts/verify_discord_connection.py

Required env vars (in bot/.env or exported):
    DISCORD_TOKEN
    DISCORD_GUILD_ID
    DISCORD_CHANNEL_ID
"""

import asyncio
import os
import sys
import logging
from datetime import datetime, timezone

import discord
from dotenv import load_dotenv

# ── Load env ──────────────────────────────────────────────────────────────────

# Look for .env in the bot/ directory (parent of scripts/)
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

TOKEN      = os.getenv("DISCORD_TOKEN")
GUILD_ID   = os.getenv("DISCORD_GUILD_ID")
CHANNEL_ID = os.getenv("DISCORD_CHANNEL_ID")

# ── Pre-flight checks ─────────────────────────────────────────────────────────

missing = [k for k, v in [
    ("DISCORD_TOKEN",      TOKEN),
    ("DISCORD_GUILD_ID",   GUILD_ID),
    ("DISCORD_CHANNEL_ID", CHANNEL_ID),
] if not v]

if missing:
    print(f"\n❌  Missing environment variables: {', '.join(missing)}")
    print(f"    Copy bot/.env.example → bot/.env and fill in your values.\n")
    sys.exit(1)

GUILD_ID   = int(GUILD_ID)
CHANNEL_ID = int(CHANNEL_ID)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,   # Suppress discord.py noise; we handle output ourselves
    format="%(message)s",
)

# ── Discord client ────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = discord.Client(intents=intents)

message_count = 0


@client.event
async def on_ready():
    """Fires once when the bot successfully connects to Discord."""
    print("\n" + "═" * 60)
    print(f"  ✅  Bot connected as: {client.user} (ID: {client.user.id})")
    print("═" * 60)

    # Check guild
    guild = client.get_guild(GUILD_ID)
    if guild is None:
        print(f"\n  ❌  Guild ID {GUILD_ID} not found.")
        print("      Is the bot invited to this server? Check DISCORD_GUILD_ID.\n")
        await client.close()
        return

    print(f"\n  ✅  Guild found:   {guild.name!r} ({guild.member_count} members)")

    # Check channel
    channel = guild.get_channel(CHANNEL_ID)
    if channel is None:
        print(f"\n  ❌  Channel ID {CHANNEL_ID} not found in {guild.name!r}.")
        print("      Check DISCORD_CHANNEL_ID.\n")
        await client.close()
        return

    print(f"  ✅  Channel found: #{channel.name} (ID: {channel.id})")
    print(f"\n  👂  Listening for messages in #{channel.name}...")
    print(f"      Send a message in Discord now. Press Ctrl+C to stop.\n")
    print("─" * 60)


@client.event
async def on_message(message: discord.Message):
    """Fires for every message the bot can see."""
    global message_count

    # Only care about our target guild + channel
    if message.author.bot:
        return
    if not message.guild or message.guild.id != GUILD_ID:
        return
    if message.channel.id != CHANNEL_ID:
        return

    message_count += 1
    now = datetime.now(timezone.utc)

    # Build the exact same payload that insert_message() would store in Supabase
    payload = {
        "message_id":     str(message.id),
        "user_id":        str(message.author.id),
        "username":       message.author.name,
        "display_name":   message.author.display_name,
        "content":        message.content[:2000],
        "created_at":     message.created_at.isoformat(),
        "has_attachment":  len(message.attachments) > 0,
        "is_reply":       message.reference is not None,
        "reply_to_id":    (
            str(message.reference.message_id)
            if message.reference else None
        ),
    }

    # ── Pretty print to terminal ──────────────────────────────────────────────
    print(f"\n  📨  Message #{message_count} received at {now.strftime('%H:%M:%S')} UTC")
    print(f"  ┌─ message_id   : {payload['message_id']}")
    print(f"  ├─ user_id      : {payload['user_id']}")
    print(f"  ├─ username     : {payload['username']}")
    print(f"  ├─ display_name : {payload['display_name']}")
    print(f"  ├─ content      : {payload['content']!r}")
    print(f"  ├─ created_at   : {payload['created_at']}")
    print(f"  ├─ has_attachment: {payload['has_attachment']}")
    print(f"  ├─ is_reply     : {payload['is_reply']}")
    print(f"  └─ reply_to_id  : {payload['reply_to_id']}")
    print("─" * 60)

    # Validation — flag any unexpected states
    issues = []
    if not payload["message_id"].isdigit():
        issues.append("⚠️  message_id is not a numeric string (Discord snowflake)")
    if not payload["user_id"].isdigit():
        issues.append("⚠️  user_id is not a numeric string")
    if not payload["content"] and not payload["has_attachment"]:
        issues.append("⚠️  Empty content with no attachment — may be a sticker/embed")
    if len(payload["content"]) > 2000:
        issues.append("⚠️  content exceeds 2000 chars (should be truncated by bot)")

    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"  ✅  Payload valid — ready to insert into Supabase messages table")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    try:
        print("\n  Connecting to Discord...")
        await client.start(TOKEN)
    except discord.LoginFailure:
        print("\n  ❌  Login failed — DISCORD_TOKEN is invalid or revoked.")
        print("      Get a new token from: https://discord.com/developers/applications\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\n  Stopped. Captured {message_count} message(s) total.")
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n  Stopped. Captured {message_count} message(s) total.\n")
