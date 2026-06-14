"""
bot/collector.py — Discord event handler.

Listens to on_message() events in the configured guild + channel.
Filters out bots, DMs, wrong channels, and wrong guilds.
Builds and inserts a Supabase payload for every valid message.

This module is imported by main.py and the `client` instance is started
via asyncio.create_task(client.start(token)) inside the FastAPI lifespan.
No standalone start_bot() function — the lifecycle is managed by main.py.
"""

import os
import logging
import discord
from db.supabase_client import insert_message

logger = logging.getLogger(__name__)

# ── Discord client setup ──────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.message_content = True   # Privileged intent — must be enabled in Dev Portal
intents.members = True           # Required for accurate display_name resolution

client = discord.Client(intents=intents)

# Read at module load time so the values are available before on_message fires.
# These will be None until the process starts with env vars set.
_GUILD_ID: int | None = None
_CHANNEL_ID: int | None = None


def _load_ids() -> tuple[int, int]:
    """Lazy-load guild/channel IDs from environment (so tests can import without env vars)."""
    global _GUILD_ID, _CHANNEL_ID
    if _GUILD_ID is None:
        _GUILD_ID = int(os.environ["DISCORD_GUILD_ID"])
    if _CHANNEL_ID is None:
        _CHANNEL_ID = int(os.environ["DISCORD_CHANNEL_ID"])
    return _GUILD_ID, _CHANNEL_ID


# ── Events ────────────────────────────────────────────────────────────────────

@client.event
async def on_ready():
    guild_id, channel_id = _load_ids()
    logger.info(
        f"Discord bot online as {client.user} | "
        f"Watching guild={guild_id} channel={channel_id}"
    )


@client.event
async def on_message(message: discord.Message):
    """
    Fires for every message the bot can see.

    Filtering order (fail-fast):
        1. Ignore bot messages
        2. Ignore DMs (no guild)
        3. Ignore wrong guild
        4. Ignore wrong channel
        5. Build payload and insert to Supabase

    Errors from Supabase are caught and logged — the bot never crashes
    on a DB write failure.
    """
    guild_id, channel_id = _load_ids()

    # 1. Ignore bots (including itself)
    if message.author.bot:
        return

    # 2. Ignore DMs
    if not message.guild:
        return

    # 3. Ignore wrong guild
    if message.guild.id != guild_id:
        return

    # 4. Ignore wrong channel
    if message.channel.id != channel_id:
        return

    # 5. Build payload
    payload = {
        "message_id":     str(message.id),
        "user_id":        str(message.author.id),
        "username":       message.author.name,
        "display_name":   message.author.display_name,
        "content":        message.content[:2000],          # Discord max anyway
        "created_at":     message.created_at.isoformat(),
        "has_attachment":  len(message.attachments) > 0,
        "is_reply":       message.reference is not None,
        "reply_to_id":    (
            str(message.reference.message_id)
            if message.reference else None
        ),
    }

    logger.info(
        f"[{message.author.name}] message in #{message.channel.name} ({len(message.content)} chars)"
    )
    logger.debug(f"Content preview: {message.content[:40]!r}")

    # 6. Persist to Supabase
    try:
        insert_message(payload)
        logger.debug(f"Stored message_id={payload['message_id']}")
    except Exception as e:
        logger.error(f"Failed to insert message {message.id}: {e}")
