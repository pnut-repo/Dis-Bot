-- ════════════════════════════════════════════════════════════════════════════
-- daily_analytics table — Expanded analytics data for the dashboard
-- ════════════════════════════════════════════════════════════════════════════
-- This table stores pre-computed chart data and per-user analytics.
-- Retained for 30 days (even after raw messages are purged at 3 days).
-- One row per day, ~10-50 KB per row depending on user count.

CREATE TABLE IF NOT EXISTS public.daily_analytics (
  report_date date NOT NULL,
  hourly_data jsonb NULL,          -- [{hour: 0, count: 45}, ...] × 24
  sentiment_hourly jsonb NULL,     -- [{hour: 0, positive: 0.3, neutral: 0.5, negative: 0.2, count: 45}, ...]
  user_analytics jsonb NULL,       -- {username: {hourly_activity, sentiment, topics, message_count, active_hours}}
  topic_analytics jsonb NULL,      -- [{name, engagement_score, message_count, ...}]
  summary_stats jsonb NULL,        -- {total_messages, total_users, total_topics, overall_sentiment}
  created_at timestamp with time zone DEFAULT now(),
  CONSTRAINT daily_analytics_pkey PRIMARY KEY (report_date)
) TABLESPACE pg_default;


-- ════════════════════════════════════════════════════════════════════════════
-- Updated Purge Policies (pg_cron)
-- ════════════════════════════════════════════════════════════════════════════
-- Messages: 3-day retention (saves Supabase row count)
-- Analytics + reports + topics + user_stats: 30-day retention
-- Audit log: indefinite

-- First, remove old purge jobs if they exist
SELECT cron.unschedule('purge-old-messages');

-- Messages: purge older than 3 days (runs daily at 00:15 UTC)
SELECT cron.schedule(
  'purge-old-messages',
  '15 0 * * *',
  $$DELETE FROM public.messages WHERE created_at < now() - interval '3 days'$$
);

-- Daily reports: purge older than 30 days
SELECT cron.schedule(
  'purge-old-reports',
  '20 0 * * *',
  $$DELETE FROM public.daily_reports WHERE report_date < (current_date - 30)$$
);

-- Topic stats: purge older than 30 days
SELECT cron.schedule(
  'purge-old-topics',
  '21 0 * * *',
  $$DELETE FROM public.topic_stats WHERE report_date < (current_date - 30)$$
);

-- User daily stats: purge older than 30 days
SELECT cron.schedule(
  'purge-old-user-stats',
  '22 0 * * *',
  $$DELETE FROM public.user_daily_stats WHERE report_date < (current_date - 30)$$
);

-- Daily analytics: purge older than 30 days
SELECT cron.schedule(
  'purge-old-analytics',
  '23 0 * * *',
  $$DELETE FROM public.daily_analytics WHERE report_date < (current_date - 30)$$
);
