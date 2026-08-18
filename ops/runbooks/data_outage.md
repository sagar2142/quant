# Runbook: data outage / stale feed

**Severity: WARN above 2s, CRITICAL above 10s (MASTER_PLAN 12.7).**

A stale price means every position is being valued, and every risk check
evaluated, against fiction. The risk engine fails closed on a missing price, but
a *stale* price is worse than a missing one: it looks valid.

## Immediately

1. Check whether it is us or them. `GET /health`, then the venue's own status
   page. A feed that is down for everyone needs patience; one that is down only
   for us needs a restart.
2. If any position is open, **halt new orders**. Existing positions are not
   automatically closed - flattening into an unknown market is its own risk.
3. Note the last good timestamp. Any decision taken after it is suspect and may
   need reversing.

## Diagnose

| Signature | Cause | Action |
|---|---|---|
| One feed stale, others fine | Venue-side outage or our websocket dropped | Restart that connector; check venue status |
| All feeds stale | Network, DNS, or the machine slept | Check connectivity from the VPS itself |
| Staleness climbs steadily | Consumer is falling behind, not the feed | Check CPU and the event queue depth |
| Clock drift warnings alongside | NTP problem | `timedatectl status`; a drifting clock corrupts every timestamp |

The third case is the sneaky one: the feed is healthy and *we* are slow. It
presents identically in the vitals bar and needs a completely different fix.

## Resume

- Only after a full bar has arrived cleanly and reconciliation is clean.
- Re-run the data quality report for the affected window before trusting it:
  `python -m apps.cli.quality`
- If bars were missed, backfill them and confirm the gap check passes. A gap
  left in the lake becomes a silently wrong backtest six months later.

## Escalate

If staleness exceeds one full session, treat the day's data as suspect. Do not
backfill and resume as though nothing happened - mark the dataset version and
note the gap on any experiment that used it.
