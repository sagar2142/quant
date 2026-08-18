# Runbook: kill switch

## Engage

Console, or:

```
curl -X POST http://127.0.0.1:8000/kill \
  -H 'Content-Type: application/json' \
  -d '{"reason":"<why>","operator":"<who>"}'
```

Both fields are mandatory. An unattributed halt with no stated cause cannot be
reviewed afterwards.

## What it does, and does not

**Does**: blocks every new order at the risk engine. Every check short-circuits
to BLOCK regardless of state.

**Does not**: cancel resting orders, or flatten positions. That is deliberate -
automatic liquidation into a disordered market is a way to turn a problem into a
loss. Flattening is a separate, manual decision.

## After engaging

1. Confirm: `GET /health` shows `kill_engaged: true`.
2. Cancel resting orders manually if they are exposed. Check open orders first.
3. Decide on positions explicitly. Holding is often correct.
4. Follow the runbook for the underlying incident.

## Release

Manual only. There is no automatic release and there should never be: whatever
engaged the switch needs a human to confirm it has been understood.

```
curl -X POST http://127.0.0.1:8000/kill/release \
  -H 'Content-Type: application/json' \
  -d '{"reason":"<what was fixed>","operator":"<who>"}'
```

Before releasing, all of:

- [ ] Cause identified, not merely stopped
- [ ] Reconciliation clean
- [ ] Data quality report clean
- [ ] Fix committed, or a decision recorded that no fix is needed and why

## Test it monthly

An untested kill switch is not a kill switch. Engage it in paper, confirm orders
are blocked, release it, confirm they resume. Record the date.
