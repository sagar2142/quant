# Runbook: reconciliation break

**Severity: CRITICAL. Trading halts until this is explained.**

An unexplained break is a system-down event, not a rounding issue (MASTER_PLAN §9).
The broker and your records disagree about what you hold. One of those numbers is
being used to size your next order and you do not know which is right.

---

## Do not

- **Do not adjust the internal record to match the broker.** That destroys the
  evidence needed to find the cause, and if your record was the correct one you
  have just corrupted it.
- **Do not resume trading "to see if it clears".** It will not clear. It will
  compound.
- **Do not assume the broker is right.** The broker is the *reference*, not
  automatically the truth — a missed fill callback on our side and a duplicated
  fill on theirs look identical in the diff.

## Immediately

1. **Halt.** The kill switch, via console or API:
   ```
   curl -X POST http://127.0.0.1:8000/kill \
     -H 'Content-Type: application/json' \
     -d '{"reason":"reconciliation break","operator":"<you>"}'
   ```
   Confirm with `GET /health` that `kill_engaged` is `true`.

2. **Snapshot before anything changes.** Broker positions, internal positions,
   today's fills, and the order table. The state at the moment of detection is
   the only complete evidence you will get.

3. **Record the break** in `reconciliation_breaks` with the run id. Do not
   resolve the row yet.

## Diagnose, in this order

Work from most likely to least. Each has a distinct signature.

| Signature | Likely cause | Confirm by |
|---|---|---|
| Break equals exactly one order's quantity | A fill callback was missed, or applied twice | Search `order_transitions` for that instrument around the break time |
| An order sits in `UNKNOWN` | A submit timed out and was never resolved | `SELECT * FROM orders WHERE state='UNKNOWN'` |
| Break appeared overnight, no orders between | A corporate action was not applied | Check `data.corpactions` for an ex-date on that instrument |
| Broker holds something we never ordered | Manual trade, or a different process on the same account | Broker's own order log — look for orders without our `tag` |
| Break is a small fraction of a share | Broker rounding on a partial fill | Compare against the contract note |

The corporate-action case is the one most often missed: a 2:1 split doubles the
broker's share count overnight while ours stays flat, and no order explains it.

## Resolve

1. Establish which record is correct, with evidence — a contract note, a venue
   order log, a corporate-action announcement. Not a guess.
2. Correct the wrong one, and write down *why* in `reconciliation_breaks.explanation`.
3. Fix the code path that allowed the divergence. A break that is corrected but
   not fixed will recur, and next time it may be larger.
4. Only then release the kill switch:
   ```
   curl -X POST http://127.0.0.1:8000/kill/release \
     -H 'Content-Type: application/json' \
     -d '{"reason":"<what it was>","operator":"<you>"}'
   ```

## Afterwards

- Add a regression test reproducing the divergence. Every reconciliation break
  that reaches production is a test that was missing.
- If the cause was a missed callback, consider whether the order state machine
  needs an additional transition, or whether `UNKNOWN` handling is being skipped
  somewhere.

## Escalate to a full stop if

- The break is larger than one day's expected turnover.
- Two breaks occur within a week on different instruments — that suggests a
  systematic accounting error rather than a one-off.
- You cannot establish which record is correct within one session.

In any of those cases, flatten manually through the broker's own interface and
stay flat until the cause is understood.
