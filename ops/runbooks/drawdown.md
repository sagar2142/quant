# Runbook: drawdown ladder engaged

**Severity: WARN. This is the system working as designed.**

The ladder is pre-committed de-risking (MASTER_PLAN 8). Its entire purpose is to
remove you from the decision at the exact moment you are least capable of making
it. Deciding "should I cut?" during a 9% drawdown is the worst possible time to
think about it.

## The rungs

| Drawdown | Scale to | Halt |
|---|---|---|
| -5% | 50% | no |
| -8% | 25% | no |
| -10% | 0% | yes |

## When a rung engages

1. **Do nothing to the ladder.** It is not negotiable during a drawdown. If you
   believe the rungs are wrong, that is a decision for a calm week, in a
   reviewed commit, with a written rationale.
2. Verify the drawdown is real: reconciliation clean, marks current, no stale
   prices inflating the loss.
3. Check whether it is one strategy or the whole book. A single strategy in
   drawdown while others are flat is a strategy problem; everything down
   together is a market or a systematic-risk problem.

## Recovery

Scaling reverses automatically as equity recovers - the ladder reads current
drawdown, it does not latch. The halting rung at -10% does latch: it requires a
manual kill-switch release after review.

## The question worth asking

Compare the realised drawdown to the Monte Carlo 5th percentile from the
gauntlet (check 11). If the realised figure is well inside the shuffled
distribution, nothing unusual has happened and the strategy is behaving as
measured. If it is outside, the return distribution has changed and the strategy
should be reviewed against its kill criteria (MASTER_PLAN 15).
