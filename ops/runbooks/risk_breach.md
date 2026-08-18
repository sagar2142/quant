# Runbook: risk limit breach

**Severity: CRITICAL.**

The risk engine blocked an order because a limit was breached. The block itself
is the system working. The question is *why the strategy wanted to place it*.

## Immediately

1. Read the verdict. Every check is reported, not just the first failure - the
   full list is in the `risk_events` row and in the console's Risk screen.
2. Identify which layer fired:
   - **Layer 1 (per order)**: usually a sizing bug or a fat-finger price.
   - **Layer 2 (portfolio)**: exposure or concentration has crept up.
   - **Layer 3 (drawdown ladder)**: this is not a bug, it is the plan working.

## By breach type

| Limit | What it usually means |
|---|---|
| `order_notional` | Sizing bug, or equity is being read wrong |
| `price_band` | Fat finger, a stale reference price, or a genuine gap |
| `position_size` | Strategy is doubling down; check whether it intended to |
| `cluster_concentration` | Correlated names accumulated into one bet |
| `daily_loss` | Session loss limit. Halt is correct - do not raise the limit |
| `drawdown_ladder` | Pre-committed de-risking engaged |
| `order_rate` | Runaway loop. Halt immediately and inspect the strategy |

`order_rate` is the one to treat as an emergency: a strategy generating orders
faster than the limit is malfunctioning, not merely aggressive.

## Do not

- **Do not raise the limit to let the order through.** The limits were set by the
  version of you that was calm. Changing one during a breach is exactly the
  decision the ladder exists to remove.
- **Do not disable a check.** If a limit is genuinely wrong, change it in a
  separate, reviewed commit, outside market hours, with a written reason.

## Resume

Only after the *cause* is understood. A breach that stops recurring because the
market moved is not resolved.
