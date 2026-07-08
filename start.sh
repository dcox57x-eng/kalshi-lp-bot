#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export KALSHI_API_KEY_ID="8d1171eb-5ca8-46b7-aeb0-4cbe7db195c9"
export KALSHI_PRIVATE_KEY_PATH="./keys/kalshi_private.pem"
export KALSHI_ENV="prod"

if [ ! -f "$KALSHI_PRIVATE_KEY_PATH" ]; then
    echo "Private key not found at $KALSHI_PRIVATE_KEY_PATH"
    exit 1
fi

pip3 install -q -r requirements.txt 2>/dev/null

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OPTIMUS v3 — Hedged Liquidity + Risk Engine"
echo "  Environment: $KALSHI_ENV"
echo ""
echo "  Risk Guards: MDD 8% | VaR \$15/day | 2-loss halt"
echo "  Kelly: Fractional 0.25x | Exposure cap \$40"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── OPTIMAL CONFIGURATION ──
# Derived from core formulas + backtest + production log analysis:
#
# --contracts 1     : Down from 5. Logs show 0 matched contracts at qty=5
#                     in thin books. Single contracts fill reliably.
# --delta-max 2     : Tight edge requirement — only clear arb opportunities.
# --min-bids 2      : Relaxed from 3. Was causing 297 skips per session.
#                     New risk guards (MDD/VaR/halt) protect downside.
# --max-hedges 3    : Concentration over diversification per paper.
# --poll 3          : Faster cycle. Target <1s per images; 3s is practical
#                     minimum with REST polling.
# --kelly-alpha 0.25: NEVER full Kelly on short windows (paper).
#                     0.25x reduces f* from 77.5% to 19.4%.
# --max-exposure 40 : Hard cap at $40 (40% of bankroll).
# --max-drawdown 0.08: Circuit breaker at 8% MDD per formulas.
# --max-consec-losses 2: Halt after 2 consecutive losses per spec.
# --daily-var-limit 15: Conservative $15/day loss limit (15% of bankroll).
# --series: BTC+ETH are the only series producing fills. Keep 4 others
#           for diversification but they'll be naturally deprioritized.

exec python3 main.py --live \
    --budget 98 \
    --contracts 1 \
    --delta-max 2 \
    --max-hedges 3 \
    --min-bids 2 \
    --poll 3 \
    --kelly-alpha 0.25 \
    --max-exposure 40 \
    --max-drawdown 0.08 \
    --max-consec-losses 2 \
    --daily-var-limit 15 \
    --series KXBTC KXETH KXINX KXNASDAQ100 KXSPY KXGOLD
