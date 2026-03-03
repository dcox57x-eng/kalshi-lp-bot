# Kalshi Hedged Liquidity Provision Bot

An automated market-making bot for [Kalshi](https://kalshi.com) prediction markets, implementing the hedged liquidity provision framework described in *"Optimal Liquidity Provision on Prediction Markets: A Hedged Market-Making Framework"*.

## Strategy Overview

The bot places simultaneous **YES and NO limit orders** on binary prediction markets, earning from the bid-ask spread while bounding worst-case losses through hedging.

**Core idea:** If you buy 1 YES contract at 47¢ and 1 NO contract at 51¢, you pay 98¢ total. At resolution, exactly one side pays out $1.00 — guaranteeing a 2¢ profit regardless of outcome. The bot automates this at scale, selecting optimal markets and prices.

### Key Equations

| Concept | Formula | Reference |
|---|---|---|
| Scoring | `S(s) = ((v - s) / v)² · b` | Eq. 1 |
| Payout share | `R_i = (S_i / ΣS_j) · P` | Eq. 2 |
| Hedge P&L | `π = 1.00 - (p_Y + p_N)` | Eq. 3 |
| Max orders | `N = B / (4 · m · p)` | Eq. 5 |
| Kelly fraction | `f* = (p·b - q) / b` | Eq. 6 |
| Expected return | `E[R] = (Rewards - L_fill) / C_risk` | Eq. 7 |
| Break-even fill rate | `f_BE = Rewards / (n · Δ_max)` | Eq. 8 |

## Setup

### 1. Prerequisites

- Python 3.10+
- A Kalshi account with API access

### 2. Install

```bash
cd kalshi_lp_bot
pip install -r requirements.txt
```

### 3. Get API Credentials

1. Log in to [Kalshi](https://kalshi.com/account/profile)
2. Go to **Profile Settings → API Keys**
3. Click **Create New API Key**
4. Save the private key as `kalshi_private_key.pem`
5. Note the Key ID

### 4. Configure

```bash
cp .env.example .env
# Edit .env with your credentials
```

Or pass credentials via CLI:

```bash
python main.py --key-id YOUR_KEY_ID --key-path ./kalshi_private_key.pem
```

## Usage

### Dry Run (recommended first)

```bash
# Scan markets and simulate orders (no real trades)
python main.py

# With specific tickers
python main.py --tickers TICKER1 TICKER2

# Verbose logging
python main.py --log-level DEBUG
```

### Live Trading on Demo

```bash
# Demo environment (fake money)
KALSHI_ENV=demo python main.py --live --budget 100
```

### Live Trading on Production

```bash
# ⚠ REAL MONEY — requires confirmation prompt
KALSHI_ENV=prod python main.py --live --budget 100
```

### All Options

```
python main.py --help

Options:
  --live              Enable live order placement (default: dry run)
  --tickers T [T ..] Specific market tickers to target
  --budget DOLLARS    Total budget in dollars (default: 100)
  --contracts N       Contracts per order (default: 1)
  --delta-max CENTS   Max overpayment per share in cents (default: 3)
  --poll SECONDS      Poll interval (default: 5)
  --log-level LEVEL   DEBUG, INFO, WARNING, ERROR
  --log-file PATH     Log file path (default: bot.log)
  --key-id ID         Kalshi API key ID
  --key-path PATH     Path to RSA private key
```

## Architecture

```
kalshi_lp_bot/
├── main.py           # CLI entry point
├── config.py         # All configuration dataclasses
├── kalshi_client.py  # Kalshi REST API client (RSA-PSS auth)
├── strategy.py       # Core math: scoring, hedging, Kelly, market selection
├── bot.py            # Main loop, order lifecycle, state management
├── requirements.txt
├── .env.example
└── README.md
```

### Bot Loop (Algorithm 2)

```
while running:
    1. Discover & rank markets (every 5 min)
       → Filter by price split, spread, volume, competition
       → Score and sort by composite suitability

    2. Place hedged orders on top markets
       → Compute optimal YES/NO prices near mid
       → Ensure p_Y + p_N ≤ 100 + Δ_max
       → Use batch API for atomicity, post_only to avoid taking

    3. Monitor active hedges
       → Both filled → record P&L, remove
       → One side filled (adverse fill!) → wait for hedge window
       → Timeout → emergency cancel, record loss
       → Stale (>5min no fills) → cancel and replace

    4. Sleep and repeat
```

## Risk Management

The bot implements several safety layers:

- **Hedge bound (Δ_max):** Maximum overpayment per contract is capped. Default 3¢ means worst case per hedge is -3¢, but expected reward exceeds this.
- **Kelly sizing:** Capital deployed is limited by the Kelly criterion based on hedge success probability.
- **Post-only orders:** Orders are rejected if they would immediately match (avoiding being a taker).
- **Emergency cancellation:** If one side fills and the other can't be hedged within the timeout, the position is abandoned.
- **Dry run mode:** Default mode simulates everything without placing real orders.
- **Production confirmation:** Live production trading requires typing "YES" at a confirmation prompt.

## Market Selection Criteria (Table 1)

| Criterion | Preferred | Rationale |
|---|---|---|
| Price split | ~50/50 | Symmetric pricing reduces hedge skew |
| Max spread | ≥ 3¢ | Wider reward zone, easier qualification |
| Min shares | Low | Lower capital commitment per order |
| Competition | Low | Larger share of reward pool |
| Volatility | Low | Reduces probability of sudden adverse fills |

Best targets: niche political primaries, regional referendums, minor sporting events — markets with active reward pools but few competing liquidity providers.

## ⚠ Disclaimers

- **This bot trades real money in live mode.** Use demo environment for testing.
- **Past performance of the mathematical framework does not guarantee results.**
- **Adverse fills can and will happen.** The hedging framework bounds but does not eliminate risk.
- **API changes:** Kalshi may update their API. Verify endpoints against [docs.kalshi.com](https://docs.kalshi.com) before running.
- **Regulatory:** Kalshi is a CFTC-regulated exchange. Ensure you comply with all applicable regulations.
- **Not financial advice.** This is educational/research software.
