"""
Configuration for the Kalshi Hedged Liquidity Provision Bot.

Implements the framework from:
"Optimal Liquidity Provision on Prediction Markets: A Hedged Market-Making Framework"

All monetary values are in CENTS unless otherwise noted.

v2 improvements:
- Tighter delta_max default (3→2) to avoid marginal hedges
- Faster reprice start (8→5s) to catch fills before they slip
- Larger reprice_max_chase (8→12) to complete more hedges
- Shorter emergency cancel (30→20s) to limit exposure time
- Higher min_bids_each_side (2→3) to avoid thin books
- Lower max_active_hedges (4→3) for better concentration
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KalshiConfig:
    """Kalshi API connection settings."""
    api_key_id: str = ""
    private_key_path: str = ""
    base_url: str = "https://demo-api.kalshi.co"  # Use demo by default
    api_path: str = "/trade-api/v2"

    @property
    def is_production(self) -> bool:
        return "demo" not in self.base_url

    @classmethod
    def from_env(cls) -> "KalshiConfig":
        env = os.getenv("KALSHI_ENV", "demo").lower()
        if env == "prod":
            base_url = "https://api.elections.kalshi.com"
        else:
            base_url = "https://demo-api.kalshi.co"

        return cls(
            api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
            private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
            base_url=base_url,
        )


@dataclass
class ScoringConfig:
    """
    Parameters for the reward scoring function (Section 2.1 of the paper).

    S(s) = ((v - s) / v)^2 * b

    v: max spread (max qualifying distance from mid price, in cents)
    b: market multiplier / base reward score
    """
    max_spread_v: int = 5
    market_multiplier_b: float = 1.0


@dataclass
class HedgeConfig:
    """
    Hedging framework parameters (Section 3 of the paper).

    Risk-free condition: p_Y + p_N < 1.00 (in dollars) or < 100 (in cents)
    Max acceptable overpayment: delta_max = R / (n * T)

    v2 tuning rationale (from 2hr run analysis):
    - 56% partial fill rate → need faster repricing
    - 133 failed hedges × ~50¢ avg loss → tighter emergency cancel
    - 6720¢ overpayment vs 70¢ P&L → stricter edge requirements
    """
    # Max acceptable overpayment per share in cents
    # Lowered from 5→2: only take hedges with clear edge
    delta_max_cents: int = 2

    # Time horizon for VaR-style bound (days)
    horizon_days: int = 30

    # Maximum time (seconds) to wait for the hedge side to fill
    hedge_timeout_seconds: int = 15

    # Emergency cancel: reduced from 30→20s to limit naked exposure
    emergency_cancel_seconds: int = 20

    # ── Aggressive reprice on partial fill ──
    # Start repricing sooner: 8→5s (half the fills slip away in the first 8s)
    reprice_after_seconds: int = 5

    # Step size for repricing (used as base for exponential in v2)
    reprice_step_cents: int = 2

    # Max chase increased from 8→12 to complete more hedges
    # The exponential schedule (1,2,4,8,12) reaches max faster
    reprice_max_chase_cents: int = 12

    # ── v2 additions ──
    # Minimum edge (100 - yes - no) required to place a hedge
    # Hedges below this threshold are rejected pre-flight
    min_edge_cents: int = 3


@dataclass
class CapitalConfig:
    """
    Capital allocation parameters (Section 4 of the paper).

    N_orders = B / (4 * m * p)
    where B = budget, m = min shares, p = mid price

    Kelly fraction: f* = (p * b - q) / b
    Fractional Kelly: f = α * f*, α ∈ (0, 1]
    where p = prob of successful hedge, q = 1-p, b = net reward-to-risk ratio
    """
    # Total budget in cents (e.g., 100_00 = $100)
    total_budget_cents: int = 100_00

    # Fraction of total budget to deploy (Kelly or manual)
    deploy_fraction: float = 0.50

    # Number of contracts per order
    contracts_per_order: int = 1

    # Kelly criterion inputs
    hedge_success_probability: float = 0.85  # p
    net_reward_to_risk_ratio: float = 2.0    # b

    # Fractional Kelly multiplier α — NEVER use full Kelly (α=1.0)
    # Paper: "NEVER full Kelly on 5min markets!" Use α=0.25-0.5
    fractional_kelly_alpha: float = 0.25

    # Maximum exposure in cents (exposure + new_bet must not exceed this)
    max_exposure_cents: int = 50_00  # $50 max exposure at any time


@dataclass
class MarketSelectionConfig:
    """
    Market selection criteria (Section 5 / Table 1 of the paper).

    v2 tuning: tighter filters to avoid markets where hedges fail.
    """
    # Price split: prefer markets near 50/50
    min_yes_price_cents: int = 15
    max_yes_price_cents: int = 85

    # Max spread: prefer wider reward zones
    min_spread_cents: int = 3

    # Competition: prefer markets with fewer resting orders
    max_resting_orders: int = 50

    # Volatility: prefer stable markets
    max_price_change_1h_cents: int = 10

    # Minimum volume to ensure market is active
    min_daily_volume: int = 10

    # ── Two-sided book filter ──
    # Lowered from 3→1: 260/260 thin skips confirmed books rarely have 3 deep on both sides
    min_bids_each_side: int = 1

    # ── Concentration ──
    # Reduced from 4→3: better to nail fewer markets than spread too thin
    max_active_hedges: int = 3

    # Maximum number of target markets to consider
    max_target_markets: int = 10

    # ── v2 additions ──
    # Minimum total book depth (yes + no) to consider a market
    min_total_book_depth: int = 10

    # Cooldown after a failed hedge on a specific market (seconds)
    market_failure_cooldown_seconds: int = 120


@dataclass
class RiskConfig:
    """
    Risk management parameters from core formulas reference.

    MDD = (Peak - Trough) / Peak — block new trades if MDD > max_drawdown_pct
    VaR = μ - 1.645 · σ — max daily loss at 95% confidence
    SR = (E[R] - Rf) / σ(R) — target Sharpe > min_sharpe_ratio
    PF = gross_profit / gross_loss — healthy bot PF > min_profit_factor
    """
    # Max Drawdown circuit breaker: block new trades if MDD exceeds this
    max_drawdown_pct: float = 0.08  # 8%

    # Consecutive loss halt: halt trading after N consecutive losses
    # Images spec: 2 consecutive losses → halt
    max_consecutive_losses: int = 2

    # Profit Factor minimum: PF = gross_profit / gross_loss
    min_profit_factor: float = 1.5

    # Sharpe Ratio target (for monitoring/logging)
    min_sharpe_ratio: float = 2.0

    # VaR daily loss limit in cents (at 95% confidence)
    # If cumulative daily loss exceeds this, pause until next session
    daily_var_limit_cents: int = 20_00  # $20 max daily loss

    # Resume trading next morning after halt
    resume_after_halt_seconds: int = 3600  # 1 hour cooldown after risk halt


@dataclass
class BotConfig:
    """Top-level bot configuration."""
    kalshi: KalshiConfig = field(default_factory=KalshiConfig.from_env)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    market_selection: MarketSelectionConfig = field(default_factory=MarketSelectionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # Main loop interval in seconds
    poll_interval_seconds: float = 5.0

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "bot.log"

    # Dry run mode: log orders but don't submit
    dry_run: bool = True

    # Tickers to monitor (empty = auto-select based on criteria)
    target_tickers: list = field(default_factory=list)

    # Series to scan for auto-discovery
    # Updated 2026-05-29: prioritized by current volume (sports dominant, then crypto, then macro)
    target_series: list = field(default_factory=lambda: [
        # ── Sports (highest volume right now — NBA Finals + NHL playoffs active) ──
        "KXNBA",          # NBA game markets and futures
        "KXNHL",          # Stanley Cup playoffs
        "KXMLB",          # MLB daily games
        # ── Crypto (consistent daily volume) ──
        "KXBTC",          # Bitcoin price
        "KXETH",          # Ethereum price
        # ── Macro / Financials (institutional hedging demand) ──
        "KXINX",          # S&P 500
        "KXFED",          # Fed rate decisions
        "KXCPI",          # CPI releases
    ])
