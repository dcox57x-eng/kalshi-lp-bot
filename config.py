"""
Configuration for the Kalshi Hedged Liquidity Provision Bot.

Implements the framework from:
"Optimal Liquidity Provision on Prediction Markets: A Hedged Market-Making Framework"

All monetary values are in CENTS unless otherwise noted.
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

    Kalshi doesn't have Polymarket's exact reward program, but this scoring
    model is used to estimate the value of order placement proximity to mid.

    v: max spread (max qualifying distance from mid price, in cents)
    b: market multiplier / base reward score
    """
    max_spread_v: int = 5       # Max qualifying distance from mid in cents
    market_multiplier_b: float = 1.0  # Market-specific multiplier


@dataclass
class HedgeConfig:
    """
    Hedging framework parameters (Section 3 of the paper).

    Risk-free condition: p_Y + p_N < 1.00 (in dollars) or < 100 (in cents)
    Max acceptable overpayment: delta_max = R / (n * T)
    """
    # Max acceptable overpayment per share in cents (Section 3.2)
    # delta_max = expected_daily_reward / (num_shares * horizon_days)
    delta_max_cents: int = 3

    # Time horizon for VaR-style bound (days)
    horizon_days: int = 30

    # Maximum time (seconds) to wait for the hedge side to fill
    hedge_timeout_seconds: int = 30

    # If one side fills, cancel the other after this many seconds
    # if no hedge is available within delta_max
    emergency_cancel_seconds: int = 60


@dataclass
class CapitalConfig:
    """
    Capital allocation parameters (Section 4 of the paper).

    N_orders = B / (4 * m * p)
    where B = budget, m = min shares, p = mid price

    Kelly fraction: f* = (p * b - q) / b
    where p = prob of successful hedge, q = 1-p, b = net reward-to-risk ratio
    """
    # Total budget in cents (e.g., 100_00 = $100)
    total_budget_cents: int = 100_00

    # Fraction of total budget to deploy (Kelly or manual)
    deploy_fraction: float = 0.25

    # Number of contracts per order
    contracts_per_order: int = 1

    # Kelly criterion inputs
    hedge_success_probability: float = 0.85  # p
    net_reward_to_risk_ratio: float = 2.0    # b


@dataclass
class MarketSelectionConfig:
    """
    Market selection criteria (Section 5 / Table 1 of the paper).
    """
    # Price split: prefer markets near 50/50
    min_yes_price_cents: int = 25   # Minimum yes price (25c = 25%)
    max_yes_price_cents: int = 75   # Maximum yes price (75c = 75%)

    # Max spread: prefer wider reward zones
    min_spread_cents: int = 3       # Minimum bid-ask spread

    # Competition: prefer markets with fewer resting orders
    max_resting_orders: int = 50    # Skip markets with too many resting orders

    # Volatility: prefer stable markets
    max_price_change_1h_cents: int = 10  # Max 1h price movement

    # Minimum volume to ensure market is active
    min_daily_volume: int = 10


@dataclass
class BotConfig:
    """Top-level bot configuration."""
    kalshi: KalshiConfig = field(default_factory=KalshiConfig.from_env)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    hedge: HedgeConfig = field(default_factory=HedgeConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    market_selection: MarketSelectionConfig = field(default_factory=MarketSelectionConfig)

    # Main loop interval in seconds
    poll_interval_seconds: float = 5.0

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "bot.log"

    # Dry run mode: log orders but don't submit
    dry_run: bool = True

    # Tickers to monitor (empty = auto-select based on criteria)
    target_tickers: list = field(default_factory=list)
