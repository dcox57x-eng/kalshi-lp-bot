"""
Strategy engine implementing the hedged market-making framework.

Implements:
- Scoring function (Section 2)
- Hedging framework (Section 3)
- Capital allocation & Kelly criterion (Section 4)
- Market selection criteria (Section 5)
- Expected return calculation (Section 4.3)
"""

import logging
import math
from dataclasses import dataclass
from typing import Optional

from config import BotConfig, CapitalConfig, HedgeConfig, MarketSelectionConfig, ScoringConfig

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────


@dataclass
class OrderBookSnapshot:
    """Parsed order book for a binary market."""
    ticker: str
    # Bids: list of (price_cents, quantity) sorted descending by price
    yes_bids: list[tuple[int, int]]
    no_bids: list[tuple[int, int]]

    @property
    def best_yes_bid(self) -> Optional[int]:
        return self.yes_bids[0][0] if self.yes_bids else None

    @property
    def best_no_bid(self) -> Optional[int]:
        return self.no_bids[0][0] if self.no_bids else None

    @property
    def best_yes_ask(self) -> Optional[int]:
        """Yes ask = 100 - best no bid."""
        return (100 - self.best_no_bid) if self.best_no_bid is not None else None

    @property
    def best_no_ask(self) -> Optional[int]:
        """No ask = 100 - best yes bid."""
        return (100 - self.best_yes_bid) if self.best_yes_bid is not None else None

    @property
    def mid_price_cents(self) -> Optional[float]:
        """
        Midpoint = (best_yes_bid + best_yes_ask) / 2
                 = (best_yes_bid + (100 - best_no_bid)) / 2
        """
        if self.best_yes_bid is not None and self.best_no_bid is not None:
            return (self.best_yes_bid + (100 - self.best_no_bid)) / 2.0
        return None

    @property
    def spread_cents(self) -> Optional[float]:
        """Bid-ask spread in cents."""
        if self.best_yes_bid is not None and self.best_yes_ask is not None:
            return self.best_yes_ask - self.best_yes_bid
        return None


@dataclass
class MarketInfo:
    """Market metadata relevant to the strategy."""
    ticker: str
    title: str
    status: str
    yes_price: Optional[int]  # Last yes price in cents
    no_price: Optional[int]   # Last no price in cents
    volume: int
    volume_24h: int
    open_interest: int
    close_time: Optional[str]


@dataclass
class HedgedOrder:
    """Represents a pair of YES + NO orders forming a hedge."""
    ticker: str
    yes_price_cents: int
    no_price_cents: int
    contracts: int
    net_cost_cents: int       # yes_price + no_price - 100 (negative = profit at resolution)
    score: float              # Expected reward score

    @property
    def is_risk_free(self) -> bool:
        """Definition 3.1: risk-free if p_Y + p_N < 100 cents."""
        return (self.yes_price_cents + self.no_price_cents) < 100

    @property
    def is_reward_neutral(self) -> bool:
        """Reward-neutral if p_Y + p_N = 100 cents."""
        return (self.yes_price_cents + self.no_price_cents) == 100

    @property
    def pnl_at_resolution(self) -> int:
        """
        Net P&L per contract at resolution (Section 3.1).
        π = 100 - (p_Y + p_N) cents
        One side resolves to 100, the other to 0.
        """
        return 100 - (self.yes_price_cents + self.no_price_cents)


# ── Scoring Function (Section 2.1) ──────────────────────────────────


def compute_score(distance_from_mid: float, config: ScoringConfig) -> float:
    """
    S(s) = ((v - s) / v)^2 * b

    Args:
        distance_from_mid: Distance of order from mid price (cents)
        config: Scoring configuration

    Returns:
        Score value. Orders at mid get maximum score b.
        Orders at distance v get score 0.
    """
    v = config.max_spread_v
    b = config.market_multiplier_b

    if distance_from_mid >= v:
        return 0.0

    s = max(0.0, distance_from_mid)
    return ((v - s) / v) ** 2 * b


def compute_total_score(
    yes_distance: float,
    no_distance: float,
    config: ScoringConfig,
) -> float:
    """
    Total score for a two-sided placement (YES + NO).
    Both sides contribute independently to total score.
    """
    return compute_score(yes_distance, config) + compute_score(no_distance, config)


# ── Hedging Framework (Section 3) ────────────────────────────────────


def compute_hedge_pnl(yes_price_cents: int, no_price_cents: int) -> int:
    """
    Net P&L of the hedged position at resolution (Equation 3).
    π = 100 - (p_Y + p_N)

    Positive = guaranteed profit per contract
    Zero = break-even
    Negative = guaranteed loss (overpayment) per contract
    """
    return 100 - (yes_price_cents + no_price_cents)


def is_within_acceptable_loss(
    yes_price_cents: int,
    no_price_cents: int,
    delta_max_cents: int,
) -> bool:
    """
    Definition 3.1: bounded loss if p_Y + p_N <= 100 + Δ_max.

    Args:
        yes_price_cents: Yes order price in cents
        no_price_cents: No order price in cents
        delta_max_cents: Maximum acceptable overpayment per share
    """
    return (yes_price_cents + no_price_cents) <= (100 + delta_max_cents)


def compute_delta_max(
    expected_daily_reward_cents: float,
    num_shares: int,
    horizon_days: int,
) -> float:
    """
    VaR-style maximum acceptable overpayment (Section 3.2).
    Δ_max = R / (n * T)

    Args:
        expected_daily_reward_cents: Expected daily reward in cents
        num_shares: Number of shares per order
        horizon_days: Time horizon in days
    """
    if num_shares <= 0 or horizon_days <= 0:
        return 0.0
    return expected_daily_reward_cents / (num_shares * horizon_days)


# ── Capital Allocation (Section 4) ───────────────────────────────────


def compute_max_orders(
    budget_cents: int,
    min_shares: int,
    mid_price_cents: float,
) -> int:
    """
    Maximum simultaneous orders per side (Equation 5).
    N_orders = B / (4 * m * p)

    The divisor of 4: both sides (YES+NO=2x) + safety buffer (2x).
    """
    if min_shares <= 0 or mid_price_cents <= 0:
        return 0
    return int(budget_cents / (4 * min_shares * mid_price_cents))


def compute_kelly_fraction(
    hedge_success_prob: float,
    reward_to_risk_ratio: float,
) -> float:
    """
    Kelly criterion for optimal capital fraction (Equation 6).
    f* = (p * b - q) / b

    Args:
        hedge_success_prob: p = probability that both sides fill within Δ_max
        reward_to_risk_ratio: b = net reward-to-risk ratio

    Returns:
        Optimal fraction of capital to deploy (0 to 1).
        Returns 0 if the edge is negative.
    """
    p = hedge_success_prob
    q = 1.0 - p
    b = reward_to_risk_ratio

    if b <= 0:
        return 0.0

    f_star = (p * b - q) / b
    return max(0.0, min(1.0, f_star))


# ── Expected Return (Section 4.3) ────────────────────────────────────


def compute_expected_return(
    expected_rewards_cents: float,
    expected_fill_loss_cents: float,
    capital_at_risk_cents: float,
) -> float:
    """
    Sharpe-like expected return (Equation 7).
    E[R] = (Rewards - L_fill) / C_risk
    """
    if capital_at_risk_cents <= 0:
        return 0.0
    return (expected_rewards_cents - expected_fill_loss_cents) / capital_at_risk_cents


def compute_breakeven_fill_rate(
    expected_rewards_cents: float,
    num_shares: int,
    delta_max_cents: float,
) -> float:
    """
    Break-even adverse fill rate (Equation 8).
    f_BE = Rewards / (n * Δ_max)

    If observed fill rate < f_BE, the strategy is profitable.
    """
    denominator = num_shares * delta_max_cents
    if denominator <= 0:
        return float("inf")
    return expected_rewards_cents / denominator


# ── Market Selection (Section 5, Table 1) ────────────────────────────


@dataclass
class MarketScore:
    """Composite score for market suitability."""
    ticker: str
    price_split_score: float  # How close to 50/50
    spread_score: float       # Wider = better for reward zone
    competition_score: float  # Fewer resting orders = better
    volatility_score: float   # Lower = better
    volume_score: float       # Higher = more active
    composite: float = 0.0

    def compute_composite(self, weights: dict = None):
        w = weights or {
            "price_split": 0.25,
            "spread": 0.20,
            "competition": 0.20,
            "volatility": 0.20,
            "volume": 0.15,
        }
        self.composite = (
            w["price_split"] * self.price_split_score
            + w["spread"] * self.spread_score
            + w["competition"] * self.competition_score
            + w["volatility"] * self.volatility_score
            + w["volume"] * self.volume_score
        )
        return self.composite


def score_market(
    market: MarketInfo,
    orderbook: OrderBookSnapshot,
    config: MarketSelectionConfig,
) -> Optional[MarketScore]:
    """
    Score a market based on Table 1 criteria.
    Returns None if the market fails any hard filter.
    """
    yes_price = market.yes_price
    if yes_price is None:
        return None

    # Hard filters
    if yes_price < config.min_yes_price_cents or yes_price > config.max_yes_price_cents:
        logger.debug(f"{market.ticker}: price {yes_price}c outside range")
        return None

    spread = orderbook.spread_cents
    if spread is not None and spread < config.min_spread_cents:
        logger.debug(f"{market.ticker}: spread {spread}c too narrow")
        return None

    if market.volume_24h < config.min_daily_volume:
        logger.debug(f"{market.ticker}: volume {market.volume_24h} too low")
        return None

    # Soft scores (0-1 scale)
    # Price split: perfect at 50, worst at edges
    price_split_score = 1.0 - abs(yes_price - 50) / 50.0

    # Spread: normalized (wider = better for our strategy up to a point)
    spread_val = spread if spread is not None else 0
    spread_score = min(1.0, spread_val / 10.0)

    # Competition: fewer orders = better (inverse)
    total_orders = sum(q for _, q in orderbook.yes_bids) + sum(q for _, q in orderbook.no_bids)
    if total_orders > config.max_resting_orders:
        competition_score = 0.2  # Penalize but don't fully exclude
    else:
        competition_score = 1.0 - (total_orders / max(config.max_resting_orders, 1))

    # Volatility: placeholder (would need price history)
    volatility_score = 0.7  # Default moderate

    # Volume score
    volume_score = min(1.0, market.volume_24h / 100.0)

    ms = MarketScore(
        ticker=market.ticker,
        price_split_score=price_split_score,
        spread_score=spread_score,
        competition_score=competition_score,
        volatility_score=volatility_score,
        volume_score=volume_score,
    )
    ms.compute_composite()
    return ms


# ── Order Price Computation ──────────────────────────────────────────


def compute_optimal_prices(
    mid_price_cents: float,
    delta_max_cents: int,
    config: ScoringConfig,
) -> tuple[int, int]:
    """
    Compute optimal YES and NO limit order prices.

    Strategy: Place orders as close to mid as possible while ensuring
    the combined cost stays within the acceptable loss bound.

    Goal: maximize score while keeping p_Y + p_N <= 100 + delta_max

    For YES: place a buy at mid (rounded down) or slightly below
    For NO: place a buy at (100 - mid) (rounded down) or slightly below

    The key insight is placing on BOTH sides with post_only to earn
    resting order rewards while bounding maximum loss.
    """
    # Ideal: both orders exactly at mid
    # YES buy at mid_price, NO buy at (100 - mid_price)
    # Combined = mid + (100 - mid) = 100 → reward-neutral

    yes_price = int(math.floor(mid_price_cents))
    no_price = 100 - yes_price  # This makes combined = 100 (reward-neutral)

    # If we want to be risk-free (combined < 100), shift each by 1 cent
    # This reduces score but guarantees profit at resolution
    if delta_max_cents < 0:
        # Force risk-free: each side backs off by 1 cent
        yes_price -= 1
        no_price -= 1

    # Validate bounds
    yes_price = max(1, min(99, yes_price))
    no_price = max(1, min(99, no_price))

    # Verify total is within acceptable range
    total = yes_price + no_price
    while total > 100 + delta_max_cents and (yes_price > 1 or no_price > 1):
        # Reduce the side further from its ideal
        if yes_price >= no_price:
            yes_price -= 1
        else:
            no_price -= 1
        total = yes_price + no_price

    return yes_price, no_price


def build_hedged_order(
    ticker: str,
    mid_price_cents: float,
    contracts: int,
    config: BotConfig,
) -> Optional[HedgedOrder]:
    """
    Build a hedged order pair for a given market.

    Returns None if the order would violate risk constraints.
    """
    yes_price, no_price = compute_optimal_prices(
        mid_price_cents,
        config.hedge.delta_max_cents,
        config.scoring,
    )

    # Check acceptable loss bound
    if not is_within_acceptable_loss(yes_price, no_price, config.hedge.delta_max_cents):
        logger.warning(
            f"{ticker}: prices {yes_price}+{no_price}={yes_price+no_price}c "
            f"exceeds 100+{config.hedge.delta_max_cents}c limit"
        )
        return None

    # Compute score
    yes_distance = abs(mid_price_cents - yes_price)
    no_distance = abs((100 - mid_price_cents) - no_price)
    score = compute_total_score(yes_distance, no_distance, config.scoring)

    net_cost = (yes_price + no_price) - 100  # cents per contract

    return HedgedOrder(
        ticker=ticker,
        yes_price_cents=yes_price,
        no_price_cents=no_price,
        contracts=contracts,
        net_cost_cents=net_cost * contracts,
        score=score,
    )
