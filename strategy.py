"""
Strategy engine implementing the hedged market-making framework.

Implements:
- Scoring function (Section 2)
- Hedging framework (Section 3)
- Capital allocation & Kelly criterion (Section 4)
- Market selection criteria (Section 5)
- Expected return calculation (Section 4.3)

v2 improvements:
- Orderbook-depth-aware pricing: adjusts aggressiveness based on queue position
- Book imbalance detection: avoids markets with informed one-sided flow
- Asymmetric pricing: bids more aggressively on the thinner side
- Better market scoring with book depth and imbalance factors
- Fill probability estimation integrated into order construction
- Queue-adjusted score: accounts for likelihood of actually getting filled
"""

import logging
import math
from dataclasses import dataclass, field
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

    # ── v2: depth and imbalance metrics ──

    @property
    def yes_total_depth(self) -> int:
        """Total quantity across all yes bid levels."""
        return sum(q for _, q in self.yes_bids)

    @property
    def no_total_depth(self) -> int:
        """Total quantity across all no bid levels."""
        return sum(q for _, q in self.no_bids)

    @property
    def total_depth(self) -> int:
        return self.yes_total_depth + self.no_total_depth

    @property
    def imbalance_ratio(self) -> float:
        """
        Book imbalance: 0.5 = perfectly balanced, 0.0 or 1.0 = completely one-sided.
        Defined as yes_depth / total_depth.
        High imbalance signals informed flow on one side.
        """
        total = self.total_depth
        if total == 0:
            return 0.5
        return self.yes_total_depth / total

    @property
    def is_imbalanced(self) -> bool:
        """True if book is significantly imbalanced (>70/30 or <30/70)."""
        ratio = self.imbalance_ratio
        return ratio < 0.3 or ratio > 0.7

    def depth_at_or_better(self, side: str, price_cents: int) -> int:
        """
        How many contracts are queued at or better than a given price?
        These are the contracts ahead of us in the queue.
        """
        bids = self.yes_bids if side == "yes" else self.no_bids
        total = 0
        for p, q in bids:
            if p >= price_cents:
                total += q
        return total

    def depth_within_range(self, side: str, best_price: int, range_cents: int) -> int:
        """How many contracts are within range_cents of the best bid."""
        bids = self.yes_bids if side == "yes" else self.no_bids
        total = 0
        for p, q in bids:
            if p >= best_price - range_cents:
                total += q
        return total


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
    estimated_fill_prob: float = 0.0  # v2: estimated probability both sides fill

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
        """
        return 100 - (self.yes_price_cents + self.no_price_cents)

    @property
    def edge_cents(self) -> int:
        """Alias for pnl_at_resolution — clearer name."""
        return self.pnl_at_resolution

    @property
    def expected_value_cents(self) -> float:
        """
        v2: Expected value accounting for fill probability.
        EV = fill_prob * edge - (1 - fill_prob) * max_loss_on_partial
        where max_loss ≈ max(yes_price, no_price) for a single unhedged leg
        """
        if self.estimated_fill_prob <= 0:
            return float('-inf')
        max_single_leg = max(self.yes_price_cents, self.no_price_cents)
        return (
            self.estimated_fill_prob * self.edge_cents
            - (1 - self.estimated_fill_prob) * max_single_leg * 0.5  # 50% chance wrong side
        )


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


# ── Fill Probability Estimation (v2) ─────────────────────────────────


def estimate_single_side_fill_prob(
    queue_depth_ahead: int,
    spread_cents: float,
    volume_24h: int,
) -> float:
    """
    Estimate probability that a single side fills within the hedge timeout.

    Uses volume-to-depth ratio instead of raw depth penalty.
    Deep books with high volume can still fill quickly; deep books
    with low volume cannot.

    Key insight: what matters is how fast the queue drains, which is
    proportional to volume / depth.

    Returns probability in [0.05, 0.95].
    """
    if queue_depth_ahead <= 0:
        # No queue — we're at the top of the book
        return min(0.90, 0.50 + volume_24h / 500.0)

    # Turnover ratio: how many times per day does the queue churn through?
    # If volume_24h = 10000 and depth = 5000, turnover = 2x/day
    # Over a 2-minute window (~1/720th of a day), that's decent fill prob
    turnover = volume_24h / max(queue_depth_ahead, 1)

    # Convert to fill probability over our ~2 min hedge window
    # turnover of 10+ means queue clears many times per day → high prob
    # turnover of 0.1 means queue barely moves → low prob
    # Using a logistic-style curve centered around turnover = 1
    fill_prob = 0.80 * (1.0 - math.exp(-0.5 * turnover))

    # Spread bonus: tighter spreads mean more crossing activity
    if spread_cents is not None and spread_cents > 0:
        fill_prob += min(0.10, 0.10 / spread_cents)

    return max(0.05, min(0.95, fill_prob))


def estimate_hedge_fill_prob(
    ob: OrderBookSnapshot,
    yes_price: int,
    no_price: int,
    volume_24h: int = 0,
) -> float:
    """
    Estimate probability that BOTH sides of a hedge fill.
    P(both) = P(yes_fill) * P(no_fill)
    """
    spread = ob.spread_cents or 5.0

    yes_ahead = ob.depth_at_or_better("yes", yes_price)
    no_ahead = ob.depth_at_or_better("no", no_price)

    yes_prob = estimate_single_side_fill_prob(yes_ahead, spread, volume_24h)
    no_prob = estimate_single_side_fill_prob(no_ahead, spread, volume_24h)

    # Joint probability (assuming independence — conservative for correlated books)
    return yes_prob * no_prob


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
    """
    if min_shares <= 0 or mid_price_cents <= 0:
        return 0
    return int(budget_cents / (4 * min_shares * mid_price_cents))


def compute_kelly_fraction(
    hedge_success_prob: float,
    reward_to_risk_ratio: float,
    alpha: float = 0.25,
) -> float:
    """
    Fractional Kelly criterion for optimal capital fraction.
    f* = (p * b - q) / b
    f  = α * f*,  α ∈ (0, 1]

    Paper: "NEVER full Kelly on 5min markets!" Use α = 0.25-0.5 to reduce variance.
    """
    p = hedge_success_prob
    q = 1.0 - p
    b = reward_to_risk_ratio

    if b <= 0:
        return 0.0

    f_star = (p * b - q) / b
    f_star = max(0.0, min(1.0, f_star))
    return f_star * alpha


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
    """
    denominator = num_shares * delta_max_cents
    if denominator <= 0:
        return float("inf")
    return expected_rewards_cents / denominator


# ── Risk Metrics (from Core Formulas reference) ──────────────────────


def compute_max_drawdown(pnl_history: list[int]) -> float:
    """
    MDD = (Peak - Trough) / Peak
    Block new trades if MDD > 8%.

    Args:
        pnl_history: list of cumulative P&L values in cents

    Returns:
        Max drawdown as a fraction (0.0 = no drawdown, 1.0 = total loss)
    """
    if not pnl_history or len(pnl_history) < 2:
        return 0.0

    peak = pnl_history[0]
    max_dd = 0.0

    for pnl in pnl_history:
        if pnl > peak:
            peak = pnl
        if peak > 0:
            dd = (peak - pnl) / peak
            max_dd = max(max_dd, dd)

    return max_dd


def compute_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """
    SR = (E[R] - Rf) / σ(R)
    Target SR > 2.0.

    Args:
        returns: list of per-trade returns (as fractions, e.g., 0.02 = 2%)
        risk_free_rate: risk-free rate per period

    Returns:
        Sharpe ratio. Returns 0.0 if insufficient data or zero variance.
    """
    if len(returns) < 2:
        return 0.0

    n = len(returns)
    mean_r = sum(returns) / n
    variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
    std_r = math.sqrt(variance) if variance > 0 else 0.0

    if std_r == 0:
        return 0.0

    return (mean_r - risk_free_rate) / std_r


def compute_profit_factor(gross_profit_cents: int, gross_loss_cents: int) -> float:
    """
    PF = gross_profit / gross_loss
    Healthy bot maintains PF > 1.5.

    Args:
        gross_profit_cents: total profit from winning trades
        gross_loss_cents: total loss from losing trades (positive number)

    Returns:
        Profit factor. Returns inf if no losses, 0.0 if no profits.
    """
    if gross_loss_cents <= 0:
        return float('inf') if gross_profit_cents > 0 else 0.0
    return gross_profit_cents / gross_loss_cents


def compute_var_95(pnl_history: list[int]) -> int:
    """
    VaR = μ - 1.645 · σ
    Max daily loss at 95% confidence.

    Args:
        pnl_history: list of per-trade P&L values in cents

    Returns:
        Value at Risk in cents (negative = expected worst loss).
    """
    if len(pnl_history) < 2:
        return 0

    n = len(pnl_history)
    mu = sum(pnl_history) / n
    variance = sum((x - mu) ** 2 for x in pnl_history) / (n - 1)
    sigma = math.sqrt(variance) if variance > 0 else 0.0

    return int(mu - 1.645 * sigma)


def check_exposure_limit(
    current_exposure_cents: int,
    new_bet_cents: int,
    max_exposure_cents: int,
) -> bool:
    """
    Pre-trade exposure check: exposure + bet <= max_exposure.

    Args:
        current_exposure_cents: total capital currently at risk
        new_bet_cents: cost of the proposed new trade
        max_exposure_cents: maximum allowed exposure

    Returns:
        True if the trade is within limits, False if it would breach.
    """
    return (current_exposure_cents + new_bet_cents) <= max_exposure_cents


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
    depth_score: float = 0.0  # v2: book depth (deeper = more reliable fills)
    balance_score: float = 0.0  # v2: book balance (balanced = safer)
    composite: float = 0.0

    def compute_composite(self, weights: Optional[dict] = None):
        """
        v2: added depth_score and balance_score to composite.
        Reweighted to emphasize fill reliability.
        """
        w = weights or {
            "price_split": 0.15,    # was 0.25 — less important than fill reliability
            "spread": 0.15,         # was 0.20
            "competition": 0.15,    # was 0.20
            "volatility": 0.10,     # was 0.20
            "volume": 0.15,         # was 0.15
            "depth": 0.15,          # v2: book depth matters for fill probability
            "balance": 0.15,        # v2: balanced books = fewer adverse partial fills
        }
        self.composite = (
            w["price_split"] * self.price_split_score
            + w["spread"] * self.spread_score
            + w["competition"] * self.competition_score
            + w["volatility"] * self.volatility_score
            + w["volume"] * self.volume_score
            + w["depth"] * self.depth_score
            + w["balance"] * self.balance_score
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

    v2 improvements:
    - Book depth scoring
    - Imbalance penalty
    - Minimum total depth filter
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

    # v2: minimum total book depth filter
    min_depth = getattr(config, 'min_total_book_depth', 0)
    if min_depth > 0 and orderbook.total_depth < min_depth:
        logger.debug(
            f"{market.ticker}: total depth {orderbook.total_depth} below minimum {min_depth}"
        )
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
        competition_score = 0.2
    else:
        competition_score = 1.0 - (total_orders / max(config.max_resting_orders, 1))

    # Volatility: placeholder (would need price history)
    # v2: use spread as a proxy — wider spread usually correlates with higher volatility
    if spread_val is not None and spread_val > 0:
        # Spread of 2-4 is ideal; very wide (10+) suggests high vol
        if spread_val <= 5:
            volatility_score = 0.8
        elif spread_val <= 8:
            volatility_score = 0.5
        else:
            volatility_score = 0.3
    else:
        volatility_score = 0.5

    # Volume score
    volume_score = min(1.0, market.volume_24h / 100.0)

    # v2: Book depth score — deeper books have more reliable fills
    # Normalize: 20+ contracts total depth is very good
    depth_score = min(1.0, orderbook.total_depth / 20.0)

    # v2: Book balance score — penalize imbalanced books
    # Imbalanced books signal informed flow → partial fills
    imbalance = orderbook.imbalance_ratio
    # Perfect balance (0.5) → score 1.0; extreme (0 or 1) → score 0.0
    balance_score = 1.0 - 2.0 * abs(imbalance - 0.5)

    ms = MarketScore(
        ticker=market.ticker,
        price_split_score=price_split_score,
        spread_score=spread_score,
        competition_score=competition_score,
        volatility_score=volatility_score,
        volume_score=volume_score,
        depth_score=depth_score,
        balance_score=balance_score,
    )
    ms.compute_composite()
    return ms


# ── Order Price Computation ──────────────────────────────────────────


def compute_optimal_prices(
    mid_price_cents: float,
    delta_max_cents: int,
    config: ScoringConfig,
    best_yes_ask: Optional[int] = None,
    best_no_ask: Optional[int] = None,
    orderbook: Optional[OrderBookSnapshot] = None,
    min_edge_cents: int = 1,
) -> tuple[int, int]:
    """
    Compute optimal YES and NO limit order prices.

    v2 improvements:
    - Orderbook-depth-aware: adjusts aggressiveness based on queue depth
    - Asymmetric pricing: bids more aggressively on the thinner side
    - Ensures positive edge (yes + no < 100) before returning

    Strategy: Place orders as close to mid as possible while ensuring:
    1. The combined cost stays < 100 (positive edge)
    2. Both prices are STRICTLY BELOW their respective asks (post_only safe)
    3. On imbalanced books, improve the price on the thinner side
    """
    # Target a combined cost that ensures min_edge_cents of guaranteed edge
    # e.g., min_edge=3 → target_combined=97 → 3¢ guaranteed edge
    target_combined = 100 - max(min_edge_cents, 1)
    yes_price = int(math.floor(mid_price_cents))
    no_price = target_combined - yes_price

    # ── v2: Asymmetric adjustment on imbalanced books ──
    # If one side has much more depth (harder to fill), shift price to
    # be more aggressive on the THINNER side (easier to fill).
    # This increases fill probability of the whole hedge at the cost of
    # slightly less edge.
    if orderbook is not None and orderbook.yes_bids and orderbook.no_bids:
        yes_depth = orderbook.yes_total_depth
        no_depth = orderbook.no_total_depth

        if yes_depth > 0 and no_depth > 0:
            depth_ratio = yes_depth / no_depth

            # If YES side is much thicker (more competition), improve NO price
            # to increase fill prob on the harder-to-fill YES leg
            if depth_ratio > 2.0 and no_price < 95:
                shift = min(2, int(math.log2(depth_ratio)))
                no_price += shift
                yes_price -= shift  # Maintain combined ≤ 99
                logger.debug(
                    f"  Asymmetric shift: YES depth={yes_depth} >> NO depth={no_depth}, "
                    f"improving NO by {shift}¢"
                )
            elif depth_ratio < 0.5 and yes_price < 95:
                shift = min(2, int(math.log2(1.0 / depth_ratio)))
                yes_price += shift
                no_price -= shift
                logger.debug(
                    f"  Asymmetric shift: NO depth={no_depth} >> YES depth={yes_depth}, "
                    f"improving YES by {shift}¢"
                )

    # Clamp prices below the opposing ask so post_only orders rest on the book
    if best_yes_ask is not None:
        yes_price = min(yes_price, best_yes_ask - 1)
        logger.debug(f"  YES price clamped to {yes_price}¢ (ask={best_yes_ask}¢)")
    if best_no_ask is not None:
        no_price = min(no_price, best_no_ask - 1)
        logger.debug(f"  NO price clamped to {no_price}¢ (ask={best_no_ask}¢)")

    # Validate bounds
    yes_price = max(1, min(99, yes_price))
    no_price = max(1, min(99, no_price))

    # Ensure positive edge: yes_price + no_price <= target_combined
    while yes_price + no_price > target_combined and (yes_price > 1 or no_price > 1):
        yes_dist = abs(mid_price_cents - yes_price)
        no_dist = abs((100 - mid_price_cents) - no_price)
        if yes_dist <= no_dist:
            no_price -= 1
        else:
            yes_price -= 1

    # Verify total is within acceptable overpayment range
    total = yes_price + no_price
    while total > 100 + delta_max_cents and (yes_price > 1 or no_price > 1):
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
    best_yes_ask: Optional[int] = None,
    best_no_ask: Optional[int] = None,
    orderbook: Optional[OrderBookSnapshot] = None,
    volume_24h: int = 0,
) -> Optional[HedgedOrder]:
    """
    Build a hedged order pair for a given market.

    v2: accepts full orderbook for depth-aware pricing and fill estimation.
    Falls back gracefully if orderbook is not provided (backward compatible).

    Returns None if the order would violate risk constraints.
    """
    yes_price, no_price = compute_optimal_prices(
        mid_price_cents,
        config.hedge.delta_max_cents,
        config.scoring,
        best_yes_ask=best_yes_ask,
        best_no_ask=best_no_ask,
        orderbook=orderbook,
        min_edge_cents=config.hedge.min_edge_cents,
    )

    # Check acceptable loss bound
    if not is_within_acceptable_loss(yes_price, no_price, config.hedge.delta_max_cents):
        logger.warning(
            f"{ticker}: prices {yes_price}+{no_price}={yes_price+no_price}c "
            f"exceeds 100+{config.hedge.delta_max_cents}c limit"
        )
        return None

    # v2: hard reject if no positive edge
    edge = 100 - (yes_price + no_price)
    if edge <= 0:
        logger.debug(f"{ticker}: no positive edge ({edge}¢), rejecting")
        return None

    # Compute score
    yes_distance = abs(mid_price_cents - yes_price)
    no_distance = abs((100 - mid_price_cents) - no_price)
    score = compute_total_score(yes_distance, no_distance, config.scoring)

    # Estimate fill probability and expected value if orderbook available
    fill_prob = 0.0
    if orderbook is not None:
        spread = orderbook.spread_cents or 5.0
        yes_ahead = orderbook.depth_at_or_better("yes", yes_price)
        no_ahead = orderbook.depth_at_or_better("no", no_price)

        p_yes = estimate_single_side_fill_prob(yes_ahead, spread, volume_24h)
        p_no = estimate_single_side_fill_prob(no_ahead, spread, volume_24h)
        fill_prob = p_yes * p_no

        # Correct EV: unhedged fills have positive expected value when bidding below mid
        # Buying YES below mid → expected profit = mid - yes_price
        # Buying NO below (100-mid) → expected profit = (100-mid) - no_price
        p_only_yes = p_yes * (1 - p_no)
        p_only_no = (1 - p_yes) * p_no
        unhedged_yes_ev = mid_price_cents - yes_price
        unhedged_no_ev = (100 - mid_price_cents) - no_price

        ev = (fill_prob * edge
              + p_only_yes * unhedged_yes_ev
              + p_only_no * unhedged_no_ev)

        if ev < 0:
            logger.debug(
                f"{ticker}: negative EV ({ev:.1f}¢) — "
                f"p_both={fill_prob:.2f}, p_yes={p_yes:.2f}, p_no={p_no:.2f}, "
                f"edge={edge}¢ — rejecting"
            )
            return None

    net_cost = (yes_price + no_price) - 100  # cents per contract

    return HedgedOrder(
        ticker=ticker,
        yes_price_cents=yes_price,
        no_price_cents=no_price,
        contracts=contracts,
        net_cost_cents=net_cost * contracts,
        score=score,
        estimated_fill_prob=fill_prob,
    )
