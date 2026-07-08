"""
Improved bot loop: Continuous monitoring and hedge execution.

Key improvements over original:
1. Adaptive repricing — faster chase on partial fills, orderbook-aware pricing
2. True P&L tracking — includes unhedged losses, not just completed hedge gains
3. Pre-flight orderbook validation before placing orders
4. Tighter emergency cancel with dynamic timeout based on spread
5. Reduced API calls via fill-status caching
6. Partial fill quantity awareness — match fill counts, not just filled/unfilled
7. Skip markets where hedge cost (yes+no) >= 100¢
8. Exponential backoff on repricing instead of linear
"""

import logging
import signal
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import BotConfig
from kalshi_client import KalshiClient, KalshiAPIError
from strategy import (
    HedgedOrder,
    MarketInfo,
    MarketScore,
    OrderBookSnapshot,
    build_hedged_order,
    check_exposure_limit,
    compute_expected_return,
    compute_kelly_fraction,
    compute_max_drawdown,
    compute_max_orders,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_var_95,
    estimate_hedge_fill_prob,
    score_market,
)

logger = logging.getLogger(__name__)


# ── State tracking ───────────────────────────────────────────────────


@dataclass
class ActiveHedge:
    """Tracks a live hedged position."""
    ticker: str
    yes_order_id: Optional[str] = None
    no_order_id: Optional[str] = None
    yes_client_id: str = ""
    no_client_id: str = ""
    yes_price_cents: int = 0
    no_price_cents: int = 0
    contracts: int = 0
    yes_filled: bool = False
    no_filled: bool = False
    yes_fill_count: int = 0
    no_fill_count: int = 0
    created_at: float = 0.0
    last_checked: float = 0.0

    # ── Improvement: track repricing state explicitly ──
    reprice_count: int = 0
    last_reprice_price_cents: int = 0
    last_reprice_time: float = 0.0

    # ── Improvement: track the orderbook spread at placement time ──
    placement_spread_cents: int = 0

    # ── Rec #4: track original hedge score for fill-weighted reward ──
    score: float = 0.0

    @property
    def is_fully_hedged(self) -> bool:
        return self.yes_filled and self.no_filled

    @property
    def is_partially_filled(self) -> bool:
        return (self.yes_filled or self.no_filled) and not self.is_fully_hedged

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def hedge_cost_cents(self) -> int:
        """Total cost of both legs. Must be < 100 for positive expected value."""
        return self.yes_price_cents + self.no_price_cents

    @property
    def expected_pnl_per_contract(self) -> int:
        """P&L if both sides fill (guaranteed profit on resolution)."""
        return 100 - self.hedge_cost_cents


@dataclass
class BotStats:
    """Performance tracking with risk metrics from core formulas reference."""
    orders_placed: int = 0
    orders_cancelled: int = 0
    hedges_completed: int = 0
    hedges_failed: int = 0
    partial_fills: int = 0
    total_yes_fills: int = 0
    total_no_fills: int = 0
    total_reward_score: float = 0.0
    total_pnl_cents: int = 0            # Completed hedge profit
    total_overpayment_cents: int = 0     # Loss from unhedged fills
    total_contracts_hedged: int = 0      # Contracts successfully hedged
    total_contracts_exposed: int = 0     # Contracts left unhedged
    markets_skipped_thin_book: int = 0   # Tracking skip reasons
    markets_skipped_no_edge: int = 0
    reprices_attempted: int = 0
    reprices_succeeded: int = 0          # Reprice that led to a fill
    started_at: float = field(default_factory=time.time)

    # Risk metrics tracking
    pnl_history: list = field(default_factory=list)        # Per-trade P&L in cents
    cumulative_pnl_history: list = field(default_factory=list)  # Running cumulative P&L
    peak_pnl_cents: int = 0                                # High-water mark
    gross_profit_cents: int = 0                            # Sum of winning trades
    gross_loss_cents: int = 0                              # Sum of losing trades (positive)
    risk_halted: bool = False                              # Whether risk halt is active
    risk_halt_time: float = 0.0                            # When risk halt was triggered
    risk_halt_reason: str = ""                             # Why halted

    @property
    def net_pnl_cents(self) -> int:
        """True P&L: hedge profits minus unhedged exposure losses."""
        return self.total_pnl_cents - self.total_overpayment_cents

    @property
    def hedge_success_rate(self) -> float:
        total = self.hedges_completed + self.hedges_failed
        return self.hedges_completed / total if total > 0 else 0.0

    @property
    def current_exposure_cents(self) -> int:
        """Capital currently at risk in active hedges."""
        return self.total_contracts_exposed * 50  # Conservative avg price estimate

    @property
    def max_drawdown(self) -> float:
        """MDD = (Peak - Trough) / Peak. Block new trades if MDD > 8%."""
        return compute_max_drawdown(self.cumulative_pnl_history)

    @property
    def sharpe_ratio(self) -> float:
        """SR = (E[R] - Rf) / σ(R). Target SR > 2.0."""
        if not self.pnl_history:
            return 0.0
        # Convert cents to return fractions (relative to avg trade cost ~50¢)
        returns = [pnl / 50.0 for pnl in self.pnl_history]
        return compute_sharpe_ratio(returns)

    @property
    def profit_factor(self) -> float:
        """PF = gross_profit / gross_loss. Healthy bot PF > 1.5."""
        return compute_profit_factor(self.gross_profit_cents, self.gross_loss_cents)

    @property
    def var_95_cents(self) -> int:
        """VaR = μ - 1.645 · σ. Max expected loss at 95% confidence."""
        return compute_var_95(self.pnl_history)

    def record_trade_pnl(self, pnl_cents: int):
        """Record a completed trade's P&L for risk metric calculations."""
        self.pnl_history.append(pnl_cents)
        cum_pnl = self.net_pnl_cents
        self.cumulative_pnl_history.append(cum_pnl)
        if cum_pnl > self.peak_pnl_cents:
            self.peak_pnl_cents = cum_pnl
        if pnl_cents > 0:
            self.gross_profit_cents += pnl_cents
        else:
            self.gross_loss_cents += abs(pnl_cents)

    def summary(self) -> str:
        runtime = time.time() - self.started_at
        hours = runtime / 3600
        mdd_pct = self.max_drawdown * 100
        sr = self.sharpe_ratio
        pf = self.profit_factor
        var = self.var_95_cents
        return (
            f"\n{'='*60}\n"
            f"  Bot Performance Summary\n"
            f"{'='*60}\n"
            f"  Runtime:            {hours:.1f} hours\n"
            f"  Orders placed:      {self.orders_placed}\n"
            f"  Orders cancelled:   {self.orders_cancelled}\n"
            f"  Hedges completed:   {self.hedges_completed}\n"
            f"  Hedges failed:      {self.hedges_failed}\n"
            f"  Hedge success rate: {self.hedge_success_rate:.1%}\n"
            f"  Partial fills:      {self.partial_fills}\n"
            f"  Contracts hedged:   {self.total_contracts_hedged}\n"
            f"  Contracts exposed:  {self.total_contracts_exposed}\n"
            f"  Total reward score: {self.total_reward_score:.2f}\n"
            f"  Hedge P&L:          {self.total_pnl_cents}¢ (${self.total_pnl_cents/100:.2f})\n"
            f"  Unhedged losses:    {self.total_overpayment_cents}¢ (${self.total_overpayment_cents/100:.2f})\n"
            f"  ── Net P&L:         {self.net_pnl_cents}¢ (${self.net_pnl_cents/100:.2f}) ──\n"
            f"  Reprices attempted: {self.reprices_attempted}\n"
            f"  Skipped (thin):     {self.markets_skipped_thin_book}\n"
            f"  Skipped (no edge):  {self.markets_skipped_no_edge}\n"
            f"{'─'*60}\n"
            f"  RISK METRICS\n"
            f"{'─'*60}\n"
            f"  Max Drawdown:       {mdd_pct:.1f}% (limit: 8%)\n"
            f"  Sharpe Ratio:       {sr:.2f} (target: >2.0)\n"
            f"  Profit Factor:      {pf:.2f} (healthy: >1.5)\n"
            f"  VaR (95%):          {var}¢ (${var/100:.2f})\n"
            f"  Gross Profit:       {self.gross_profit_cents}¢\n"
            f"  Gross Loss:         {self.gross_loss_cents}¢\n"
            f"  Risk Halted:        {'YES — ' + self.risk_halt_reason if self.risk_halted else 'No'}\n"
            f"{'='*60}\n"
        )


# ── Orderbook parsing ────────────────────────────────────────────────


def _parse_bid_list(entries: list) -> list[tuple[int, int]]:
    """
    Parse a list of orderbook entries into (price_cents, quantity) tuples.

    Handles multiple Kalshi response formats:
      - [[price_cents, qty], ...]            — integer cents (legacy)
      - [[price_dollar_float, qty], ...]     — dollar floats like 0.29
      - [["0.2900", "43.00"], ...]           — fixed-point string dollars (orderbook_fp)
      - [[price], ...]                       — price only, assume qty=1
    """
    bids = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 1:
            continue

        # Count may be an int, float, or fixed-point string like "43.00".
        # float() first so "43.00" parses; then round to whole contracts.
        try:
            qty = int(round(float(entry[1]))) if len(entry) >= 2 else 1
        except (ValueError, TypeError):
            qty = 1

        # Price may be integer cents (29), a dollar float (0.29), or a
        # fixed-point string ("0.2900"). float() first, then normalize.
        try:
            price_val = float(entry[0])
        except (ValueError, TypeError):
            continue

        if price_val < 1.0:                       # dollars → cents
            price_cents = int(round(price_val * 100))
        else:                                      # already in cents
            price_cents = int(round(price_val))

        if 1 <= price_cents <= 99:
            bids.append((price_cents, qty))
    return bids


def parse_orderbook(ticker: str, raw: dict) -> OrderBookSnapshot:
    """Parse Kalshi orderbook response into our data structure."""
    # Kalshi wraps the book in "orderbook_fp" (current fixed-point format),
    # or "orderbook" (legacy). Fall back to the raw dict if neither is present.
    ob = raw.get("orderbook_fp") or raw.get("orderbook") or raw

    yes_bids = []
    no_bids = []

    logger.debug(f"{ticker}: orderbook keys={list(ob.keys())}")

    yes_raw = ob.get("yes") or ob.get("yes_dollars") or ob.get("bids") or []
    if yes_raw:
        yes_bids = _parse_bid_list(yes_raw)
        logger.debug(f"{ticker}: parsed {len(yes_bids)} yes bids from {len(yes_raw)} raw entries")

    no_raw = ob.get("no") or ob.get("no_dollars") or ob.get("asks") or []
    if no_raw:
        no_bids = _parse_bid_list(no_raw)
        logger.debug(f"{ticker}: parsed {len(no_bids)} no bids from {len(no_raw)} raw entries")

    if not yes_bids and not no_bids:
        import json
        logger.warning(
            f"{ticker}: EMPTY ORDERBOOK — raw response:\n"
            f"{json.dumps(raw, indent=2, default=str)[:2000]}"
        )

    yes_bids.sort(key=lambda x: x[0], reverse=True)
    no_bids.sort(key=lambda x: x[0], reverse=True)

    return OrderBookSnapshot(ticker=ticker, yes_bids=yes_bids, no_bids=no_bids)


def parse_market(raw: dict) -> MarketInfo:
    """Parse Kalshi market response."""
    m = raw.get("market", raw)  # Handle both wrapped and unwrapped
    return MarketInfo(
        ticker=m.get("ticker", ""),
        title=m.get("title", ""),
        status=m.get("status", ""),
        yes_price=m.get("yes_bid") or m.get("last_price"),
        no_price=m.get("no_bid"),
        volume=m.get("volume", 0),
        volume_24h=m.get("volume_24h", 0),
        open_interest=m.get("open_interest", 0),
        close_time=m.get("close_time"),
    )


# ── Main Bot ─────────────────────────────────────────────────────────


class LiquidityBot:
    """
    Hedged liquidity provision bot for Kalshi.

    Improvements:
    1. Pre-flight validation: skip orders where yes+no >= 100
    2. Orderbook-aware pricing: factor in queue depth
    3. Faster repricing: exponential chase instead of linear
    4. Dynamic stale timeout: based on spread at placement
    5. True net P&L: subtract unhedged losses from reported P&L
    6. Fill-count matching: handle partial contract fills correctly
    7. Cooldown after failures: back off from markets that keep failing
    """

    def __init__(self, config: BotConfig, client: KalshiClient):
        self.config = config
        self.client = client
        self.stats = BotStats()
        self.active_hedges: dict[str, ActiveHedge] = {}  # ticker -> ActiveHedge
        self.market_scores: dict[str, MarketScore] = {}
        self._running = False
        self._setup_signal_handlers()

        # ── Improvement: per-market failure cooldowns ──
        # Maps ticker -> timestamp when cooldown expires
        self._market_cooldowns: dict[str, float] = {}
        self._cooldown_seconds = 120  # Back off for 2 min after a failed hedge

        # ── Risk guard: consecutive loss halt (2 per images spec) ──
        self._consecutive_failures = 0
        self._max_consecutive_before_pause = config.risk.max_consecutive_losses

        # ── Rec #3: track last observed top-of-book per market ──
        self._last_spread: dict[str, tuple] = {}

        # ── Rec #5: track thin-market skip counts and backoff ──
        self._thin_market_skips: dict[str, int] = {}
        self._thin_market_backoff: dict[str, float] = {}

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received, cleaning up...")
        self._running = False

    def _is_cooled_down(self, ticker: str) -> bool:
        """Check if a market is still in cooldown after a failed hedge."""
        expires = self._market_cooldowns.get(ticker, 0)
        return time.time() < expires

    def _set_cooldown(self, ticker: str):
        """Put a market on cooldown after a failure."""
        self._market_cooldowns[ticker] = time.time() + self._cooldown_seconds
        logger.debug(f"{ticker}: cooling down for {self._cooldown_seconds}s")

    # ── Rec #5: thin-market backoff ──

    def _record_thin_skip(self, ticker: str):
        """Track consecutive thin-market skips and apply exponential backoff."""
        self._thin_market_skips[ticker] = self._thin_market_skips.get(ticker, 0) + 1
        skips = self._thin_market_skips[ticker]
        if skips >= 3:
            backoff = min(60, 5 * 2 ** (skips - 3))
            self._thin_market_backoff[ticker] = time.time() + backoff
            logger.debug(f"{ticker}: thin market backoff {backoff}s after {skips} consecutive skips")

    def _reset_thin_tracking(self, ticker: str):
        """Reset thin-market tracking when a market becomes tradeable."""
        self._thin_market_skips.pop(ticker, None)
        self._thin_market_backoff.pop(ticker, None)

    # ── Market Discovery ─────────────────────────────────────────

    def discover_markets(self) -> list[tuple[MarketInfo, MarketScore]]:
        """
        Scan markets across all target series and rank by suitability.
        Returns list of (market, score) tuples sorted by composite score.
        """
        series_list = self.config.target_series
        logger.info(f"Scanning markets across series: {', '.join(series_list)}")
        scored_markets = []
        total_scanned = 0

        for series in series_list:
            try:
                result = self.client.get_markets(
                    limit=100, status="open", series_ticker=series
                )
                markets = result.get("markets", [])
                logger.debug(f"  {series}: found {len(markets)} open markets")
            except KalshiAPIError as e:
                logger.warning(f"  {series}: failed to fetch markets: {e}")
                continue

            for raw_market in markets:
                total_scanned += 1
                market = parse_market(raw_market)
                if market.status != "active":
                    continue

                try:
                    ob_raw = self.client.get_orderbook(market.ticker, depth=10)
                    ob = parse_orderbook(market.ticker, ob_raw)
                except KalshiAPIError:
                    continue

                # Skip one-sided or thin books during discovery
                min_bids = self.config.market_selection.min_bids_each_side
                if len(ob.yes_bids) < min_bids or len(ob.no_bids) < min_bids:
                    self.stats.markets_skipped_thin_book += 1
                    logger.debug(
                        f"  {market.ticker}: skipping — thin/one-sided book "
                        f"(yes={len(ob.yes_bids)}, no={len(ob.no_bids)})"
                    )
                    continue

                # ── Improvement: pre-flight edge check during discovery ──
                if ob.best_yes_bid and ob.best_no_bid:
                    # If even the best prices don't yield an edge, skip
                    if ob.best_yes_bid + ob.best_no_bid >= 100:
                        self.stats.markets_skipped_no_edge += 1
                        logger.debug(
                            f"  {market.ticker}: skipping — no edge "
                            f"(best_yes={ob.best_yes_bid} + best_no={ob.best_no_bid} "
                            f"= {ob.best_yes_bid + ob.best_no_bid} >= 100)"
                        )
                        continue

                score = score_market(market, ob, self.config.market_selection)
                if score is not None:
                    scored_markets.append((market, score))
                    self.market_scores[market.ticker] = score

        # Sort by composite score descending
        scored_markets.sort(key=lambda x: x[1].composite, reverse=True)

        # Concentrate on top N markets only
        max_targets = self.config.market_selection.max_target_markets
        if len(scored_markets) > max_targets:
            logger.info(
                f"Concentrating on top {max_targets} of {len(scored_markets)} eligible markets"
            )
            scored_markets = scored_markets[:max_targets]

        logger.info(
            f"Found {len(scored_markets)} target markets out of {total_scanned} scanned "
            f"across {len(series_list)} series"
        )
        for market, score in scored_markets[:5]:
            logger.info(
                f"  {market.ticker}: score={score.composite:.3f} "
                f"price={market.yes_price}¢ vol24h={market.volume_24h}"
            )

        return scored_markets

    # ── Order Placement ──────────────────────────────────────────

    def place_hedged_orders(
        self, ticker: str, mid_price: float, orderbook: OrderBookSnapshot,
        volume_24h: int = 0,
    ) -> Optional[ActiveHedge]:
        """
        Place a hedged YES + NO order pair on a market.

        Improvements:
        - Pre-flight validation: reject if yes+no >= 100
        - Orderbook-aware: check queue depth before committing
        - Record spread at placement for dynamic timeout
        """
        contracts = self.config.capital.contracts_per_order
        # v2: pass full orderbook for depth-aware pricing + fill estimation
        hedged = build_hedged_order(
            ticker, mid_price, contracts, self.config,
            best_yes_ask=orderbook.best_yes_ask,
            best_no_ask=orderbook.best_no_ask,
            orderbook=orderbook,
            volume_24h=volume_24h,
        )

        if hedged is None:
            logger.warning(f"{ticker}: could not build valid hedged order")
            return None

        # ── Improvement: hard reject if no positive edge ──
        total_cost = hedged.yes_price_cents + hedged.no_price_cents
        if total_cost >= 100:
            logger.warning(
                f"{ticker}: REJECTING hedge — no edge! "
                f"YES@{hedged.yes_price_cents} + NO@{hedged.no_price_cents} = {total_cost}¢ >= 100¢"
            )
            self.stats.markets_skipped_no_edge += 1
            return None

        edge_cents = 100 - total_cost

        # ── Reject if edge is below the configured minimum ──
        if edge_cents < self.config.hedge.min_edge_cents:
            logger.info(
                f"{ticker}: edge too thin ({edge_cents}¢ < min_edge={self.config.hedge.min_edge_cents}¢), skipping"
            )
            self.stats.markets_skipped_no_edge += 1
            return None

        # ── Improvement: estimate fill probability ──
        fill_prob = estimate_hedge_fill_prob(
            orderbook, hedged.yes_price_cents, hedged.no_price_cents
        )
        if fill_prob < 0.15:
            # Rec #5: track consecutive thin skips for backoff
            self._record_thin_skip(ticker)
            logger.debug(
                f"{ticker}: low fill probability ({fill_prob:.2f}), skipping"
            )
            return None

        logger.info(
            f"{ticker}: placing hedge YES@{hedged.yes_price_cents}¢ + "
            f"NO@{hedged.no_price_cents}¢ = {total_cost}¢ "
            f"(edge={edge_cents}¢, fill_prob={fill_prob:.2f}, "
            f"{hedged.contracts} contracts, score={hedged.score:.3f})"
        )

        if self.config.dry_run:
            logger.info(f"  [DRY RUN] Would place orders on {ticker}")
            ah = ActiveHedge(
                ticker=ticker,
                yes_client_id=str(uuid.uuid4()),
                no_client_id=str(uuid.uuid4()),
                yes_price_cents=hedged.yes_price_cents,
                no_price_cents=hedged.no_price_cents,
                contracts=contracts,
                created_at=time.time(),
                placement_spread_cents=int(orderbook.spread_cents or 0),
                score=hedged.score,
            )
            self.stats.orders_placed += 2
            return ah

        yes_cid = str(uuid.uuid4())
        no_cid = str(uuid.uuid4())

        # Try batch order first (preferred — atomic placement)
        try:
            orders = [
                {
                    "ticker": ticker,
                    "side": "yes",
                    "action": "buy",
                    "count": contracts,
                    "yes_price": hedged.yes_price_cents,
                    "client_order_id": yes_cid,
                    "post_only": True,
                    "type": "limit",
                },
                {
                    "ticker": ticker,
                    "side": "no",
                    "action": "buy",
                    "count": contracts,
                    "yes_price": 100 - hedged.no_price_cents,
                    "client_order_id": no_cid,
                    "post_only": True,
                    "type": "limit",
                },
            ]
            result = self.client.batch_create_orders(orders)
            logger.debug(f"{ticker}: batch response: {result}")
            order_results = result.get("orders", [])

            ah = ActiveHedge(
                ticker=ticker,
                yes_client_id=yes_cid,
                no_client_id=no_cid,
                yes_price_cents=hedged.yes_price_cents,
                no_price_cents=hedged.no_price_cents,
                contracts=contracts,
                created_at=time.time(),
                placement_spread_cents=int(orderbook.spread_cents or 0),
                score=hedged.score,
            )

            for order_res in order_results:
                if not isinstance(order_res, dict):
                    logger.warning(f"{ticker}: unexpected batch entry type: {type(order_res)}")
                    continue
                order = order_res.get("order")
                if order is None:
                    err = order_res.get("error", order_res)
                    logger.warning(f"{ticker}: batch entry returned no order: {err}")
                    continue
                oid = order.get("order_id", "")
                cid = order.get("client_order_id", "")
                if cid == yes_cid:
                    ah.yes_order_id = oid
                elif cid == no_cid:
                    ah.no_order_id = oid

            # Validate BOTH sides placed
            if not ah.yes_order_id or not ah.no_order_id:
                logger.error(
                    f"{ticker}: batch order incomplete — "
                    f"YES={ah.yes_order_id} NO={ah.no_order_id}, cancelling surviving side"
                )
                for oid in [ah.yes_order_id, ah.no_order_id]:
                    if oid:
                        try:
                            self.client.cancel_order(oid)
                            self.stats.orders_cancelled += 1
                        except KalshiAPIError as e:
                            logger.warning(f"  Failed to cancel {oid}: {e}")
                return None

            self.stats.orders_placed += 2
            # Rec #4: score deferred to fill time (see manage_hedge_lifecycle)
            # Rec #5: reset thin-market tracking on successful placement
            self._reset_thin_tracking(ticker)
            logger.info(f"  ✓ Batch orders placed: YES={ah.yes_order_id} NO={ah.no_order_id}")
            return ah

        except KalshiAPIError as e:
            logger.warning(f"Batch order failed ({e}), trying sequential...")

        # Sequential fallback
        ah = ActiveHedge(
            ticker=ticker,
            yes_client_id=yes_cid,
            no_client_id=no_cid,
            yes_price_cents=hedged.yes_price_cents,
            no_price_cents=hedged.no_price_cents,
            contracts=contracts,
            created_at=time.time(),
            placement_spread_cents=int(orderbook.spread_cents or 0),
            score=hedged.score,
        )

        try:
            yes_result = self.client.create_order(
                ticker=ticker,
                side="yes",
                action="buy",
                count=contracts,
                yes_price=hedged.yes_price_cents,
                client_order_id=yes_cid,
                post_only=True,
            )
            ah.yes_order_id = yes_result.get("order", {}).get("order_id")
            self.stats.orders_placed += 1
        except KalshiAPIError as e:
            logger.error(f"  ✗ YES order failed: {e}")
            return None

        try:
            no_result = self.client.create_order(
                ticker=ticker,
                side="no",
                action="buy",
                count=contracts,
                yes_price=100 - hedged.no_price_cents,
                client_order_id=no_cid,
                post_only=True,
            )
            ah.no_order_id = no_result.get("order", {}).get("order_id")
            self.stats.orders_placed += 1
        except KalshiAPIError as e:
            logger.error(f"  ✗ NO order failed: {e}")
            if ah.yes_order_id:
                try:
                    self.client.cancel_order(ah.yes_order_id)
                    self.stats.orders_cancelled += 1
                except KalshiAPIError:
                    pass
            return None

        # Rec #4: score deferred to fill time (see manage_hedge_lifecycle)
        # Rec #5: reset thin-market tracking on successful placement
        self._reset_thin_tracking(ticker)
        logger.info(f"  ✓ Sequential orders placed: YES={ah.yes_order_id} NO={ah.no_order_id}")
        return ah

    # ── Fill Monitoring ──────────────────────────────────────────

    def check_fills(self, hedge: ActiveHedge) -> ActiveHedge:
        """Check if orders in a hedge have been filled."""
        if self.config.dry_run:
            return hedge

        # ── Improvement: don't re-check already-filled sides ──
        sides_to_check = []
        if not hedge.yes_filled and hedge.yes_order_id:
            sides_to_check.append((hedge.yes_order_id, "yes"))
        if not hedge.no_filled and hedge.no_order_id:
            sides_to_check.append((hedge.no_order_id, "no"))

        for order_id, side in sides_to_check:
            try:
                result = self.client.get_order(order_id)
                order = result.get("order", {})
                status = order.get("status", "")
                remaining = order.get("remaining_count", 0)
                fill_count = order.get("fill_count", 0)

                if side == "yes":
                    hedge.yes_fill_count = fill_count
                    # Rec #1: require fill_count > 0 — prevents marking cancelled/expired
                    # orders (remaining=0, fill_count=0) as "filled"
                    if fill_count > 0 and (status == "executed" or remaining == 0):
                        hedge.yes_filled = True
                    elif status in ("cancelled", "expired") and fill_count == 0:
                        logger.debug(f"  {hedge.ticker}: YES order {status} with 0 fills")
                else:
                    hedge.no_fill_count = fill_count
                    if fill_count > 0 and (status == "executed" or remaining == 0):
                        hedge.no_filled = True
                    elif status in ("cancelled", "expired") and fill_count == 0:
                        logger.debug(f"  {hedge.ticker}: NO order {status} with 0 fills")

            except KalshiAPIError as e:
                logger.warning(f"Failed to check order {order_id}: {e}")

        hedge.last_checked = time.time()
        return hedge

    def manage_hedge_lifecycle(self, hedge: ActiveHedge) -> bool:
        """
        Manage a hedge through its lifecycle.

        Returns True if the hedge should be kept, False if it should be removed.

        Improvements:
        - Dynamic emergency timeout based on spread at placement
        - Exponential reprice chase
        - Proper P&L accounting for partial fills
        - Cooldown on failure
        """
        hedge = self.check_fills(hedge)

        # Case: both sides filled → complete hedge
        if hedge.is_fully_hedged:
            matched_contracts = min(hedge.yes_fill_count, hedge.no_fill_count)

            # Rec #2: defensive guard — cooldown if somehow completed with 0 fills
            if matched_contracts == 0:
                self.stats.hedges_completed += 1
                self._set_cooldown(hedge.ticker)
                logger.info(
                    f"  ⚠ Hedge on {hedge.ticker} completed with 0 fills, adding cooldown"
                )
                return False

            pnl_per = 100 - (hedge.yes_price_cents + hedge.no_price_cents)
            pnl_total = pnl_per * matched_contracts
            self.stats.hedges_completed += 1
            self.stats.total_pnl_cents += pnl_total
            self.stats.total_contracts_hedged += matched_contracts
            self.stats.total_yes_fills += hedge.yes_fill_count
            self.stats.total_no_fills += hedge.no_fill_count
            self._consecutive_failures = 0  # Reset failure streak

            # Record for risk metrics (MDD, Sharpe, Profit Factor, VaR)
            self.stats.record_trade_pnl(pnl_total)

            # Rec #4: fill-weighted score — only count score for filled contracts
            if hedge.contracts > 0:
                fill_ratio = matched_contracts / hedge.contracts
                self.stats.total_reward_score += hedge.score * fill_ratio

            logger.info(
                f"  ✓ Hedge complete on {hedge.ticker}: "
                f"{matched_contracts} contracts × {pnl_per}¢ = {pnl_total}¢ P&L"
            )
            return False  # Remove from active

        # Case: one side filled (adverse fill!) → hedge urgently
        if hedge.is_partially_filled:
            self.stats.partial_fills += 1
            filled_side = "YES" if hedge.yes_filled else "NO"
            unfilled_id = hedge.no_order_id if hedge.yes_filled else hedge.yes_order_id
            elapsed = time.time() - hedge.created_at

            # ── Improvement: dynamic emergency timeout ──
            # Wider spreads at placement → more time needed; tighter → less
            base_timeout = self.config.hedge.emergency_cancel_seconds
            spread_factor = max(1.0, (hedge.placement_spread_cents or 3) / 3.0)
            dynamic_timeout = min(base_timeout * spread_factor, base_timeout * 2)

            logger.warning(
                f"  ⚠ Partial fill on {hedge.ticker}: {filled_side} filled, "
                f"waiting for other side ({elapsed:.0f}s / {dynamic_timeout:.0f}s timeout)"
            )

            # Phase 1: Brief grace period
            if elapsed < self.config.hedge.reprice_after_seconds:
                return True

            # Phase 2: Aggressive reprice — exponential chase
            if elapsed < dynamic_timeout:
                self._reprice_unfilled_leg(hedge)
                return True

            # Phase 3: Emergency cancel
            logger.error(
                f"  ✗ Emergency cancel on {hedge.ticker}: "
                f"unfilled side after {elapsed:.0f}s"
            )
            if unfilled_id and not self.config.dry_run:
                try:
                    self.client.cancel_order(unfilled_id)
                    self.stats.orders_cancelled += 1
                except KalshiAPIError:
                    pass

            self.stats.hedges_failed += 1
            self._consecutive_failures += 1

            # Precise loss calculation
            if hedge.yes_filled:
                loss = hedge.yes_price_cents * hedge.yes_fill_count
            else:
                loss = hedge.no_price_cents * hedge.no_fill_count
            self.stats.total_overpayment_cents += loss
            self.stats.total_contracts_exposed += max(hedge.yes_fill_count, hedge.no_fill_count)

            # Record loss for risk metrics (MDD, Sharpe, PF, VaR)
            self.stats.record_trade_pnl(-loss)

            # Cooldown this market
            self._set_cooldown(hedge.ticker)

            return False

        # Case: neither filled, check for staleness
        # ── Improvement: dynamic stale timeout ──
        stale_timeout = 120  # base 2 minutes
        if hedge.placement_spread_cents and hedge.placement_spread_cents > 5:
            # Wider spread markets are slower — give more time
            stale_timeout = min(180, stale_timeout + hedge.placement_spread_cents * 5)

        if hedge.age_seconds > stale_timeout:
            logger.info(
                f"  Cancelling stale hedge on {hedge.ticker} "
                f"(no fills after {hedge.age_seconds:.0f}s)"
            )
            for order_id in [hedge.yes_order_id, hedge.no_order_id]:
                if order_id and not self.config.dry_run:
                    try:
                        self.client.cancel_order(order_id)
                        self.stats.orders_cancelled += 1
                    except KalshiAPIError:
                        pass
            # Rec #2: cooldown after stale timeout — same orders would likely stale again
            self._set_cooldown(hedge.ticker)
            return False

        return True  # Keep monitoring

    # ── Aggressive Repricing ─────────────────────────────────────

    def _reprice_unfilled_leg(self, hedge: ActiveHedge):
        """
        Improve the price on the unfilled leg to chase a fill.

        Improvement: exponential price improvement instead of linear.
        Step sizes: 1, 2, 4, 6, 8... cents from original.
        This front-loads the chase to maximize fill probability.
        """
        if self.config.dry_run:
            return

        if hedge.yes_filled and not hedge.no_filled:
            unfilled_side = "no"
            unfilled_id = hedge.no_order_id
            original_price = hedge.no_price_cents
        elif hedge.no_filled and not hedge.yes_filled:
            unfilled_side = "yes"
            unfilled_id = hedge.yes_order_id
            original_price = hedge.yes_price_cents
        else:
            return

        if not unfilled_id:
            return

        max_chase = self.config.hedge.reprice_max_chase_cents

        # ── Improvement: exponential chase schedule ──
        # Each reprice attempt doubles the step (1, 2, 4, 8...)
        # but capped at max_chase
        hedge.reprice_count += 1
        improvement = min(2 ** (hedge.reprice_count - 1), max_chase)

        new_price = original_price + improvement
        new_price = min(new_price, 95)  # Never bid above 95¢

        # ── Improvement: check if new hedge cost still has edge ──
        if unfilled_side == "yes":
            new_total = new_price + hedge.no_price_cents
        else:
            new_total = hedge.yes_price_cents + new_price
        if new_total >= 100:
            logger.warning(
                f"  Reprice would eliminate edge on {hedge.ticker} "
                f"(total={new_total}¢), skipping"
            )
            return

        # Don't reprice if we're already at or past the new level
        last_reprice = hedge.last_reprice_price_cents or original_price
        if new_price <= last_reprice:
            return

        # For the amend API, Kalshi uses yes_price for both sides
        if unfilled_side == "no":
            amend_yes_price = 100 - new_price
        else:
            amend_yes_price = new_price

        self.stats.reprices_attempted += 1
        try:
            self.client.amend_order(
                order_id=unfilled_id,
                ticker=hedge.ticker,
                side=unfilled_side,
                action="buy",
                yes_price=amend_yes_price,
            )
            hedge.last_reprice_price_cents = new_price
            hedge.last_reprice_time = time.time()
            logger.info(
                f"  ↑ Repriced {hedge.ticker} {unfilled_side.upper()} "
                f"{original_price}¢ → {new_price}¢ (+{improvement}¢, "
                f"total_cost={new_total}¢, edge={100-new_total}¢)"
            )
        except KalshiAPIError as e:
            logger.warning(f"  Reprice failed on {hedge.ticker}: {e}")

    # ── Capital Management ───────────────────────────────────────

    def compute_deploy_budget(self) -> int:
        """
        Compute how much capital to deploy based on Fractional Kelly criterion.
        f = α · f*, α = 0.25 (NEVER full Kelly per paper).

        Uses observed hedge success rate after enough data (Bayesian-style update).
        """
        alpha = self.config.capital.fractional_kelly_alpha

        # Adaptive Kelly after enough observations
        total_hedges = self.stats.hedges_completed + self.stats.hedges_failed
        if total_hedges >= 20:
            observed_p = self.stats.hedge_success_rate
            kelly_f = compute_kelly_fraction(
                observed_p,
                self.config.capital.net_reward_to_risk_ratio,
                alpha=alpha,
            )
            logger.debug(f"Using observed hedge success rate: {observed_p:.2%}, α={alpha}")
        else:
            kelly_f = compute_kelly_fraction(
                self.config.capital.hedge_success_probability,
                self.config.capital.net_reward_to_risk_ratio,
                alpha=alpha,
            )

        # Use the smaller of fractional Kelly and manual fraction
        fraction = min(kelly_f, self.config.capital.deploy_fraction)

        # Reduce deployment after consecutive failures
        if self._consecutive_failures >= 1:
            scale_down = max(0.10, 1.0 - 0.30 * self._consecutive_failures)
            fraction *= scale_down
            logger.warning(
                f"Scaling down deployment to {scale_down:.0%} after "
                f"{self._consecutive_failures} consecutive failures"
            )

        budget = int(self.config.capital.total_budget_cents * fraction)

        logger.debug(
            f"Capital: Fractional Kelly (α={alpha}) f={kelly_f:.3f}, "
            f"manual={self.config.capital.deploy_fraction}, "
            f"using={fraction:.3f}, budget={budget}¢"
        )
        return budget

    def get_available_slots(self, budget_cents: int, mid_price: float) -> int:
        """How many more hedges can we place given current budget and active positions."""
        max_orders = compute_max_orders(
            budget_cents,
            self.config.capital.contracts_per_order,
            mid_price,
        )
        active_count = len(self.active_hedges)
        return max(0, max_orders - active_count)

    # ── Main Loop ────────────────────────────────────────────────

    def run(self):
        """
        Algorithm 2: MainLoop — Continuous monitoring and hedge execution.

        Improvements:
        - Per-market cooldowns after failures
        - Consecutive failure detection with auto-pause
        - Pre-flight edge validation
        - Dynamic timeouts
        """
        self._running = True
        logger.info("=" * 60)
        logger.info("  OPTIMUS v3 — Hedged Liquidity Bot + Risk Engine")
        logger.info("=" * 60)
        logger.info(f"  Mode:        {'DRY RUN' if self.config.dry_run else 'LIVE'}")
        logger.info(f"  Environment: {self.config.kalshi.base_url}")
        logger.info(f"  Budget:      {self.config.capital.total_budget_cents}¢ (${self.config.capital.total_budget_cents/100:.2f})")
        logger.info(f"  Δ_max:       {self.config.hedge.delta_max_cents}¢")
        logger.info(f"  Kelly α:     {self.config.capital.fractional_kelly_alpha} (fractional)")
        logger.info(f"  Max Expose:  {self.config.capital.max_exposure_cents}¢ (${self.config.capital.max_exposure_cents/100:.2f})")
        logger.info(f"  Poll:        {self.config.poll_interval_seconds}s")
        logger.info(f"  MDD Limit:   {self.config.risk.max_drawdown_pct:.0%}")
        logger.info(f"  VaR Limit:   {self.config.risk.daily_var_limit_cents}¢/day")
        logger.info(f"  Loss Halt:   {self.config.risk.max_consecutive_losses} consecutive")
        if self.config.target_tickers:
            logger.info(f"  Tickers:     {', '.join(self.config.target_tickers)}")
        else:
            logger.info(f"  Series:      {', '.join(self.config.target_series)}")
            logger.info(f"  Mode:        Auto-discover best markets")
        logger.info("=" * 60)

        # Verify connectivity
        if not self.config.dry_run:
            try:
                balance = self.client.get_balance()
                bal_cents = balance.get("balance", 0)
                logger.info(f"  Account balance: {bal_cents}¢ (${bal_cents/100:.2f})")
            except KalshiAPIError as e:
                logger.error(f"Cannot connect to Kalshi: {e}")
                return

        market_refresh_interval = 300  # Re-scan markets every 5 minutes
        last_market_scan = 0
        target_markets: list[tuple[MarketInfo, MarketScore]] = []

        while self._running:
            try:
                loop_start = time.time()

                # ── Risk Guard: check if risk halt should be lifted ──
                if self.stats.risk_halted:
                    elapsed_since_halt = time.time() - self.stats.risk_halt_time
                    if elapsed_since_halt < self.config.risk.resume_after_halt_seconds:
                        remaining = self.config.risk.resume_after_halt_seconds - elapsed_since_halt
                        logger.debug(
                            f"Risk halt active ({self.stats.risk_halt_reason}), "
                            f"resuming in {remaining:.0f}s"
                        )
                        time.sleep(min(30, remaining))
                        continue
                    else:
                        logger.info("Risk halt expired, resuming trading")
                        self.stats.risk_halted = False
                        self.stats.risk_halt_reason = ""
                        self._consecutive_failures = 0

                # ── Risk Guard: Max Drawdown circuit breaker ──
                mdd = self.stats.max_drawdown
                if mdd > self.config.risk.max_drawdown_pct:
                    self.stats.risk_halted = True
                    self.stats.risk_halt_time = time.time()
                    self.stats.risk_halt_reason = (
                        f"MDD {mdd:.1%} > {self.config.risk.max_drawdown_pct:.0%} limit"
                    )
                    logger.error(f"RISK HALT: {self.stats.risk_halt_reason}")
                    continue

                # ── Risk Guard: VaR daily loss limit ──
                daily_loss = -self.stats.net_pnl_cents  # positive = loss
                if daily_loss > self.config.risk.daily_var_limit_cents:
                    self.stats.risk_halted = True
                    self.stats.risk_halt_time = time.time()
                    self.stats.risk_halt_reason = (
                        f"Daily loss {daily_loss}¢ > VaR limit {self.config.risk.daily_var_limit_cents}¢"
                    )
                    logger.error(f"RISK HALT: {self.stats.risk_halt_reason}")
                    continue

                # ── Risk Guard: consecutive loss halt (2 per images spec) ──
                if self._consecutive_failures >= self._max_consecutive_before_pause:
                    self.stats.risk_halted = True
                    self.stats.risk_halt_time = time.time()
                    self.stats.risk_halt_reason = (
                        f"{self._consecutive_failures} consecutive losses"
                    )
                    logger.error(
                        f"RISK HALT: {self.stats.risk_halt_reason} — "
                        f"pausing for {self.config.risk.resume_after_halt_seconds}s"
                    )
                    continue

                # ── Step 1: Refresh market list periodically ─────────
                if time.time() - last_market_scan > market_refresh_interval:
                    if self.config.target_tickers:
                        target_markets = []
                        for ticker in self.config.target_tickers:
                            try:
                                raw = self.client.get_market(ticker)
                                market = parse_market(raw)
                                if market.status == "active":
                                    ob_raw = self.client.get_orderbook(ticker, depth=10)
                                    ob = parse_orderbook(ticker, ob_raw)
                                    score = score_market(market, ob, self.config.market_selection)
                                    if score:
                                        target_markets.append((market, score))
                                    else:
                                        logger.info(
                                            f"{ticker}: failed score_market hard filter "
                                            f"(price={market.yes_price}, vol24h={market.volume_24h}, "
                                            f"spread={ob.spread_cents}) — using dummy score"
                                        )
                                        dummy_score = MarketScore(
                                            ticker=ticker,
                                            price_split_score=0.5,
                                            spread_score=0.5,
                                            competition_score=0.5,
                                            volatility_score=0.5,
                                            volume_score=0.5,
                                        )
                                        dummy_score.compute_composite()
                                        target_markets.append((market, dummy_score))
                            except KalshiAPIError as e:
                                logger.warning(f"Cannot fetch {ticker}: {e}")
                    else:
                        target_markets = self.discover_markets()

                    last_market_scan = time.time()

                # ── Step 2: Place new hedges on suitable markets ─────
                budget = self.compute_deploy_budget()

                # Pre-check: compute current exposure for cap enforcement
                current_exposure = sum(
                    h.hedge_cost_cents * h.contracts
                    for h in self.active_hedges.values()
                )

                for market, score in target_markets:
                    if not self._running:
                        break

                    ticker = market.ticker

                    # Skip if we already have an active hedge on this market
                    if ticker in self.active_hedges:
                        continue

                    # ── Improvement: skip markets on cooldown ──
                    if self._is_cooled_down(ticker):
                        logger.debug(f"{ticker}: still on cooldown, skipping")
                        continue

                    # ── Rec #5: skip markets on thin-market backoff ──
                    if ticker in self._thin_market_backoff and time.time() < self._thin_market_backoff[ticker]:
                        continue

                    # Exposure cap: exposure + bet <= max_exposure
                    estimated_bet_cost = 100 * self.config.capital.contracts_per_order
                    if not check_exposure_limit(
                        current_exposure, estimated_bet_cost,
                        self.config.capital.max_exposure_cents,
                    ):
                        logger.debug(
                            f"Exposure cap reached ({current_exposure}¢ + {estimated_bet_cost}¢ > "
                            f"{self.config.capital.max_exposure_cents}¢), skipping new hedges"
                        )
                        break

                    # Concentration cap
                    if len(self.active_hedges) >= self.config.market_selection.max_active_hedges:
                        logger.debug(
                            f"Active hedge cap reached ({self.config.market_selection.max_active_hedges}), "
                            f"skipping remaining markets"
                        )
                        break

                    # Get fresh orderbook
                    try:
                        ob_raw = self.client.get_orderbook(ticker, depth=10)
                        ob = parse_orderbook(ticker, ob_raw)
                    except KalshiAPIError as e:
                        logger.warning(f"{ticker}: orderbook fetch failed: {e}")
                        continue

                    # ── Rec #3: skip if top-of-book hasn't changed since last scan ──
                    spread_key = (ob.best_yes_bid, ob.best_no_bid)
                    if spread_key == self._last_spread.get(ticker):
                        continue
                    self._last_spread[ticker] = spread_key

                    # Two-sided book filter
                    min_bids = self.config.market_selection.min_bids_each_side
                    if len(ob.yes_bids) < min_bids or len(ob.no_bids) < min_bids:
                        self.stats.markets_skipped_thin_book += 1
                        logger.debug(
                            f"{ticker}: skipping — thin/one-sided book "
                            f"(yes_bids={len(ob.yes_bids)}, no_bids={len(ob.no_bids)}, "
                            f"need {min_bids} each)"
                        )
                        continue

                    mid = ob.mid_price_cents
                    if mid is None:
                        logger.warning(
                            f"{ticker}: mid_price is None — "
                            f"yes_bids={len(ob.yes_bids)} no_bids={len(ob.no_bids)} "
                            f"best_yes_bid={ob.best_yes_bid} best_no_bid={ob.best_no_bid}"
                        )
                        continue

                    # Check capital slots
                    slots = self.get_available_slots(budget, mid)
                    if slots <= 0:
                        logger.debug(
                            f"No available capital slots (budget={budget}¢, mid={mid:.1f}¢, "
                            f"active={len(self.active_hedges)}), skipping new hedges"
                        )
                        break

                    # Place hedged orders
                    hedge = self.place_hedged_orders(ticker, mid, ob, volume_24h=market.volume_24h)
                    if hedge:
                        self.active_hedges[ticker] = hedge
                        current_exposure += hedge.hedge_cost_cents * hedge.contracts

                # ── Step 3: Monitor active hedges ────────────────────
                to_remove = []
                for ticker, hedge in self.active_hedges.items():
                    if not self._running:
                        break
                    keep = self.manage_hedge_lifecycle(hedge)
                    if not keep:
                        to_remove.append(ticker)

                for ticker in to_remove:
                    del self.active_hedges[ticker]

                # ── Step 4: Log status with risk metrics ──────────────
                logger.debug(
                    f"Loop complete: {len(target_markets)} target markets | "
                    f"Active hedges: {len(self.active_hedges)} | "
                    f"Completed: {self.stats.hedges_completed} | "
                    f"Orders placed: {self.stats.orders_placed} | "
                    f"Net P&L: {self.stats.net_pnl_cents}¢ | "
                    f"MDD: {self.stats.max_drawdown:.1%} | "
                    f"PF: {self.stats.profit_factor:.2f} | "
                    f"SR: {self.stats.sharpe_ratio:.2f}"
                )

                # ── Step 5: Sleep ────────────────────────────────────
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.config.poll_interval_seconds - elapsed)
                if sleep_time > 0 and self._running:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                time.sleep(5)

        # ── Cleanup ──────────────────────────────────────────────
        self._cleanup()

    def _cleanup(self):
        """Cancel all active orders on shutdown."""
        logger.info("Cleaning up active orders...")
        for ticker, hedge in self.active_hedges.items():
            for order_id in [hedge.yes_order_id, hedge.no_order_id]:
                if order_id and not self.config.dry_run:
                    try:
                        self.client.cancel_order(order_id)
                        self.stats.orders_cancelled += 1
                        logger.info(f"  Cancelled {order_id} on {ticker}")
                    except KalshiAPIError as e:
                        logger.warning(f"  Failed to cancel {order_id}: {e}")

        logger.info(self.stats.summary())
