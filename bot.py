"""
Main bot loop: Continuous monitoring and hedge execution.

Implements Algorithm 2 (MainLoop) from the paper:
1. Scan markets and select suitable ones
2. Compute optimal order prices
3. Place hedged (YES + NO) limit orders
4. Monitor fills and manage hedge lifecycle
5. Cancel/replace stale orders
6. Track P&L and performance metrics
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
    compute_expected_return,
    compute_kelly_fraction,
    compute_max_orders,
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

    @property
    def is_fully_hedged(self) -> bool:
        return self.yes_filled and self.no_filled

    @property
    def is_partially_filled(self) -> bool:
        return (self.yes_filled or self.no_filled) and not self.is_fully_hedged

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class BotStats:
    """Performance tracking."""
    orders_placed: int = 0
    orders_cancelled: int = 0
    hedges_completed: int = 0
    hedges_failed: int = 0
    partial_fills: int = 0
    total_yes_fills: int = 0
    total_no_fills: int = 0
    total_reward_score: float = 0.0
    total_pnl_cents: int = 0
    total_overpayment_cents: int = 0
    started_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        runtime = time.time() - self.started_at
        hours = runtime / 3600
        return (
            f"\n{'='*60}\n"
            f"  Bot Performance Summary\n"
            f"{'='*60}\n"
            f"  Runtime:            {hours:.1f} hours\n"
            f"  Orders placed:      {self.orders_placed}\n"
            f"  Orders cancelled:   {self.orders_cancelled}\n"
            f"  Hedges completed:   {self.hedges_completed}\n"
            f"  Hedges failed:      {self.hedges_failed}\n"
            f"  Partial fills:      {self.partial_fills}\n"
            f"  Total reward score: {self.total_reward_score:.2f}\n"
            f"  Total P&L:          {self.total_pnl_cents}¢ (${self.total_pnl_cents/100:.2f})\n"
            f"  Overpayment:        {self.total_overpayment_cents}¢\n"
            f"{'='*60}\n"
        )


# ── Orderbook parsing ────────────────────────────────────────────────


def parse_orderbook(ticker: str, raw: dict) -> OrderBookSnapshot:
    """Parse Kalshi orderbook response into our data structure."""
    ob = raw.get("orderbook", {})

    # Kalshi returns [[price, quantity], ...] for yes and no bids
    # Using the dollar-denominated fields for clarity
    yes_bids = []
    no_bids = []

    # Try cents first, fall back to dollar fields
    if "yes" in ob and ob["yes"]:
        for entry in ob["yes"]:
            if isinstance(entry, list) and len(entry) >= 2:
                yes_bids.append((int(entry[0]), int(entry[1])))
            elif isinstance(entry, list) and len(entry) == 1:
                # Some responses are just [price]
                yes_bids.append((int(entry[0]), 1))
    elif "yes_dollars" in ob and ob["yes_dollars"]:
        for entry in ob["yes_dollars"]:
            if isinstance(entry, list) and len(entry) >= 2:
                price_cents = int(float(entry[0]) * 100)
                qty = int(entry[1])
                yes_bids.append((price_cents, qty))

    if "no" in ob and ob["no"]:
        for entry in ob["no"]:
            if isinstance(entry, list) and len(entry) >= 2:
                no_bids.append((int(entry[0]), int(entry[1])))
            elif isinstance(entry, list) and len(entry) == 1:
                no_bids.append((int(entry[0]), 1))
    elif "no_dollars" in ob and ob["no_dollars"]:
        for entry in ob["no_dollars"]:
            if isinstance(entry, list) and len(entry) >= 2:
                price_cents = int(float(entry[0]) * 100)
                qty = int(entry[1])
                no_bids.append((price_cents, qty))

    # Sort descending by price (best bids first)
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

    Implements the full framework:
    1. Market scanning & selection
    2. Hedged order placement (YES + NO simultaneously)
    3. Fill monitoring & hedge lifecycle management
    4. Risk-bounded capital allocation
    """

    def __init__(self, config: BotConfig, client: KalshiClient):
        self.config = config
        self.client = client
        self.stats = BotStats()
        self.active_hedges: dict[str, ActiveHedge] = {}  # ticker -> ActiveHedge
        self.market_scores: dict[str, MarketScore] = {}
        self._running = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown signal received, cleaning up...")
        self._running = False

    # ── Market Discovery ─────────────────────────────────────────

    def discover_markets(self) -> list[tuple[MarketInfo, MarketScore]]:
        """
        Scan all open markets and rank by suitability.
        Returns list of (market, score) tuples sorted by composite score.
        """
        logger.info("Scanning markets...")
        scored_markets = []

        try:
            result = self.client.get_markets(limit=100, status="open")
            markets = result.get("markets", [])
        except KalshiAPIError as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

        for raw_market in markets:
            market = parse_market(raw_market)
            if market.status != "open":
                continue

            try:
                ob_raw = self.client.get_orderbook(market.ticker, depth=10)
                ob = parse_orderbook(market.ticker, ob_raw)
            except KalshiAPIError:
                continue

            score = score_market(market, ob, self.config.market_selection)
            if score is not None:
                scored_markets.append((market, score))
                self.market_scores[market.ticker] = score

        # Sort by composite score descending
        scored_markets.sort(key=lambda x: x[1].composite, reverse=True)

        logger.info(
            f"Found {len(scored_markets)} suitable markets out of {len(markets)} total"
        )
        for market, score in scored_markets[:5]:
            logger.info(
                f"  {market.ticker}: score={score.composite:.3f} "
                f"price={market.yes_price}¢ vol24h={market.volume_24h}"
            )

        return scored_markets

    # ── Order Placement ──────────────────────────────────────────

    def place_hedged_orders(self, ticker: str, mid_price: float) -> Optional[ActiveHedge]:
        """
        Place a hedged YES + NO order pair on a market.

        Uses batch order API when available for atomicity.
        Falls back to sequential placement.
        """
        contracts = self.config.capital.contracts_per_order
        hedged = build_hedged_order(ticker, mid_price, contracts, self.config)

        if hedged is None:
            logger.warning(f"{ticker}: could not build valid hedged order")
            return None

        logger.info(
            f"{ticker}: placing hedge YES@{hedged.yes_price_cents}¢ + "
            f"NO@{hedged.no_price_cents}¢ = {hedged.yes_price_cents + hedged.no_price_cents}¢ "
            f"({hedged.contracts} contracts, score={hedged.score:.3f})"
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
            )
            self.stats.orders_placed += 2
            return ah

        yes_cid = str(uuid.uuid4())
        no_cid = str(uuid.uuid4())

        # Try batch order first
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
                    "no_price": hedged.no_price_cents,
                    "client_order_id": no_cid,
                    "post_only": True,
                    "type": "limit",
                },
            ]
            result = self.client.batch_create_orders(orders)
            order_results = result.get("orders", [])

            ah = ActiveHedge(
                ticker=ticker,
                yes_client_id=yes_cid,
                no_client_id=no_cid,
                yes_price_cents=hedged.yes_price_cents,
                no_price_cents=hedged.no_price_cents,
                contracts=contracts,
                created_at=time.time(),
            )

            for order_res in order_results:
                order = order_res.get("order", {})
                oid = order.get("order_id", "")
                cid = order.get("client_order_id", "")
                if cid == yes_cid:
                    ah.yes_order_id = oid
                elif cid == no_cid:
                    ah.no_order_id = oid

            self.stats.orders_placed += 2
            self.stats.total_reward_score += hedged.score
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
                no_price=hedged.no_price_cents,
                client_order_id=no_cid,
                post_only=True,
            )
            ah.no_order_id = no_result.get("order", {}).get("order_id")
            self.stats.orders_placed += 1
        except KalshiAPIError as e:
            logger.error(f"  ✗ NO order failed: {e}")
            # Cancel the YES order since hedge is incomplete
            if ah.yes_order_id:
                try:
                    self.client.cancel_order(ah.yes_order_id)
                    self.stats.orders_cancelled += 1
                except KalshiAPIError:
                    pass
            return None

        self.stats.total_reward_score += hedged.score
        logger.info(f"  ✓ Sequential orders placed: YES={ah.yes_order_id} NO={ah.no_order_id}")
        return ah

    # ── Fill Monitoring ──────────────────────────────────────────

    def check_fills(self, hedge: ActiveHedge) -> ActiveHedge:
        """Check if orders in a hedge have been filled."""
        if self.config.dry_run:
            return hedge

        for order_id, side in [
            (hedge.yes_order_id, "yes"),
            (hedge.no_order_id, "no"),
        ]:
            if order_id is None:
                continue
            try:
                result = self.client.get_order(order_id)
                order = result.get("order", {})
                status = order.get("status", "")
                remaining = order.get("remaining_count", 0)
                fill_count = order.get("fill_count", 0)

                if side == "yes":
                    hedge.yes_fill_count = fill_count
                    if status == "executed" or remaining == 0:
                        hedge.yes_filled = True
                else:
                    hedge.no_fill_count = fill_count
                    if status == "executed" or remaining == 0:
                        hedge.no_filled = True

            except KalshiAPIError as e:
                logger.warning(f"Failed to check order {order_id}: {e}")

        hedge.last_checked = time.time()
        return hedge

    def manage_hedge_lifecycle(self, hedge: ActiveHedge) -> bool:
        """
        Manage a hedge through its lifecycle.

        Returns True if the hedge should be kept, False if it should be removed.

        Lifecycle:
        1. Both resting → keep monitoring
        2. One side filled (adverse fill) → urgently try to fill the other side
        3. Both filled → hedge complete, record P&L
        4. Timeout → cancel remaining orders
        """
        hedge = self.check_fills(hedge)

        # Case: both sides filled → complete hedge
        if hedge.is_fully_hedged:
            pnl = 100 - (hedge.yes_price_cents + hedge.no_price_cents)
            pnl_total = pnl * min(hedge.yes_fill_count, hedge.no_fill_count)
            self.stats.hedges_completed += 1
            self.stats.total_pnl_cents += pnl_total
            self.stats.total_yes_fills += hedge.yes_fill_count
            self.stats.total_no_fills += hedge.no_fill_count
            logger.info(
                f"  ✓ Hedge complete on {hedge.ticker}: "
                f"P&L={pnl_total}¢ per contract resolution"
            )
            return False  # Remove from active

        # Case: one side filled (adverse fill!) → hedge urgently
        if hedge.is_partially_filled:
            self.stats.partial_fills += 1
            filled_side = "YES" if hedge.yes_filled else "NO"
            unfilled_id = hedge.no_order_id if hedge.yes_filled else hedge.yes_order_id
            elapsed = time.time() - hedge.created_at

            logger.warning(
                f"  ⚠ Partial fill on {hedge.ticker}: {filled_side} filled, "
                f"waiting for other side ({elapsed:.0f}s elapsed)"
            )

            # If within hedge timeout, keep waiting
            if elapsed < self.config.hedge.hedge_timeout_seconds:
                return True

            # Timeout reached: cancel the unfilled side to limit exposure
            if elapsed >= self.config.hedge.emergency_cancel_seconds:
                logger.error(
                    f"  ✗ Emergency cancel on {hedge.ticker}: "
                    f"unfilled side after {elapsed:.0f}s"
                )
                if unfilled_id:
                    try:
                        self.client.cancel_order(unfilled_id)
                        self.stats.orders_cancelled += 1
                    except KalshiAPIError:
                        pass
                self.stats.hedges_failed += 1
                # Record the loss from the unhedged fill
                if hedge.yes_filled:
                    loss = hedge.yes_price_cents  # Worst case: market goes to 0
                else:
                    loss = hedge.no_price_cents
                self.stats.total_overpayment_cents += loss
                return False

            return True  # Keep monitoring

        # Case: neither filled, check for staleness
        if hedge.age_seconds > 300:  # 5 minutes with no fills
            logger.info(f"  Cancelling stale hedge on {hedge.ticker}")
            for order_id in [hedge.yes_order_id, hedge.no_order_id]:
                if order_id:
                    try:
                        self.client.cancel_order(order_id)
                        self.stats.orders_cancelled += 1
                    except KalshiAPIError:
                        pass
            return False

        return True  # Keep monitoring

    # ── Capital Management ───────────────────────────────────────

    def compute_deploy_budget(self) -> int:
        """
        Compute how much capital to deploy based on Kelly criterion.
        Returns budget in cents.
        """
        kelly_f = compute_kelly_fraction(
            self.config.capital.hedge_success_probability,
            self.config.capital.net_reward_to_risk_ratio,
        )

        # Use the smaller of Kelly and manual fraction
        fraction = min(kelly_f, self.config.capital.deploy_fraction)
        budget = int(self.config.capital.total_budget_cents * fraction)

        logger.debug(
            f"Capital: Kelly f*={kelly_f:.3f}, manual={self.config.capital.deploy_fraction}, "
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

        1. Discover and rank markets
        2. For each suitable market with available capital:
           a. Get current orderbook
           b. Compute mid price
           c. Place hedged orders
        3. Monitor all active hedges
        4. Sleep and repeat
        """
        self._running = True
        logger.info("=" * 60)
        logger.info("  Kalshi Hedged Liquidity Provision Bot")
        logger.info("=" * 60)
        logger.info(f"  Mode:        {'DRY RUN' if self.config.dry_run else 'LIVE'}")
        logger.info(f"  Environment: {self.config.kalshi.base_url}")
        logger.info(f"  Budget:      {self.config.capital.total_budget_cents}¢ (${self.config.capital.total_budget_cents/100:.2f})")
        logger.info(f"  Δ_max:       {self.config.hedge.delta_max_cents}¢")
        logger.info(f"  Poll:        {self.config.poll_interval_seconds}s")
        if self.config.target_tickers:
            logger.info(f"  Tickers:     {', '.join(self.config.target_tickers)}")
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

                # ── Step 1: Refresh market list periodically ─────────
                if time.time() - last_market_scan > market_refresh_interval:
                    if self.config.target_tickers:
                        # Use specified tickers
                        target_markets = []
                        for ticker in self.config.target_tickers:
                            try:
                                raw = self.client.get_market(ticker)
                                market = parse_market(raw)
                                if market.status == "open":
                                    ob_raw = self.client.get_orderbook(ticker, depth=10)
                                    ob = parse_orderbook(ticker, ob_raw)
                                    score = score_market(market, ob, self.config.market_selection)
                                    if score:
                                        target_markets.append((market, score))
                                    else:
                                        # Still include manually specified tickers
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

                for market, score in target_markets:
                    if not self._running:
                        break

                    ticker = market.ticker

                    # Skip if we already have an active hedge on this market
                    if ticker in self.active_hedges:
                        continue

                    # Get fresh orderbook
                    try:
                        ob_raw = self.client.get_orderbook(ticker, depth=10)
                        ob = parse_orderbook(ticker, ob_raw)
                    except KalshiAPIError:
                        continue

                    mid = ob.mid_price_cents
                    if mid is None:
                        continue

                    # Check if we have capital slots available
                    slots = self.get_available_slots(budget, mid)
                    if slots <= 0:
                        logger.debug("No available capital slots, skipping new hedges")
                        break

                    # Place hedged orders
                    hedge = self.place_hedged_orders(ticker, mid)
                    if hedge:
                        self.active_hedges[ticker] = hedge

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

                # ── Step 4: Log status ───────────────────────────────
                if self.active_hedges:
                    logger.debug(
                        f"Active hedges: {len(self.active_hedges)} | "
                        f"Completed: {self.stats.hedges_completed} | "
                        f"P&L: {self.stats.total_pnl_cents}¢"
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
                time.sleep(5)  # Back off on errors

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
