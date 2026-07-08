"""
Kalshi API client with RSA-PSS authentication.

v2 improvements:
- Automatic retry with exponential backoff on transient errors (429, 500, 502, 503, 504)
- Separate rate limiters for read vs write with configurable limits
- Connection keep-alive and timeout tuning
- Request ID tracking for debugging
- Structured error classification (transient vs permanent)
"""

import base64
import datetime
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional, cast

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from config import KalshiConfig

logger = logging.getLogger(__name__)

# HTTP status codes that are safe to retry
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
def cents_to_dollar_str(cents: int) -> str:
    """
    Convert integer cents (1-99) to a Kalshi dollar string like '0.6500'.

    Kalshi removed integer cent fields in March 2026. All order prices must
    now be sent as fixed-point dollar strings with up to 4 decimal places.
    Subpenny markets may use values like '0.6505'.
    """
    if not isinstance(cents, int):
        raise TypeError(f"cents must be int, got {type(cents).__name__}")
    if cents < 1 or cents > 99:
        raise ValueError(f"cents must be in [1, 99], got {cents}")
    return f"{cents / 100:.4f}"

class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns an error."""
    def __init__(self, status_code: int, message: str, response: Optional[dict] = None):
        self.status_code = status_code
        self.message = message
        self.response = response or {}
        super().__init__(f"Kalshi API {status_code}: {message}")

    @property
    def is_transient(self) -> bool:
        """Whether this error is likely transient and worth retrying."""
        return self.status_code in TRANSIENT_STATUS_CODES

    @property
    def is_rate_limited(self) -> bool:
        return self.status_code == 429


class RateLimiter:
    """
    Token-bucket rate limiter with burst support.

    v2: tracks recent request timestamps for more accurate limiting.
    """
    def __init__(self, max_calls: int = 10, period_seconds: float = 1.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: list[float] = []

    def wait(self):
        now = time.time()
        # Prune old entries
        self.calls = [t for t in self.calls if now - t < self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                logger.debug(f"Rate limiter: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
        self.calls.append(time.time())

    @property
    def current_usage(self) -> int:
        """How many calls have been made in the current window."""
        now = time.time()
        return len([t for t in self.calls if now - t < self.period])


class KalshiClient:
    """
    Authenticated Kalshi REST API client.

    v2 improvements:
    - Automatic retry on transient errors
    - Better connection management
    - Request ID tracking
    - Separate read/write rate limits

    Usage:
        config = KalshiConfig.from_env()
        client = KalshiClient(config)
        balance = client.get_balance()
    """

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 0.5  # seconds
    RETRY_BACKOFF_MAX = 8.0   # seconds

    def __init__(self, config: KalshiConfig):
        self.config = config
        self.base_url = config.base_url + config.api_path
        self.api_key_id = config.api_key_id
        self.private_key = self._load_private_key(config.private_key_path)

        # ── Improvement: connection pooling with keep-alive ──
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=0,  # We handle retries ourselves for auth header freshness
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # Separate rate limiters (Kalshi has different limits for read vs write)
        self.read_limiter = RateLimiter(max_calls=10, period_seconds=1.0)
        self.write_limiter = RateLimiter(max_calls=10, period_seconds=1.0)

        # ── Improvement: request tracking ──
        self._request_count = 0
        self._error_count = 0

    def _load_private_key(self, path: str) -> rsa.RSAPrivateKey:
        """Load RSA private key from PEM file."""
        key_path = Path(path)
        if not key_path.exists():
            raise FileNotFoundError(f"Private key not found: {path}")
        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend(),
            )
        return private_key  # type: ignore[return-value]

    def _sign(self, text: str) -> str:
        """Sign text using RSA-PSS with SHA-256."""
        message = text.encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _headers(self, method: str, path: str) -> dict:
        """Generate authenticated headers for a request."""
        timestamp_ms = str(int(datetime.datetime.now().timestamp() * 1000))
        full_path = self.config.api_path + path
        path_for_signing = full_path.split("?")[0]
        msg = timestamp_ms + method.upper() + path_for_signing
        signature = self._sign(msg)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        data: Optional[dict] = None,
        is_write: bool = False,
    ) -> dict:
        """
        Make an authenticated request to the Kalshi API.

        v2: automatic retry on transient errors with fresh auth headers.
        """
        if is_write:
            self.write_limiter.wait()
        else:
            self.read_limiter.wait()

        url = self.base_url + path
        query_string = ""
        if params:
            query_string = "?" + "&".join(f"{k}={v}" for k, v in params.items() if v is not None)

        last_error = None
        for attempt in range(self.MAX_RETRIES + 1):
            # Generate fresh headers each attempt (timestamp must be current)
            headers = self._headers(method, path)
            self._request_count += 1

            try:
                response = self.session.request(
                    method=method,
                    url=url + query_string,
                    headers=headers,
                    json=data if data else None,
                    timeout=10,
                )
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    wait = min(self.RETRY_BACKOFF_BASE * (2 ** attempt), self.RETRY_BACKOFF_MAX)
                    logger.warning(f"Connection error (attempt {attempt+1}), retrying in {wait:.1f}s: {e}")
                    time.sleep(wait)
                    continue
                logger.error(f"Request failed after {self.MAX_RETRIES + 1} attempts: {e}")
                raise
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    wait = min(self.RETRY_BACKOFF_BASE * (2 ** attempt), self.RETRY_BACKOFF_MAX)
                    logger.warning(f"Timeout (attempt {attempt+1}), retrying in {wait:.1f}s")
                    time.sleep(wait)
                    continue
                logger.error(f"Request timed out after {self.MAX_RETRIES + 1} attempts")
                raise
            except requests.exceptions.RequestException as e:
                # Catch-all for other request errors (ChunkedEncodingError, etc.)
                last_error = e
                if attempt < self.MAX_RETRIES:
                    wait = min(self.RETRY_BACKOFF_BASE * (2 ** attempt), self.RETRY_BACKOFF_MAX)
                    logger.warning(f"Request error (attempt {attempt+1}), retrying in {wait:.1f}s: {e}")
                    time.sleep(wait)
                    continue
                logger.error(f"Request failed after {self.MAX_RETRIES + 1} attempts: {e}")
                raise

            if response.status_code >= 400:
                try:
                    error_body = response.json()
                except Exception:
                    error_body = {"raw": response.text}

                error = KalshiAPIError(
                    response.status_code,
                    error_body.get("message", response.text),
                    error_body,
                )
                self._error_count += 1

                # ── Improvement: retry on transient errors ──
                if error.is_transient and attempt < self.MAX_RETRIES:
                    wait = min(self.RETRY_BACKOFF_BASE * (2 ** attempt), self.RETRY_BACKOFF_MAX)
                    if error.is_rate_limited:
                        # Use Retry-After header if available
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait = max(wait, float(retry_after))
                        logger.warning(f"Rate limited, waiting {wait:.1f}s")
                    else:
                        logger.warning(
                            f"Transient error {response.status_code} "
                            f"(attempt {attempt+1}), retrying in {wait:.1f}s"
                        )
                    time.sleep(wait)
                    continue

                raise error

            if response.status_code == 204:
                return {}
            return response.json()

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        return {}

    # ── Market Data ──────────────────────────────────────────────

    def get_markets(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
        status: str = "open",
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
    ) -> dict:
        """Get list of markets."""
        params = {
            "limit": limit,
            "cursor": cursor,
            "status": status,
            "series_ticker": series_ticker,
            "event_ticker": event_ticker,
        }
        return self._request("GET", "/markets", params=params)

    def get_market(self, ticker: str) -> dict:
        """Get a single market by ticker."""
        return self._request("GET", f"/markets/{ticker}")

    def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """
        Get the order book for a market.

        Returns yes bids and no bids. In binary markets:
        - yes bid at X¢ = no ask at (100-X)¢
        - no bid at X¢ = yes ask at (100-X)¢
        """
        params = {"depth": depth}
        return self._request("GET", f"/markets/{ticker}/orderbook", params=params)

    def get_trades(self, ticker: str, limit: int = 50, cursor: Optional[str] = None) -> dict:
        """Get recent trades for a market."""
        params = {"ticker": ticker, "limit": limit, "cursor": cursor}
        return self._request("GET", "/markets/trades", params=params)

    def get_events(self, limit: int = 100, status: str = "open", cursor: Optional[str] = None) -> dict:
        """Get list of events."""
        params = {"limit": limit, "status": status, "cursor": cursor}
        return self._request("GET", "/events", params=params)

    # ── Portfolio ────────────────────────────────────────────────

    def get_balance(self) -> dict:
        """Get account balance."""
        return self._request("GET", "/portfolio/balance")

    def get_positions(self, limit: int = 100, cursor: Optional[str] = None, settlement_status: Optional[str] = None) -> dict:
        """Get current positions."""
        params = {"limit": limit, "cursor": cursor, "settlement_status": settlement_status}
        return self._request("GET", "/portfolio/positions", params=params)

    def get_fills(self, ticker: Optional[str] = None, limit: int = 100, cursor: Optional[str] = None) -> dict:
        """Get fill history."""
        params = {"ticker": ticker, "limit": limit, "cursor": cursor}
        return self._request("GET", "/portfolio/fills", params=params)

    # ── Orders ───────────────────────────────────────────────────

    def create_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        action: str,  # "buy" or "sell"
        count: int,
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        client_order_id: Optional[str] = None,
        post_only: bool = True,
        time_in_force: Optional[str] = None,
        expiration_ts: Optional[int] = None,
    ) -> dict:
        """
        Place a limit order.

        Note: Kalshi's March 2026 API removed integer cent fields.
        yes_price/no_price are still accepted here in integer cents for
        backwards-compatible call sites; they are converted to the
        '*_dollars' string format at the wire boundary.
        """
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        order_data = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "type": "limit",
            "count": count,
            "client_order_id": client_order_id,
        }
        if yes_price is not None:
            order_data["yes_price_dollars"] = cents_to_dollar_str(yes_price)
        if no_price is not None:
            order_data["no_price_dollars"] = cents_to_dollar_str(no_price)
        if post_only:
            order_data["post_only"] = True
        if time_in_force:
            order_data["time_in_force"] = time_in_force
        if expiration_ts:
            order_data["expiration_ts"] = expiration_ts

        logger.info(f"Creating order: {order_data}")
        return self._request("POST", "/portfolio/orders", data=order_data, is_write=True)

    def batch_create_orders(self, orders: list[dict]) -> dict:
        """
        Place multiple orders in a single batch (up to 20).

        Accepts integer cent fields (yes_price / no_price) in the input dicts
        for backwards compatibility and converts them to '*_dollars' strings
        before transmission.
        """
        for order in orders:
            if "client_order_id" not in order:
                order["client_order_id"] = str(uuid.uuid4())
            order.setdefault("type", "limit")
            # Convert any integer cent fields to dollar strings
            if "yes_price" in order:
                order["yes_price_dollars"] = cents_to_dollar_str(order.pop("yes_price"))
            if "no_price" in order:
                order["no_price_dollars"] = cents_to_dollar_str(order.pop("no_price"))

        logger.info(f"Batch creating {len(orders)} orders")
        return self._request(
            "POST", "/portfolio/orders/batched",
            data={"orders": orders},
            is_write=True,
        )

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a single order."""
        logger.info(f"Cancelling order: {order_id}")
        return self._request("DELETE", f"/portfolio/orders/{order_id}", is_write=True)

    def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> dict:
        """Get orders, optionally filtered by ticker and status."""
        params = {"ticker": ticker, "status": status, "limit": limit, "cursor": cursor}
        return self._request("GET", "/portfolio/orders", params=params)

    def get_order(self, order_id: str) -> dict:
        """Get a single order by ID."""
        return self._request("GET", f"/portfolio/orders/{order_id}")

    def amend_order(
        self,
        order_id: str,
        ticker: str,
        side: str,
        action: str,
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        count: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """Amend an existing order's price or count."""
        data: dict[str, Any] = {"ticker": ticker, "side": side, "action": action}
        if yes_price is not None:
            data["yes_price_dollars"] = cents_to_dollar_str(yes_price)
        if no_price is not None:
            data["no_price_dollars"] = cents_to_dollar_str(no_price)
        if count is not None:
            data["count"] = count
        if client_order_id:
            data["client_order_id"] = client_order_id
        logger.info(f"Amending order {order_id}: {data}")
        return self._request(
            "POST", f"/portfolio/orders/{order_id}/amend",
            data=data, is_write=True,
        )

    @property
    def request_stats(self) -> dict:
        """Return request/error counts for monitoring."""
        return {
            "total_requests": self._request_count,
            "total_errors": self._error_count,
            "read_limiter_usage": self.read_limiter.current_usage,
            "write_limiter_usage": self.write_limiter.current_usage,
        }
