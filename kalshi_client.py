"""
Kalshi API client with RSA-PSS authentication.

Handles:
- RSA-PSS request signing
- Rate limiting
- Order placement, amendment, cancellation
- Market data retrieval
- Portfolio/position queries
"""

import base64
import datetime
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from config import KalshiConfig

logger = logging.getLogger(__name__)


class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns an error."""
    def __init__(self, status_code: int, message: str, response: dict = None):
        self.status_code = status_code
        self.message = message
        self.response = response or {}
        super().__init__(f"Kalshi API {status_code}: {message}")


class RateLimiter:
    """Simple token-bucket rate limiter."""
    def __init__(self, max_calls: int = 10, period_seconds: float = 1.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls: list[float] = []

    def wait(self):
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.calls.append(time.time())


class KalshiClient:
    """
    Authenticated Kalshi REST API client.

    Usage:
        config = KalshiConfig.from_env()
        client = KalshiClient(config)
        balance = client.get_balance()
    """

    def __init__(self, config: KalshiConfig):
        self.config = config
        self.base_url = config.base_url + config.api_path
        self.api_key_id = config.api_key_id
        self.private_key = self._load_private_key(config.private_key_path)
        self.session = requests.Session()
        self.read_limiter = RateLimiter(max_calls=10, period_seconds=1.0)
        self.write_limiter = RateLimiter(max_calls=10, period_seconds=1.0)

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
        return private_key

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
        # The full path including /trade-api/v2 prefix must be signed
        full_path = self.config.api_path + path
        # Strip query parameters before signing
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
        params: dict = None,
        data: dict = None,
        is_write: bool = False,
    ) -> dict:
        """Make an authenticated request to the Kalshi API."""
        if is_write:
            self.write_limiter.wait()
        else:
            self.read_limiter.wait()

        url = self.base_url + path
        query_string = ""
        if params:
            query_string = "?" + "&".join(f"{k}={v}" for k, v in params.items() if v is not None)

        headers = self._headers(method, path)

        try:
            response = self.session.request(
                method=method,
                url=url + query_string,
                headers=headers,
                json=data if data else None,
                timeout=10,
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

        if response.status_code >= 400:
            try:
                error_body = response.json()
            except Exception:
                error_body = {"raw": response.text}
            raise KalshiAPIError(
                response.status_code,
                error_body.get("message", response.text),
                error_body,
            )

        if response.status_code == 204:
            return {}
        return response.json()

    # ── Market Data ──────────────────────────────────────────────

    def get_markets(
        self,
        limit: int = 100,
        cursor: str = None,
        status: str = "open",
        series_ticker: str = None,
        event_ticker: str = None,
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

    def get_trades(self, ticker: str, limit: int = 50, cursor: str = None) -> dict:
        """Get recent trades for a market."""
        params = {"ticker": ticker, "limit": limit, "cursor": cursor}
        return self._request("GET", "/markets/trades", params=params)

    def get_events(self, limit: int = 100, status: str = "open", cursor: str = None) -> dict:
        """Get list of events."""
        params = {"limit": limit, "status": status, "cursor": cursor}
        return self._request("GET", "/events", params=params)

    # ── Portfolio ────────────────────────────────────────────────

    def get_balance(self) -> dict:
        """Get account balance."""
        return self._request("GET", "/portfolio/balance")

    def get_positions(self, limit: int = 100, cursor: str = None, settlement_status: str = None) -> dict:
        """Get current positions."""
        params = {"limit": limit, "cursor": cursor, "settlement_status": settlement_status}
        return self._request("GET", "/portfolio/positions", params=params)

    def get_fills(self, ticker: str = None, limit: int = 100, cursor: str = None) -> dict:
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
        yes_price: int = None,  # price in cents
        no_price: int = None,   # price in cents
        client_order_id: str = None,
        post_only: bool = True,
        time_in_force: str = None,
        expiration_ts: int = None,
    ) -> dict:
        """
        Place a limit order.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            yes_price: Price in cents (1-99) for yes side
            no_price: Price in cents (1-99) for no side
            client_order_id: Unique client ID for deduplication
            post_only: If True, order is rejected if it would immediately match
            time_in_force: "gtc" (default), "fill_or_kill", "ioc"
            expiration_ts: Unix timestamp in seconds for order expiry
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
            order_data["yes_price"] = yes_price
        if no_price is not None:
            order_data["no_price"] = no_price
        if post_only:
            order_data["post_only"] = True
        if time_in_force:
            order_data["time_in_force"] = time_in_force
        if expiration_ts:
            order_data["expiration_ts"] = expiration_ts

        logger.info(f"Creating order: {order_data}")
        return self._request("POST", "/portfolio/orders", data=order_data, is_write=True)

    def batch_create_orders(self, orders: list[dict]) -> dict:
        """Place multiple orders in a single batch (up to 20)."""
        for order in orders:
            if "client_order_id" not in order:
                order["client_order_id"] = str(uuid.uuid4())
            order.setdefault("type", "limit")

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
        ticker: str = None,
        status: str = None,
        limit: int = 100,
        cursor: str = None,
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
        yes_price: int = None,
        no_price: int = None,
        count: int = None,
        client_order_id: str = None,
    ) -> dict:
        """Amend an existing order's price or count."""
        data = {"ticker": ticker, "side": side, "action": action}
        if yes_price is not None:
            data["yes_price"] = yes_price
        if no_price is not None:
            data["no_price"] = no_price
        if count is not None:
            data["count"] = count
        if client_order_id:
            data["client_order_id"] = client_order_id
        logger.info(f"Amending order {order_id}: {data}")
        return self._request(
            "POST", f"/portfolio/orders/{order_id}/amend",
            data=data, is_write=True,
        )
