#!/usr/bin/env python3
"""
Kalshi Hedged Liquidity Provision Bot — Entry Point

Usage:
    # Dry run with demo environment (default):
    python main.py

    # Live trading on demo:
    python main.py --live

    # Specify tickers:
    python main.py --tickers TICKER1 TICKER2

    # Production (real money!):
    KALSHI_ENV=prod python main.py --live

    # Custom budget:
    python main.py --budget 500 --live

Environment Variables:
    KALSHI_API_KEY_ID       Your Kalshi API key ID
    KALSHI_PRIVATE_KEY_PATH Path to your RSA private key .pem file
    KALSHI_ENV              "demo" (default) or "prod"
"""

import argparse
import logging
import sys
from pathlib import Path

from config import BotConfig
from kalshi_client import KalshiClient
from bot import LiquidityBot


def setup_logging(level: str = "INFO", log_file: str = None):
    """Configure logging for the bot."""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kalshi Hedged Liquidity Provision Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live order placement (default is dry run)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=[],
        help="Specific market tickers to target",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="Total budget in dollars (default: $100)",
    )
    parser.add_argument(
        "--contracts",
        type=int,
        default=1,
        help="Contracts per order (default: 1)",
    )
    parser.add_argument(
        "--delta-max",
        type=int,
        default=3,
        help="Max acceptable overpayment per share in cents (default: 3)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=5.0,
        help="Poll interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--log-file",
        default="bot.log",
        help="Log file path (default: bot.log)",
    )
    parser.add_argument(
        "--key-id",
        default=None,
        help="Kalshi API key ID (overrides KALSHI_API_KEY_ID env var)",
    )
    parser.add_argument(
        "--key-path",
        default=None,
        help="Path to RSA private key (overrides KALSHI_PRIVATE_KEY_PATH env var)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger("main")

    # Build configuration
    config = BotConfig()
    config.dry_run = not args.live
    config.target_tickers = args.tickers
    config.capital.total_budget_cents = int(args.budget * 100)
    config.capital.contracts_per_order = args.contracts
    config.hedge.delta_max_cents = args.delta_max
    config.poll_interval_seconds = args.poll
    config.log_level = args.log_level
    config.log_file = args.log_file

    # Override API credentials if provided via CLI
    if args.key_id:
        config.kalshi.api_key_id = args.key_id
    if args.key_path:
        config.kalshi.private_key_path = args.key_path

    # Validate credentials
    if not config.kalshi.api_key_id:
        logger.error(
            "No API key ID provided. Set KALSHI_API_KEY_ID env var or use --key-id"
        )
        sys.exit(1)
    if not config.kalshi.private_key_path:
        logger.error(
            "No private key path provided. Set KALSHI_PRIVATE_KEY_PATH env var or use --key-path"
        )
        sys.exit(1)
    if not Path(config.kalshi.private_key_path).exists():
        logger.error(f"Private key file not found: {config.kalshi.private_key_path}")
        sys.exit(1)

    # Safety check for production
    if config.kalshi.is_production and not config.dry_run:
        logger.warning("=" * 60)
        logger.warning("  ⚠  PRODUCTION MODE — REAL MONEY AT RISK  ⚠")
        logger.warning("=" * 60)
        confirm = input("Type 'YES' to confirm live production trading: ")
        if confirm != "YES":
            logger.info("Aborted.")
            sys.exit(0)

    # Initialize client and bot
    try:
        client = KalshiClient(config.kalshi)
    except FileNotFoundError as e:
        logger.error(f"Cannot initialize client: {e}")
        sys.exit(1)

    bot = LiquidityBot(config, client)

    # Run
    logger.info("Starting bot...")
    bot.run()
    logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
