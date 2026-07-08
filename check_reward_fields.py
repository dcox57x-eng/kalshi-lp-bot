#!/usr/bin/env python3
"""
One-off probe: does the Kalshi market object expose liquidity-reward
eligibility / pool fields the bot could target?

Run from ~/Desktop/OPTIMUS with your env loaded, e.g.:
    source <(grep '^export' start.sh)
    python3 check_reward_fields.py

Read-only. Calls get_markets + get_market. Touches no orders, no balance.
"""

import json
import re

from config import KalshiConfig
from kalshi_client import KalshiClient, KalshiAPIError

# Anything whose key hints at the incentive program
HINT = re.compile(r"reward|incentiv|liquid|pool|eligib|rebate|maker|subsid", re.I)

SERIES = ["KXNBA", "KXNHL", "KXINX", "KXBTC", "KXETH"]


def flatten(obj, prefix=""):
    """Yield (dotted_key, value) for every leaf in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):  # sample first few list items
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def main():
    client = KalshiClient(KalshiConfig.from_env())

    all_keys = set()
    hint_hits = {}          # key -> sample value
    sample_ticker = None

    for series in SERIES:
        try:
            res = client.get_markets(limit=100, status="open", series_ticker=series)
        except KalshiAPIError as e:
            print(f"  {series}: fetch failed ({e})")
            continue
        markets = res.get("markets", [])
        print(f"  {series}: {len(markets)} open markets")
        for m in markets:
            if sample_ticker is None and m.get("ticker"):
                sample_ticker = m["ticker"]
            for key, val in flatten(m):
                all_keys.add(key)
                if HINT.search(key) and key not in hint_hits:
                    hint_hits[key] = val

    print("\n" + "=" * 60)
    print("REWARD / INCENTIVE FIELDS FOUND IN LIST RESPONSE")
    print("=" * 60)
    if hint_hits:
        for k in sorted(hint_hits):
            print(f"  {k} = {hint_hits[k]!r}")
    else:
        print("  (none — no obvious reward/eligibility keys in get_markets)")

    # The single-market endpoint is sometimes richer than the list — check it too.
    if sample_ticker:
        print("\n" + "=" * 60)
        print(f"FULL MARKET OBJECT — get_market('{sample_ticker}')")
        print("=" * 60)
        try:
            full = client.get_market(sample_ticker)
            print(json.dumps(full, indent=2, default=str))
            extra = {
                k: v for k, v in flatten(full)
                if HINT.search(k) and k.split(".", 1)[-1] not in {kk.split(".",1)[-1] for kk in hint_hits}
            }
            if extra:
                print("\n  Extra reward-ish fields only in single-market endpoint:")
                for k in sorted(extra):
                    print(f"    {k} = {extra[k]!r}")
        except KalshiAPIError as e:
            print(f"  get_market failed: {e}")

    print("\n" + "=" * 60)
    print(f"ALL TOP-LEVEL-ISH KEYS SEEN ({len(all_keys)} total)")
    print("=" * 60)
    print("  " + ", ".join(sorted(k for k in all_keys if "." not in k and "[" not in k)))


if __name__ == "__main__":
    main()
