#!/usr/bin/env python3
"""
Probe: what liquidity-incentive pools are LIVE right now, and how crowded?

Hits GET /incentive_programs?status=active, keeps the liquidity ones, ranks by
pool size, and for the top markets peeks at the book to gauge how much
qualifying size is already resting (your payout = your share of that size).

Run from ~/Desktop/OPTIMUS with env loaded:
    source <(grep '^export' start.sh)
    python3 check_incentives.py

Read-only. No orders, no balance access.
"""

import json

from config import KalshiConfig
from kalshi_client import KalshiClient, KalshiAPIError

TOP_N_BOOK_PEEK = 8   # how many top pools to inspect the book for


def fetch_active_incentives(client):
    """Page through active incentive programs."""
    programs, cursor = [], None
    for _ in range(10):  # safety cap on pagination
        params = {"status": "active", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        res = client._request("GET", "/incentive_programs", params=params)
        programs.extend(res.get("incentive_programs", []))
        cursor = res.get("next_cursor")
        if not cursor:
            break
    return programs


def best_level(client, ticker):
    """Return (best_yes_bid_$, yes_size, best_no_bid_$, no_size) or Nones."""
    try:
        ob = client.get_orderbook(ticker, depth=1).get("orderbook_fp", {})
    except KalshiAPIError:
        return (None, None, None, None)
    y = (ob.get("yes_dollars") or [])
    n = (ob.get("no_dollars") or [])
    yb = y[-1] if y else None   # best (highest) bid is last after API's ascending sort
    nb = n[-1] if n else None
    return (
        yb[0] if yb else None, yb[1] if yb else None,
        nb[0] if nb else None, nb[1] if nb else None,
    )


def main():
    client = KalshiClient(KalshiConfig.from_env())

    programs = fetch_active_incentives(client)
    liq = [p for p in programs if p.get("incentive_type") == "liquidity"]

    print(f"Active incentive programs: {len(programs)} total, {len(liq)} liquidity\n")
    if not liq:
        print("No active LIQUIDITY incentives right now. Either the program is")
        print("between periods, or pools are currently volume-type only.")
        # Show one raw program so we can see the real field shape regardless.
        if programs:
            print("\nSample program object:")
            print(json.dumps(programs[0], indent=2, default=str))
        return

    # Rank by pool size
    liq.sort(key=lambda p: p.get("period_reward", 0), reverse=True)

    print("=" * 78)
    print(f"{'POOL':>8}  {'TARGET':>8}  {'TICKER':<32}  ENDS")
    print("=" * 78)
    for p in liq:
        print(f"{p.get('period_reward', 0):>8}  "
              f"{p.get('target_size_fp', '?'):>8}  "
              f"{p.get('market_ticker', '?'):<32}  "
              f"{p.get('end_date', '?')}")

    print("\n" + "=" * 78)
    print(f"BOOK PEEK — top {TOP_N_BOOK_PEEK} pools (is the qualifying size crowded?)")
    print("=" * 78)
    for p in liq[:TOP_N_BOOK_PEEK]:
        t = p.get("market_ticker", "")
        yb_p, yb_s, nb_p, nb_s = best_level(client, t)
        print(f"\n{t}  pool={p.get('period_reward')}  target_size={p.get('target_size_fp')}")
        print(f"   best YES bid: ${yb_p} x {yb_s}     best NO bid: ${nb_p} x {nb_s}")

    print("\n" + "=" * 78)
    print("Full first liquidity program object (field reference):")
    print(json.dumps(liq[0], indent=2, default=str))


if __name__ == "__main__":
    main()
