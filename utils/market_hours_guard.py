#!/usr/bin/env python3
# /opt/coreflow/utils/market_hours_guard.py – Institutional Edition

import datetime
import logging
from typing import Dict, List, Tuple
from enum import Enum, auto

class MarketType(Enum):
    FOREX = auto()
    METALS = auto()
    CRYPTO = auto()
    INDICES = auto()
    STOCKS = auto()

# Market hours per weekday (UTC): 0 = Monday, 6 = Sunday
MARKET_HOURS: Dict[MarketType, List[Tuple[int, int]]] = {
    MarketType.FOREX: [(0, 23)] * 5 + [(), ()],           # Mon–Fri
    MarketType.METALS: [(0, 23)] * 5 + [(), ()],          # Mon–Fri
    MarketType.CRYPTO: [(0, 24)] * 7,                     # 24/7
    MarketType.INDICES: [(2, 21)] * 5 + [(), ()],         # Mon–Fri 02:00–21:00
    MarketType.STOCKS: [(9, 17)] * 5 + [(), ()],          # Placeholder
}

SYMBOL_MAPPING = {
    'XAU': MarketType.METALS,
    'XAG': MarketType.METALS,
    'BTC': MarketType.CRYPTO,
    'ETH': MarketType.CRYPTO,
    'SPX': MarketType.INDICES,
    'NAS': MarketType.INDICES,
}

MARKET_HOLIDAYS = {
    datetime.date(2023, 12, 25),
    datetime.date(2024, 1, 1),
}

def get_market_type(symbol: str) -> MarketType:
    symbol = symbol.upper().replace('/', '')
    for prefix, market_type in SYMBOL_MAPPING.items():
        if symbol.startswith(prefix):
            return market_type
    return MarketType.FOREX

def is_market_open(market_type: MarketType, dt: datetime.datetime) -> bool:
    if dt.date() in MARKET_HOLIDAYS:
        return False
    weekday = dt.weekday()
    if weekday > 6 or weekday >= len(MARKET_HOURS[market_type]):
        return False
    open_ranges = MARKET_HOURS[market_type][weekday]
    if not open_ranges:
        return False
    start, end = open_ranges
    return start <= dt.hour < end

def is_trading_allowed(symbol: str, dt: datetime.datetime = None) -> bool:
    dt = dt or datetime.datetime.utcnow()
    market_type = get_market_type(symbol)
    if not is_market_open(market_type, dt):
        _log_block(symbol, dt, market_type)
        return False
    return True

def _log_block(symbol: str, dt: datetime.datetime, market_type: MarketType):
    logging.warning(
        f"🚫 Trading blocked for {symbol} ({market_type.name}) "
        f"at {dt.isoformat()} UTC (outside market hours)"
    )

def get_next_market_open(symbol: str, dt: datetime.datetime = None) -> datetime.datetime:
    dt = dt or datetime.datetime.utcnow()
    market_type = get_market_type(symbol)
    for i in range(1, 8):
        next_dt = dt + datetime.timedelta(days=i)
        weekday = next_dt.weekday()
        if next_dt.date() in MARKET_HOLIDAYS:
            continue
        if weekday >= len(MARKET_HOURS[market_type]):
            continue
        hours = MARKET_HOURS[market_type][weekday]
        if not hours:
            continue
        start, _ = hours
        next_open = datetime.datetime.combine(next_dt.date(), datetime.time(start, 0))
        return next_open
    return None

# Test block (optional)
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    test_symbols = ["EURUSD", "XAUUSD", "BTC/USD", "ETHUSD", "SPX500"]
    now = datetime.datetime.utcnow()
    print(f"\nMarket Status at {now.isoformat()}")
    print("=" * 50)
    for symbol in test_symbols:
        allowed = is_trading_allowed(symbol, now)
        next_open = get_next_market_open(symbol, now)
        status = "✅ ALLOWED" if allowed else "❌ BLOCKED"
        print(f"{symbol:<8} {status:<10} | Next open: {next_open}")
