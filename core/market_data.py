#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow Institutional Market Data Provider
FTMO-compliant real-time market data module with enhanced error handling
"""

import logging
import MetaTrader5 as mt5
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import threading
import queue
from dataclasses import dataclass
import pytz

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('/opt/coreflow/logs/market_data.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class TickData:
    symbol: str
    bid: float
    ask: float
    time: datetime
    spread: float
    volume: int

class MarketDataProvider:
    def __init__(self):
        """Institutional-grade MT5 market data provider"""
        self.logger = logging.getLogger("CF.MarketData")
        self._shutdown_flag = False
        self._tick_queue = queue.Queue(maxsize=1000)
        self._subscriptions = set()
        self._last_tick_time = {}
        
        # FTMO compliance parameters
        self.MAX_RECONNECT_ATTEMPTS = 3
        self.RECONNECT_DELAY = 5
        self.TICK_TIMEOUT = timedelta(seconds=30)
        
        self._initialize_mt5()

    def _initialize_mt5(self) -> bool:
        """Secure MT5 initialization with retry logic"""
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            try:
                if not mt5.initialize():
                    raise ConnectionError(f"MT5 returned initialize()=False")
                
                self.logger.info("✅ MT5 Connected | Version: %s", mt5.version())
                return True
                
            except Exception as e:
                self.logger.error("⚠️ Connection attempt %d/%d failed: %s", 
                                attempt, self.MAX_RECONNECT_ATTEMPTS, str(e))
                if attempt < self.MAX_RECONNECT_ATTEMPTS:
                    time.sleep(self.RECONNECT_DELAY)
        
        self.logger.critical("❌ Failed to initialize MT5 after %d attempts", 
                           self.MAX_RECONNECT_ATTEMPTS)
        raise RuntimeError("MT5 initialization failed")

    def start_tick_listener(self):
        """Start background thread for tick monitoring"""
        self._shutdown_flag = False
        threading.Thread(
            target=self._tick_monitor,
            daemon=True,
            name="TickMonitor"
        ).start()
        self.logger.info("📡 Started tick monitoring thread")

    def _tick_monitor(self):
        """Background thread monitoring subscribed symbols"""
        while not self._shutdown_flag:
            try:
                current_time = datetime.now(pytz.UTC)
                
                # Check for stale ticks
                for symbol, last_time in list(self._last_tick_time.items()):
                    if current_time - last_time > self.TICK_TIMEOUT:
                        self.logger.warning("⌛ Tick timeout for %s (last update: %s)", 
                                          symbol, last_time.isoformat())
                        self._last_tick_time.pop(symbol)
                
                # Process ticks for subscribed symbols
                for symbol in list(self._subscriptions):
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue
                        
                    tick_time = datetime.fromtimestamp(tick.time, pytz.UTC)
                    self._last_tick_time[symbol] = tick_time
                    
                    tick_data = TickData(
                        symbol=symbol,
                        bid=tick.bid,
                        ask=tick.ask,
                        time=tick_time,
                        spread=(tick.ask - tick.bid) * 10_000,
                        volume=tick.volume
                    )
                    
                    try:
                        self._tick_queue.put_nowait(tick_data)
                    except queue.Full:
                        self.logger.warning("⚠️ Tick queue full - dropping tick for %s", symbol)
                
                time.sleep(0.1)  # 10ms interval
                
            except Exception as e:
                self.logger.error("‼️ Tick monitor error: %s", str(e), exc_info=True)
                time.sleep(1)

    def subscribe(self, symbol: str) -> bool:
        """Subscribe to market data for a symbol"""
        if not mt5.symbol_select(symbol, True):
            self.logger.error("❌ Failed to subscribe to %s", symbol)
            return False
            
        self._subscriptions.add(symbol)
        self.logger.info("➕ Subscribed to %s", symbol)
        return True

    def unsubscribe(self, symbol: str):
        """Unsubscribe from market data"""
        if symbol in self._subscriptions:
            self._subscriptions.remove(symbol)
            self.logger.info("➖ Unsubscribed from %s", symbol)

    def get_next_tick(self, timeout: float = 1.0) -> Optional[TickData]:
        """Get next tick data from queue (thread-safe)"""
        try:
            return self._tick_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_current_price(self, symbol: str) -> Optional[TickData]:
        """Get current price (thread-safe)"""
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
            
        return TickData(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            time=datetime.fromtimestamp(tick.time, pytz.UTC),
            spread=(tick.ask - tick.bid) * 10_000,
            volume=tick.volume
        )

    def shutdown(self):
        """Graceful shutdown procedure"""
        self._shutdown_flag = True
        mt5.shutdown()
        self.logger.info("🔌 MT5 connection terminated")

    # FTMO Compliance Methods
    def get_market_state(self) -> Dict:
        """Get comprehensive market state for compliance"""
        return {
            'server_time': datetime.now(pytz.UTC),
            'connected': mt5.terminal_info().connected,
            'symbols_subscribed': list(self._subscriptions),
            'tick_quality': {sym: (datetime.now(pytz.UTC) - t).total_seconds() 
                           for sym, t in self._last_tick_time.items()}
        }

    def verify_spread(self, symbol: str, max_spread: float) -> bool:
        """FTMO spread verification"""
        tick = self.get_current_price(symbol)
        if not tick:
            return False
        return tick.spread <= max_spread

# Example usage
if __name__ == "__main__":
    try:
        mdp = MarketDataProvider()
        mdp.subscribe("EURUSD")
        mdp.subscribe("GBPUSD")
        mdp.start_tick_listener()
        
        while True:
            tick = mdp.get_next_tick()
            if tick:
                print(f"[{tick.time}] {tick.symbol} "
                     f"Bid: {tick.bid:.5f} Ask: {tick.ask:.5f} "
                     f"Spread: {tick.spread:.1f}pips")
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        mdp.shutdown()
