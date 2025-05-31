#!/usr/bin/env python3
"""
FTMO-Compliant Signal Emitter - Stabilisierte Version
"""

import os
import time
import json
import logging
from pathlib import Path
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

class FTMOEmitter:
    """Stabilisierte Version des FTMO-Signal-Emitters"""
    
    # Signal types
    HEARTBEAT = 'heartbeat'
    TRADE_ENTRY = 'trade_entry'
    TRADE_EXIT = 'trade_exit'
    
    def __init__(
        self,
        signal_dir: Union[str, Path] = '/opt/coreflow/signals',
        log_dir: Union[str, Path] = '/opt/coreflow/logs',
        max_daily_trades: int = 10,
        max_risk_per_trade: float = 0.01,
        strategy_name: str = "FTMO_Stable"
    ):
        self.signal_dir = Path(signal_dir)
        self.log_dir = Path(log_dir)
        self.max_daily_trades = max_daily_trades
        self.max_risk_per_trade = max_risk_per_trade
        self.strategy_name = strategy_name
        self.today_trade_count = 0
        self.last_trade_day = None
        
        self._setup_logging()
        self._setup_directories()
        self.logger.info(f"Stable FTMO Emitter initialized - {strategy_name}")

    def _setup_directories(self) -> None:
        """Create required directories"""
        try:
            self.signal_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Directory error: {e}")
            raise

    def _setup_logging(self):
        """Configure reliable logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir/'ftmo_stable.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FTMO_Stable')

    def _reset_daily_counts(self):
        """Reset daily counters"""
        today = datetime.now(timezone.utc).date()
        if self.last_trade_day != today:
            self.today_trade_count = 0
            self.last_trade_day = today

    def create_signal(
        self,
        signal_type: str,
        instrument: str,
        price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        risk_percent: Optional[float] = None
    ) -> bool:
        """
        Stabilisierte Signal-Erstellung ohne Komprimierung
        """
        self._reset_daily_counts()
        
        # Validate trade signals
        if signal_type in [self.TRADE_ENTRY, self.TRADE_EXIT]:
            if self.today_trade_count >= self.max_daily_trades:
                self.logger.warning("Daily trade limit reached")
                return False
                
            if risk_percent and risk_percent > self.max_risk_per_trade:
                self.logger.warning(f"Risk exceeds {self.max_risk_per_trade*100}% limit")
                return False

        # Create signal data
        signal_id = f"FTMO_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        signal_data = {
            'id': signal_id,
            'type': signal_type,
            'timestamp': time.time(),
            'datetime': datetime.now(timezone.utc).isoformat(),
            'instrument': instrument,
            'price': price,
            'strategy': self.strategy_name,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_percent': risk_percent
        }

        try:
            signal_file = self.signal_dir / f"{signal_id}.json"
            with open(signal_file, 'w') as f:
                json.dump(signal_data, f, separators=(',', ':'))
            
            if signal_type in [self.TRADE_ENTRY, self.TRADE_EXIT]:
                self.today_trade_count += 1
                
            self.logger.info(f"Created {signal_type} signal for {instrument}")
            return True
            
        except Exception as e:
            self.logger.error(f"Signal creation failed: {e}")
            return False

    def run(self, interval: int = 30):
        """Stabilisierte Hauptschleife"""
        self.logger.info(f"Starting stable emitter (interval: {interval}s)")
        try:
            while True:
                # System heartbeat
                self.create_signal(
                    signal_type=self.HEARTBEAT,
                    instrument='SYSTEM',
                    price=0.0
                )
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Emitter stopped by user")
        except Exception as e:
            self.logger.error(f"Critical error: {e}")
            raise

if __name__ == "__main__":
    try:
        emitter = FTMOEmitter()
        
        # Beispiel Trade-Signal
        emitter.create_signal(
            signal_type=FTMOEmitter.TRADE_ENTRY,
            instrument='EURUSD',
            price=1.0850,
            stop_loss=1.0820,
            take_profit=1.0900,
            risk_percent=0.01
        )
        
        emitter.run()
        
    except Exception as e:
        logging.critical(f"Start failed: {e}")
