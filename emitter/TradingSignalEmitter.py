#!/usr/bin/env python3
"""
TradingSignalEmitter mit vollständiger Implementierung
"""

import os
import time
import json
import logging
import argparse
from pathlib import Path
import uuid
from typing import Any, Dict, Optional, Union

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis-Python-Bibliothek nicht verfügbar")

class TradingSignalEmitter:
    """Handels-Signal Emitter mit vollständiger Implementierung"""
    
    HEARTBEAT = 'heartbeat'
    BUY = 'buy'
    SELL = 'sell'
    ALERT = 'alert'
    
    def __init__(
        self,
        signal_dir: str = '/opt/coreflow/signals',
        log_dir: str = '/opt/coreflow/logs',
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 0,
        require_redis: bool = False
    ):
        self.signal_dir = Path(signal_dir)
        self.log_dir = Path(log_dir)
        self.redis_connected = False
        self.require_redis = require_redis
        
        # Methodenaufrufe in korrekter Reihenfolge
        self._setup_logging()  # Zuerst Logging initialisieren
        self._setup_directories()
        self._init_redis(redis_host, redis_port, redis_db)

    def _setup_logging(self):
        """Konfiguriert die Log-Ausgabe"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir/'trading_signal_emitter.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('TradingSignalEmitter')
        self.logger.info("Logging erfolgreich initialisiert")

    def _setup_directories(self):
        """Erstellt benötigte Verzeichnisse"""
        try:
            self.signal_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
            self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
            self.logger.info(f"Verzeichnisse erstellt: {self.signal_dir}, {self.log_dir}")
        except Exception as e:
            self.logger.error(f"Verzeichnis konnte nicht erstellt werden: {e}")
            raise

    def _init_redis(self, host: str, port: int, db: int):
        """Initialisiert die Redis-Verbindung"""
        if not REDIS_AVAILABLE:
            if self.require_redis:
                raise ImportError("Redis-Python-Bibliothek benötigt aber nicht verfügbar")
            self.logger.warning("Redis-Unterstützung deaktiviert")
            return

        try:
            self.redis = redis.StrictRedis(
                host=host,
                port=port,
                db=db,
                socket_timeout=5,
                socket_connect_timeout=5,
                decode_responses=True
            )
            if self.redis.ping():
                self.redis_connected = True
                self.logger.info(f"Erfolgreich mit Redis verbunden ({host}:{port})")
            elif self.require_redis:
                raise ConnectionError(f"Redis-Verbindung fehlgeschlagen ({host}:{port})")
        except Exception as e:
            self.logger.warning(f"Redis-Verbindungsfehler: {e}")
            if self.require_redis:
                raise

    def create_trading_signal(
        self,
        signal_type: str,
        asset: str,
        price: Optional[float] = None,
        reason: str = "",
        strategy: str = ""
    ) -> Optional[Path]:
        """
        Erzeugt ein Handels-Signal
        """
        if signal_type not in [self.BUY, self.SELL, self.HEARTBEAT, self.ALERT]:
            self.logger.error(f"Ungültiger Signaltyp: {signal_type}")
            return None

        signal_id = f"TRADE_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        signal_file = self.signal_dir / f"{signal_id}.json"
        
        signal_data = {
            'id': signal_id,
            'type': signal_type,
            'timestamp': time.time(),
            'asset': asset,
            'price': price or 0.0,
            'reason': reason,
            'strategy': strategy,
            'redis_used': self.redis_connected
        }
        
        try:
            with open(signal_file, 'w') as f:
                json.dump(signal_data, f, indent=2)
            
            if self.redis_connected:
                try:
                    self.redis.lpush('trading:signals', json.dumps(signal_data))
                    self.redis.expire('trading:signals', 86400)
                except Exception as e:
                    self.logger.warning(f"Redis-Speicherung fehlgeschlagen: {e}")
            
            self.logger.info(f"Signal erzeugt: {signal_type} {asset}")
            return signal_file
            
        except Exception as e:
            self.logger.error(f"Signal konnte nicht gespeichert werden: {e}")
            return None

    def run(self, interval: int = 10):
        """Haupt-Schleife des Signal Emitters"""
        self.logger.info(f"Signal Emitter gestartet (Interval: {interval}s)")
        
        try:
            while True:
                # Heartbeat-Signal
                self.create_trading_signal(
                    signal_type=self.HEARTBEAT,
                    asset='SYSTEM',
                    reason='system_check'
                )
                
                # Beispiel-Signale
                if not self.redis_connected:
                    self._generate_sample_signals()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Signal Emitter gestoppt")
        except Exception as e:
            self.logger.error(f"Kritischer Fehler: {e}", exc_info=True)
            raise

    def _generate_sample_signals(self):
        """Erzeugt Beispiel-Signale wenn Redis nicht verfügbar ist"""
        sample_assets = ['BTC-USD', 'ETH-USD', 'XRP-USD']
        for asset in sample_assets:
            signal_type = self.BUY if time.time() % 2 == 0 else self.SELL
            self.create_trading_signal(
                signal_type=signal_type,
                asset=asset,
                price=50000 + (time.time() % 1000),
                reason='simulated_signal',
                strategy='sample_data'
            )

def parse_args():
    """Kommandozeilenargumente parsen"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--redis-host', default='localhost', help='Redis Server Host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis Server Port')
    parser.add_argument('--interval', type=int, default=10, help='Signal-Intervall in Sekunden')
    parser.add_argument('--require-redis', action='store_true', help='Redis als zwingende Voraussetzung')
    return parser.parse_args()

if __name__ == "__main__":
    try:
        args = parse_args()
        emitter = TradingSignalEmitter(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            require_redis=args.require_redis
        )
        emitter.run(interval=args.interval)
    except Exception as e:
        logging.critical(f"Start fehlgeschlagen: {e}", exc_info=True)
        raise
