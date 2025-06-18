#!/usr/bin/env python3
"""
Trading Signal Emitter mit Redis-Integration für Zustandsprüfungen
"""

import os
import time
import json
import logging
import redis
from pathlib import Path
import uuid
from typing import Any, Dict, Optional

class TradingSignalEmitter:
    """Erzeugt Handels-Signale mit Redis-Integration"""
    
    # Signaltypen
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
        redis_db: int = 0
    ):
        self.signal_dir = Path(signal_dir)
        self.log_dir = Path(log_dir)
        self.redis = redis.StrictRedis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True
        )
        self._setup_logging()
        self._setup_directories()
        self._check_redis_connection()
        
    def _check_redis_connection(self):
        """Überprüft die Redis-Verbindung"""
        try:
            if self.redis.ping():
                self.logger.info("Erfolgreich mit Redis verbunden")
        except redis.ConnectionError as e:
            self.logger.error(f"Redis-Verbindungsfehler: {e}")
            raise

    def _setup_directories(self):
        """Erstellt benötigte Verzeichnisse"""
        try:
            self.signal_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
            self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        except Exception as e:
            self.logger.error(f"Verzeichnis konnte nicht erstellt werden: {e}")
            raise

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

    def _check_market_conditions(self, asset: str) -> bool:
        """
        Überprüft Marktbedingungen in Redis
        Gibt True zurück wenn Handelsbedingungen erfüllt sind
        """
        try:
            # Beispiel: Prüfe ob Preis innerhalb der letzten Stunde um >2% gestiegen ist
            last_price = float(self.redis.get(f"price:{asset}") or 0)
            hour_ago_price = float(self.redis.get(f"price:{asset}:1h") or 0)
            
            if hour_ago_price == 0:
                return False
                
            price_change = (last_price - hour_ago_price) / hour_ago_price
            return abs(price_change) > 0.02  # 2% Änderung
            
        except Exception as e:
            self.logger.error(f"Fehler bei Marktbedingungsprüfung: {e}")
            return False

    def _get_current_price(self, asset: str) -> float:
        """Holt den aktuellen Preis aus Redis"""
        try:
            return float(self.redis.get(f"price:{asset}") or 0.0)
        except Exception as e:
            self.logger.error(f"Fehler beim Preisabruf: {e}")
            return 0.0

    def create_trading_signal(
        self,
        signal_type: str,
        asset: str,
        price: Optional[float] = None,
        reason: str = "",
        strategy: str = ""
    ) -> Optional[Path]:
        """
        Erzeugt ein Handels-Signal mit Redis-Validierung
        
        Args:
            signal_type: BUY oder SELL
            asset: Handels-Paar (z.B. 'BTC-USD')
            price: Optionaler Preis (wenn nicht angegeben, wird er aus Redis geholt)
            reason: Grund für das Signal
            strategy: Name der Handelsstrategie
            
        Returns:
            Pfad zur Signaldatei oder None bei Fehlern
        """
        if signal_type not in [self.BUY, self.SELL, self.HEARTBEAT, self.ALERT]:
            self.logger.error(f"Ungültiger Signaltyp: {signal_type}")
            return None
            
        # Für Heartbeat-Signale überspringen wir die Marktprüfung
        if signal_type != self.HEARTBEAT and not self._check_market_conditions(asset):
            self.logger.debug(f"Marktbedingungen nicht erfüllt für {asset}")
            return None
            
        current_price = price if price is not None else self._get_current_price(asset)
        if current_price <= 0 and signal_type != self.HEARTBEAT:
            self.logger.error("Ungültiger Preis für Handels-Signal")
            return None
            
        signal_id = f"TRADE_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        signal_file = self.signal_dir / f"{signal_id}.json"
        
        signal_data = {
            'id': signal_id,
            'type': signal_type,
            'timestamp': time.time(),
            'asset': asset,
            'price': current_price,
            'reason': reason,
            'strategy': strategy,
            'conditions_met': True,
            'redis_checked': True
        }
        
        try:
            with open(signal_file, 'w') as f:
                json.dump(signal_data, f, indent=2)
            
            # Signal in Redis protokollieren
            self.redis.lpush('trading:signals', json.dumps(signal_data))
            self.redis.expire('trading:signals', 86400)  # Behalte 24 Stunden
            
            self.logger.info(f"Signal erzeugt: {signal_type} {asset} bei {current_price}")
            return signal_file
            
        except Exception as e:
            self.logger.error(f"Signal konnte nicht gespeichert werden: {e}")
            return None

    def run(self, interval: int = 10):
        """Haupt-Schleife mit Redis-Überwachung"""
        self.logger.info("Trading Signal Emitter gestartet mit Redis-Integration")
        
        try:
            while True:
                # Heartbeat-Signal
                self.create_trading_signal(
                    signal_type=self.HEARTBEAT,
                    asset='SYSTEM',
                    reason='system_check',
                    strategy='monitoring'
                )
                
                # Markt-Signale für verschiedene Assets
                for asset in ['BTC-USD', 'ETH-USD', 'XRP-USD']:
                    price = self._get_current_price(asset)
                    
                    # Beispiel-Logik: Kaufsignal wenn Preis unter dem 24h-Durchschnitt
                    day_avg = float(self.redis.get(f"price:{asset}:24h_avg") or 0)
                    if day_avg > 0 and price > 0:
                        if price < day_avg * 0.98:  # 2% unter Durchschnitt
                            self.create_trading_signal(
                                signal_type=self.BUY,
                                asset=asset,
                                reason='below_daily_average',
                                strategy='mean_reversion'
                            )
                        elif price > day_avg * 1.02:  # 2% über Durchschnitt
                            self.create_trading_signal(
                                signal_type=self.SELL,
                                asset=asset,
                                reason='above_daily_average',
                                strategy='mean_reversion'
                            )
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Trading Signal Emitter gestoppt")
        except Exception as e:
            self.logger.error(f"Kritischer Fehler: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    try:
        emitter = TradingSignalEmitter(
            redis_host='redis',  # Docker Service Name oder Host
            redis_port=6379
        )
        emitter.run(interval=15)  # 15 Sekunden Intervall
    except Exception as e:
        logging.critical(f"Start fehlgeschlagen: {e}", exc_info=True)
        raise

