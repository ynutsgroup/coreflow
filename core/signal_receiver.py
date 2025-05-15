#!/usr/bin/env python3
"""FTMO-konformer Signal Emitter für Trading-Systeme"""

import json
import logging
import sys
import time
from typing import Dict, Any, Literal

import redis
from core.config import Config

# FTMO-spezifische Konstanten
MAX_RISK_PERCENT = 1.0  # Maximal 1% Risiko pro Trade
TRADE_ACTIONS = Literal["BUY", "SELL", "HOLD"]

class FTMOEmitter:
    def __init__(self):
        self.logger = self._setup_ftmo_logging()
        self.redis = self._init_secure_redis()
        self.trade_count = 0

    def _setup_ftmo_logging(self) -> logging.Logger:
        """FTMO-konforme Logging-Konfiguration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s|%(levelname)s|%(message)s',
            handlers=[
                logging.FileHandler(f"{Config.LOG_DIR}/ftmo_emitter.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger('FTMO_Emitter')

    def _init_secure_redis(self) -> redis.Redis:
        """Sichere Redis-Verbindung mit FTMO-Anforderungen"""
        return redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            password=Config.REDIS_PASSWORD,
            ssl=True,
            ssl_cert_reqs='required',
            socket_timeout=5,
            socket_connect_timeout=3,
            decode_responses=False  # Sicherheit: Rohbytes
        )

    def _validate_ftmo_rules(self, action: str, price: float) -> bool:
        """FTMO-Regelvalidierung"""
        if action not in TRADE_ACTIONS.__args__:
            self.logger.error(f"Invalid FTMO action: {action}")
            return False
            
        if price <= 0:
            self.logger.error("Price must be positive")
            return False
            
        if self.trade_count >= 100:  # FTMO Max Trades/Day
            self.logger.warning("FTMO trade limit reached")
            return False
            
        return True

    def _create_ftmo_message(self, symbol: str, action: str, price: float) -> Dict[str, Any]:
        """FTMO-konforme Nachrichtenstruktur"""
        return {
            "version": "1.0",
            "account": Config.FTMO_ACCOUNT_ID,
            "symbol": symbol.upper(),
            "action": action,
            "price": round(float(price), 5),
            "timestamp": int(time.time()),
            "risk": MAX_RISK_PERCENT,
            "metadata": {
                "source": "coreflow",
                "compliance": "ftmo-v2"
            }
        }

    def emit_signal(self, symbol: str, action: str, price: str) -> bool:
        """
        Sendet FTMO-konformes Signal
        
        Args:
            symbol: Trading-Symbol (z.B. 'EURUSD')
            action: BUY/SELL/HOLD
            price: Ausführungspreis
            
        Returns:
            bool: True wenn FTMO-konform gesendet
        """
        try:
            price_float = float(price)
            if not self._validate_ftmo_rules(action, price_float):
                return False

            message = self._create_ftmo_message(symbol, action, price_float)
            serialized = json.dumps(message).encode('utf-8')
            
            self.redis.publish(Config.REDIS_CHANNEL, serialized)
            self.trade_count += 1
            
            self.logger.info(
                "FTMO_SIGNAL|%s|%s|%.5f|%d",
                message['symbol'], 
                message['action'],
                message['price'],
                message['timestamp']
            )
            return True
            
        except ValueError:
            self.logger.error("Invalid price format")
        except redis.RedisError as e:
            self.logger.error(f"Redis error: {str(e)}")
        except Exception as e:
            self.logger.critical(f"Unexpected error: {str(e)}")
        
        return False

def main():
    if len(sys.argv) != 4:
        print("Usage: ftmo_emitter.py <SYMBOL> <ACTION> <PRICE>")
        sys.exit(1)

    emitter = FTMOEmitter()
    success = emitter.emit_signal(sys.argv[1], sys.argv[2], sys.argv[3])
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
