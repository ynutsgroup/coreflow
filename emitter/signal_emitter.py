#!/usr/bin/env python3
"""
CoreFlow Institutional Trading Signal Emitter v9.0 - KORRIGIERTE VERSION
- KI/FTMO-konforme Validierung
- 100% modular & ohne Hardcode
- PUB/SUB-ZMQ-Implementierung
- Multi-Asset Support (Forex, Crypto, Metals, Indices)
"""

import os
import sys
import json
import redis
import zmq
import logging
import argparse
import re
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from enum import Enum, auto

# === Enums für klare Typisierung ===
class AssetType(Enum):
    FOREX = auto()
    CRYPTO = auto()
    METAL = auto()
    INDEX = auto()

class TradeAction(Enum):
    BUY = auto()
    SELL = auto()

# === Konfiguration ===
class AppConfig:
    def __init__(self):
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', '6380'))
        self.zmq_port = int(os.getenv('ZMQ_PORT', '5555'))  # ZMQ Standardport
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', '0.65'))
        self.lot_limits = self._load_lot_limits()
        
        self.asset_patterns = {
            AssetType.FOREX: re.compile(r'^[A-Z]{6}$'),
            AssetType.CRYPTO: re.compile(r'^(BTC|ETH|XRP)[A-Z]{3}$'),
            AssetType.METAL: re.compile(r'^X(AU|AG)[A-Z]{3}$'),
            AssetType.INDEX: re.compile(r'^(US30|NAS100|SPX500)')
        }

    def _load_lot_limits(self) -> Dict[AssetType, float]:
        try:
            limits = json.loads(os.getenv('LOT_LIMITS', '{}'))
            return {
                AssetType.FOREX: float(limits.get('FOREX', 10.0)),
                AssetType.CRYPTO: float(limits.get('CRYPTO', 5.0)),
                AssetType.METAL: float(limits.get('METAL', 15.0)),
                AssetType.INDEX: float(limits.get('INDEX', 7.0))
            }
        except Exception as e:
            logging.warning(f"Using default lot limits: {e}")
            return {
                AssetType.FOREX: 10.0,
                AssetType.CRYPTO: 5.0,
                AssetType.METAL: 15.0,
                AssetType.INDEX: 7.0
            }

# === Logging ===
logger = logging.getLogger("CF_Emitter")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s')
logger.addHandler(logging.StreamHandler())

# === ZMQ Publisher ===
class ZMQPublisher:
    """KORRIGIERT: PUB-Socket für Publisher"""
    def __init__(self, port: int):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)  # PUB-Socket
        self._bind()

    def _bind(self):
        try:
            self.socket.bind(f"tcp://*:{self.port}")
            logger.info(f"ZMQ PUB bound to port {self.port}")
        except zmq.ZMQError as e:
            logger.error(f"ZMQ bind failed: {e}")
            raise

    def publish(self, message: Dict[str, Any]):
        try:
            self.socket.send_string(json.dumps(message))
            logger.debug(f"Published to ZMQ: {message}")
        except Exception as e:
            logger.error(f"ZMQ publish error: {e}")

    def __del__(self):
        self.socket.close()
        self.context.term()

# === Signal Validierung ===
class SignalValidator:
    """Unverändert"""
    def __init__(self, config: AppConfig):
        self.config = config

    def validate(self, symbol: str, action: str, lot: float, confidence: float) -> Dict[str, Any]:
        symbol = symbol.upper().strip()
        action = action.upper().strip()
        asset_type = self._classify_asset(symbol)
        
        self._validate_action(action)
        self._validate_confidence(confidence)
        self._validate_lot_size(lot, asset_type)

        return {
            'symbol': symbol,
            'action': action,
            'lot': round(lot, 2),
            'confidence': round(confidence, 2),
            'asset_type': asset_type.name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'system': 'CoreFlow',
            'version': '9.0'
        }

    def _classify_asset(self, symbol: str) -> AssetType:
        for asset_type, pattern in self.config.asset_patterns.items():
            if pattern.match(symbol):
                return asset_type
        return AssetType.FOREX

    def _validate_action(self, action: str):
        if action not in [a.name for a in TradeAction]:
            raise ValueError(f"Invalid action: {action}")

    def _validate_confidence(self, confidence: float):
        if not (self.config.min_confidence <= confidence <= 1.0):
            raise ValueError(f"Confidence {confidence} out of range")

    def _validate_lot_size(self, lot: float, asset_type: AssetType):
        max_lot = self.config.lot_limits.get(asset_type, 1.0)
        if not (0.01 <= lot <= max_lot):
            raise ValueError(f"Lot size {lot} exceeds {asset_type.name} limit")

# === Hauptlogik ===
def parse_args():
    parser = argparse.ArgumentParser(description='CoreFlow Signal Emitter')
    parser.add_argument('--symbol', required=True, help='Trading symbol')
    parser.add_argument('--action', required=True, choices=['BUY', 'SELL'])
    parser.add_argument('--lot', type=float, required=True)
    parser.add_argument('--confidence', type=float, required=True)
    parser.add_argument('--sl', type=float, help='Stop Loss')
    parser.add_argument('--tp', type=float, help='Take Profit')
    return parser.parse_args()

def main():
    args = parse_args()
    config = AppConfig()
    
    # Initialisiere Komponenten
    zmq_pub = ZMQPublisher(config.zmq_port)  # ZMQ Publisher
    validator = SignalValidator(config)

    try:
        # Validierung
        signal = validator.validate(
            symbol=args.symbol,
            action=args.action,
            lot=args.lot,
            confidence=args.confidence
        )

        # Optionale Felder
        if args.sl: signal['sl'] = args.sl
        if args.tp: signal['tp'] = args.tp

        # Publikation
        zmq_pub.publish(signal)  # ZMQ
        logger.info(f"Signal published: {signal}")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
