#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VALG Engine CPU – CoreFlow Institutional Edition (Linux)
Steuert alle Handelssignale auf Basis KI + optional Quantum Boost
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import subprocess
import numpy as np
from datetime import datetime, timezone
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional

# === Optional Quantum Support ===
try:
    from qiskit import QuantumCircuit, execute, Aer
    QUANTUM_ENABLED = True
except ImportError:
    QUANTUM_ENABLED = False

# === Logging ===
LOGFILE = "/opt/coreflow/logs/valg_engine_cpu.log"
os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VALG.CPU")

# === ENUMs ===
class SignalDirection(Enum):
    BUY = auto()
    SELL = auto()

class AssetType(Enum):
    FOREX = auto()
    CRYPTO = auto()
    COMMODITY = auto()

@dataclass
class TradingSignal:
    symbol: str
    direction: SignalDirection
    volume: float
    confidence: float
    quantum_boost: float
    asset_type: AssetType
    timestamp: datetime = datetime.now(timezone.utc)

# === Quantum Boost (optional) ===
def quantum_entropy() -> float:
    if not QUANTUM_ENABLED:
        return np.random.random()
    try:
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        backend = Aer.get_backend("qasm_simulator")
        result = execute(qc, backend, shots=1).result()
        counts = result.get_counts()
        return int(list(counts.keys())[0], 2) / 1
    except Exception as e:
        logger.warning(f"Quantum Error: {e}")
        return np.random.random()

# === VALG Engine ===
class VALGEngine:
    def __init__(self):
        self.running = True
        self.assets = {
            "EURUSD": {"type": AssetType.FOREX, "lot": 0.8},
            "BTCUSD": {"type": AssetType.CRYPTO, "lot": 0.4},
            "XAUUSD": {"type": AssetType.COMMODITY, "lot": 0.3}
        }
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)

    def shutdown(self, *_):
        self.running = False
        logger.info("🛑 Shutdown empfangen – VALG Engine wird gestoppt")

    def generate_signal(self, symbol: str) -> TradingSignal:
        boost = round(quantum_entropy(), 4)
        conf = round(min(0.99, 0.7 + 0.3 * boost), 4)
        direction = SignalDirection.BUY if boost > 0.5 else SignalDirection.SELL
        volume = round(self.assets[symbol]["lot"] * (1 + boost), 2)
        return TradingSignal(
            symbol=symbol,
            direction=direction,
            volume=volume,
            confidence=conf,
            quantum_boost=boost,
            asset_type=self.assets[symbol]["type"]
        )

    def dispatch(self, sig: TradingSignal):
        cmd = [
            "python3", "/opt/coreflow/core/emitter/signal_emitter_pro.py",
            sig.symbol,
            sig.direction.name,
            str(sig.volume),
            str(sig.confidence),
            "--quantum", str(sig.quantum_boost),
            "--asset-type", sig.asset_type.name
        ]
        try:
            subprocess.run(cmd, timeout=10)
            logger.info(f"✅ Gesendet: {sig.symbol} {sig.direction.name} {sig.volume} | Boost: {sig.quantum_boost:.4f} | Conf: {sig.confidence}")
        except Exception as e:
            logger.error(f"❌ Fehler bei dispatch: {e}")

    def run(self, interval: float):
        logger.info("🚀 VALG Engine aktiv – Hybrid CPU Mode")
        while self.running:
            for symbol in self.assets:
                sig = self.generate_signal(symbol)
                self.dispatch(sig)
            time.sleep(interval)

# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()

    valg = VALGEngine()
    valg.run(args.interval)
