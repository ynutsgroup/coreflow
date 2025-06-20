#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VALG Hybrid Engine (CPU-Modus) – FTMO-konform, Redis+ZMQ aktiviert
"""

import asyncio
import logging
import zmq
import redis
import numpy as np
from datetime import datetime
from typing import Dict, List
import json
import aiohttp
import ssl
from dataclasses import dataclass
from dotenv import load_dotenv
import os

# === Load .env config ===
load_dotenv()

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/opt/coreflow/logs/valg_engine.log')
    ]
)
logger = logging.getLogger("VALG.CPU")

# === CONFIG ===
@dataclass
class CrossPlatformConfig:
    zmq_host: str = os.getenv("ZMQ_HOST", "tcp://*:5555")
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    mt5_api_url: str = os.getenv("MT5_API_URL", "https://windows-mt5-server:8443/api")
    mt5_api_key: str = os.getenv("MT5_API_KEY", "secure-key")

# === CPU EMA Analyzer ===
class CPUMarketAnalyzer:
    @staticmethod
    def ema(array: np.ndarray, period: int) -> np.ndarray:
        ema = np.zeros_like(array)
        multiplier = 2 / (period + 1)
        ema[0] = array[0]
        for i in range(1, len(array)):
            ema[i] = (array[i] - ema[i - 1]) * multiplier + ema[i - 1]
        return ema

    def analyze(self, symbol_data: Dict[str, np.ndarray]) -> Dict:
        results = {}
        for symbol, data in symbol_data.items():
            closes = np.array(data['close'])
            ema_50 = self.ema(closes, 50)
            ema_200 = self.ema(closes, 200)
            results[symbol] = {
                'ema_50': float(ema_50[-1]),
                'ema_200': float(ema_200[-1]),
                'trend': 'UP' if ema_50[-1] > ema_200[-1] else 'DOWN'
            }
        return results

# === MT5 API Bridge ===
class MT5WindowsBridge:
    def __init__(self, config: CrossPlatformConfig):
        self.base_url = config.mt5_api_url
        self.api_key = config.mt5_api_key
        self.ssl_context = ssl.create_default_context()
        self.session = aiohttp.ClientSession()

    async def execute_order(self, signal: Dict) -> Dict:
        url = f"{self.base_url}/order"
        headers = {"X-API-KEY": self.api_key}
        async with self.session.post(url, json=signal, headers=headers, ssl=self.ssl_context) as resp:
            if resp.status == 200:
                return await resp.json()
            raise Exception(f"MT5 API ERROR: {await resp.text()}")

# === VALG Core Engine ===
class VALGHybridEngine:
    def __init__(self, config: CrossPlatformConfig):
        self.config = config
        self.analyzer = CPUMarketAnalyzer()
        self.mt5 = MT5WindowsBridge(config)
        self._init_comm()

    def _init_comm(self):
        ctx = zmq.Context()
        self.publisher = ctx.socket(zmq.PUB)
        self.publisher.bind(self.config.zmq_host)
        self.redis = redis.Redis(host=self.config.redis_host, decode_responses=True)

    def generate_signals(self, analysis: Dict) -> List[Dict]:
        signals = []
        for symbol, data in analysis.items():
            if data['trend'] == 'UP':
                signals.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "volume": 0.1,
                    "timestamp": datetime.utcnow().isoformat(),
                    "comment": "VALG_CPU"
                })
        return signals

    async def process_market_data(self, market_data: Dict):
        try:
            result = self.analyzer.analyze(market_data)
            signals = self.generate_signals(result)
            for signal in signals:
                execution = await self.mt5.execute_order(signal)
                self.redis.xadd("execution_stream", {k: json.dumps(v) for k, v in execution.items()})
                self.publisher.send_string(json.dumps(execution))
                logger.info(f"📤 Order veröffentlicht: {execution.get('order_id', 'N/A')}")
        except Exception as e:
            logger.error(f"❌ VALG Fehler: {e}", exc_info=True)

# === Main Loop ===
async def main():
    config = CrossPlatformConfig()
    valg = VALGHybridEngine(config)

    # 🔁 Beispiel-Datenstream
    dummy_data = {
        "EURUSD": {
            "close": np.random.normal(1.08, 0.005, 200).cumsum(),
            "volume": np.random.randint(100, 1000, 200)
        }
    }

    try:
        while True:
            await valg.process_market_data(dummy_data)
            await asyncio.sleep(30)
    except KeyboardInterrupt:
        logger.info("⏹️ Manuell gestoppt")
    finally:
        await valg.mt5.session.close()

if __name__ == "__main__":
    asyncio.run(main())
