#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VALG Multi-Asset Engine Pro – FTMO-Edition

Features:
- .env-basierte Konfiguration (Redis, MT5)
- Authentifizierter Redis-Zugriff
- ZMQ Signal-Publishing
- MT5 Orderausführung über REST
- EMA-basierte Trendlogik für FOREX, CRYPTO, METALS
"""

import os
import json
import ssl
import zmq
import redis
import asyncio
import logging
import aiohttp
import numpy as np
from enum import Enum, auto
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# === .env laden ===
load_dotenv("/opt/coreflow/.env")

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-15s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("/opt/coreflow/logs/valg_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("VALG.CPU")

# === Enums ===
class AssetType(Enum):
    FOREX = auto()
    METALS = auto()
    CRYPTO = auto()

class TrendDirection(Enum):
    UP = auto()
    DOWN = auto()
    SIDEWAYS = auto()

# === Konfiguration ===
@dataclass
class EngineConfig:
    zmq_host: str = os.getenv("ZMQ_HOST", "tcp://*:5555")
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD")
    redis_channel: str = os.getenv("REDIS_CHANNEL", "valg_signals")
    mt5_api_url: str = os.getenv("MT5_API_URL", "https://mt5-server.example.com/api")
    mt5_api_key: str = os.getenv("MT5_API_KEY", "demo-key")
    trade_volume: float = float(os.getenv("TRADE_VOLUME", 0.1))
    max_retries: int = int(os.getenv("MAX_RETRIES", 3))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", 10))

# === Marktanalyse ===
class EnhancedMarketAnalyzer:
    def __init__(self):
        self.indicators = {
            "forex": {"ema_fast": 12, "ema_slow": 26},
            "crypto": {"ema_fast": 25, "ema_slow": 99},
            "metals": {"ema_fast": 50, "ema_slow": 200}
        }

    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema

    def _crypto_trend_logic(self, closes: np.ndarray, ema_f: np.ndarray, ema_s: np.ndarray) -> TrendDirection:
        price = closes[-1]
        if abs(price - ema_f[-1]) > 0.03 * price:
            return TrendDirection.UP if price > ema_f[-1] else TrendDirection.DOWN
        return TrendDirection.SIDEWAYS

    def analyze(self, symbol: str, data: dict) -> Optional[dict]:
        try:
            closes = data['close']
            if symbol.startswith(("EUR", "GBP", "USD")):
                params = self.indicators["forex"]
            elif symbol == "BTCUSD":
                params = self.indicators["crypto"]
            elif symbol == "XAUUSD":
                params = self.indicators["metals"]
            else:
                return None

            ema_fast = self.calculate_ema(closes, params["ema_fast"])
            ema_slow = self.calculate_ema(closes, params["ema_slow"])

            if symbol == "BTCUSD":
                trend = self._crypto_trend_logic(closes, ema_fast, ema_slow)
            else:
                trend_diff = ema_fast[-1] - ema_slow[-1]
                if abs(trend_diff) < 0.0005 * closes[-1]:
                    trend = TrendDirection.SIDEWAYS
                else:
                    trend = TrendDirection.UP if trend_diff > 0 else TrendDirection.DOWN

            return {
                "symbol": symbol,
                "trend": trend.name,
                "ema_fast": float(ema_fast[-1]),
                "ema_slow": float(ema_slow[-1]),
                "asset_type": "FOREX" if symbol in ["EURUSD", "GBPUSD", "USDJPY"] else "METALS" if symbol == "XAUUSD" else "CRYPTO"
            }
        except Exception as e:
            logger.error(f"Analyse fehlgeschlagen für {symbol}: {str(e)}")
            return None

# === MT5 REST Bridge ===
class MT5Bridge:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.session = None
        self.ssl_ctx = ssl.create_default_context()
        self.ssl_ctx.check_hostname = False
        self.ssl_ctx.verify_mode = ssl.CERT_NONE

    async def initialize(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
            connector=aiohttp.TCPConnector(ssl=self.ssl_ctx)
        )

    async def execute_order(self, signal: dict) -> bool:
        try:
            if not self.session:
                await self.initialize()

            endpoint = f"{self.config.mt5_api_url}/order"
            headers = {"X-API-KEY": self.config.mt5_api_key}

            response = await self.session.post(endpoint, json=signal, headers=headers)

            if response.status == 200:
                return True
            logger.error(f"Order fehlgeschlagen: {await response.text()}")
            return False
        except Exception as e:
            logger.error(f"Orderausführung fehlgeschlagen: {str(e)}")
            return False

    async def close(self):
        if self.session:
            await self.session.close()

# === Hauptengine ===
class VALGEnginePro:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.analyzer = EnhancedMarketAnalyzer()
        self.bridge = MT5Bridge(config)
        self.redis = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password,
            decode_responses=True
        )
        self.zmq_context = zmq.Context()
        self.publisher = None

    async def initialize(self):
        await self.bridge.initialize()
        self.publisher = self.zmq_context.socket(zmq.PUB)
        self.publisher.bind(self.config.zmq_host)
        logger.info("Engine erfolgreich initialisiert")

    def publish_signal(self, signal: dict):
        try:
            signal_str = json.dumps(signal)
            self.redis.publish(self.config.redis_channel, signal_str)
            self.publisher.send_string(signal_str)
            logger.info(f"Signal veröffentlicht: {signal['symbol']}")
        except Exception as e:
            logger.error(f"Signalveröffentlichung fehlgeschlagen: {str(e)}")

    async def process_signal(self, symbol: str, data: dict):
        try:
            analysis = self.analyzer.analyze(symbol, data)
            if not analysis:
                return

            signal = {
                **analysis,
                "volume": self.config.trade_volume,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            self.publish_signal(signal)

            if await self.bridge.execute_order(signal):
                logger.info(f"Order erfolgreich für {symbol}")
            else:
                logger.warning(f"Order fehlgeschlagen für {symbol}")

        except Exception as e:
            logger.error(f"Signalverarbeitung fehlgeschlagen: {str(e)}")

    async def shutdown(self):
        try:
            await self.bridge.close()
            self.publisher.close()
            self.zmq_context.term()
            logger.info("Engine heruntergefahren")
        except Exception as e:
            logger.error(f"Fehler beim Herunterfahren: {str(e)}")

# === Main Loop ===
async def main():
    config = EngineConfig()
    engine = VALGEnginePro(config)

    try:
        await engine.initialize()

        sample_data = {
            "EURUSD": {"close": np.random.normal(1.08, 0.005, 1000).cumsum()},
            "GBPUSD": {"close": np.random.normal(1.25, 0.006, 1000).cumsum()},
            "USDJPY": {"close": np.random.normal(110.0, 0.5, 1000).cumsum()},
            "BTCUSD": {"close": np.random.normal(30000, 500, 1000).cumsum()},
            "XAUUSD": {"close": np.random.normal(1800, 10, 1000).cumsum()}
        }

        while True:
            tasks = [engine.process_signal(sym, data) for sym, data in sample_data.items()]
            await asyncio.gather(*tasks)
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logger.info("Empfange Herunterfahr-Signal...")
    except Exception as e:
        logger.critical(f"Kritischer Fehler: {str(e)}", exc_info=True)
    finally:
        await engine.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
