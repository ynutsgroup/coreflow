#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow TickDataProcessor – Enhanced Version
✅ Redis Streams + Pub/Sub Support
✅ Robust Time-Series Handling
✅ Memory-Efficient Processing
✅ Simulation-Ready Multi-File Aggregation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import redis
import json
import logging
import warnings
from datetime import datetime
import gc
import resource
from typing import Optional, Callable, Tuple
import os
import sys

# === Dynamischer Import-Pfad für decrypt_env.py ===
decrypt_path = Path("/opt/coreflow/utils")
if decrypt_path.exists():
    sys.path.insert(0, str(decrypt_path))
else:
    raise FileNotFoundError(f"❌ Verzeichnis nicht gefunden: {decrypt_path}")

from decrypt_env import load_env  # 🔐 Neue Zeile zur Integration der Verschlüsselung

# === Logger Setup ===
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/opt/coreflow/data/dukascopy/tick_processor.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    warnings.filterwarnings('ignore', category=FutureWarning)
    return logger

logger = setup_logger()

class TickDataProcessor:
    def __init__(self, data_dir: str = "/opt/coreflow/data/dukascopy/csv_ticks"):
        self.DATA_DIR = Path(data_dir)
        self.redis_conn = None
        self.redis_enabled = False
        logger.info(f"Processor initialized | Data Dir: {data_dir}")

    def enable_redis(self, host="10.10.10.50", port=6380, password=None, timeout=5) -> bool:
        try:
            self.redis_conn = redis.Redis(
                host=host,
                port=port,
                password=password,
                socket_timeout=timeout,
                decode_responses=False,
                health_check_interval=30
            )
            if self.redis_conn.ping():
                self.redis_enabled = True
                logger.info(f"Redis connected | Host: {host}:{port}")
                try:
                    self.redis_conn.ts().create('ticks:EURUSD:bid', retention_msec=604800000)
                    self.redis_conn.ts().create('ticks:EURUSD:ask', retention_msec=604800000)
                    logger.info("Redis TimeSeries initialized")
                except:
                    logger.warning("RedisTimeSeries module not available")
                return True
        except Exception as e:
            logger.error(f"Redis connection failed: {str(e)}")
        return False

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.lower().str.strip()
        column_map = {
            'bidprice': 'bid', 'askprice': 'ask', 'price': 'bid',
            'time': 'timestamp', 'date': 'timestamp', 'datetime': 'timestamp',
            'amount': 'volume', 'value': 'volume', 'size': 'volume'
        }
        return df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    def _process_timestamp(self, ts_series) -> pd.Series:
        if np.issubdtype(ts_series.dtype, np.number):
            for unit in ['ns', 'us', 'ms', 's']:
                try:
                    return pd.to_datetime(ts_series, unit=unit, utc=True)
                except:
                    continue
        return pd.to_datetime(ts_series, utc=True, errors='coerce')

    def load_tick_file(self, file_path: Path) -> pd.DataFrame:
        try:
            if file_path.stat().st_size > 50 * 1024 * 1024:
                logger.warning(f"Large file detected ({file_path.stat().st_size/1024/1024:.2f}MB)")
                return self._load_in_chunks(file_path)

            df = pd.read_parquet(file_path)
            df = self._normalize_columns(df)
            df['timestamp'] = self._process_timestamp(df['timestamp'])
            df = df[df['timestamp'].notna()].set_index('timestamp').sort_index()

            if 'ask' not in df.columns or df['ask'].isna().all():
                df['ask'] = df['bid'] + df['bid'] * 0.0001
                logger.warning(f"ASK reconstructed for {file_path.name} (missing or invalid)")

            return df[['bid', 'ask', 'volume']] if 'volume' in df.columns else df[['bid', 'ask']]
        except Exception as e:
            logger.error(f"File load error {file_path.name}: {str(e)}")
            return pd.DataFrame()

    def _load_in_chunks(self, file_path: Path, chunk_size: int = 100000) -> pd.DataFrame:
        try:
            reader = pd.read_parquet(file_path, iterator=True)
            chunks = []
            while True:
                try:
                    chunk = reader.read(chunk_size)
                    if chunk.empty:
                        break
                    chunk = self._normalize_columns(chunk)
                    chunk['timestamp'] = self._process_timestamp(chunk['timestamp'])

                    if 'ask' not in chunk.columns or chunk['ask'].isna().all():
                        chunk['ask'] = chunk['bid'] + chunk['bid'] * 0.0001
                        logger.warning(f"ASK reconstructed for chunk in {file_path.name}")

                    chunks.append(chunk[['timestamp', 'bid', 'ask']])
                    gc.collect()
                except StopIteration:
                    break
            if chunks:
                return pd.concat(chunks).set_index('timestamp').sort_index()
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Chunked load failed: {str(e)}")
            return pd.DataFrame()

    def process_all_files(self, symbol: str = "EURUSD") -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            all_files = sorted(self.DATA_DIR.glob(f"{symbol}_*.parquet"))
            all_ticks = [self.load_tick_file(f) for f in all_files]
            df_ticks = pd.concat(all_ticks).sort_index()
            df_ohlcv = self.resample_ticks_to_ohlcv(df_ticks)
            return df_ticks, df_ohlcv
        except Exception as e:
            logger.error(f"Failed to process all files: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def resample_ticks_to_ohlcv(self, ticks: pd.DataFrame, tf="1Min") -> pd.DataFrame:
        if ticks.empty:
            return pd.DataFrame()

        try:
            ticks['mid'] = (ticks['bid'] + ticks['ask']) / 2
            ohlcv = ticks['mid'].resample(tf).ohlc()
            ohlcv['volume'] = ticks['bid'].resample(tf).count()
            return ohlcv.dropna()
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            return pd.DataFrame()

    def plot_ohlcv(self, ohlcv: pd.DataFrame):
        try:
            ohlcv[['open', 'high', 'low', 'close']].plot(title="OHLCV Chart", figsize=(12, 6))
            plt.tight_layout()
            plt.savefig("ohlcv_plot.png")
            plt.close()
            logger.info("Saved plot: ohlcv_plot.png")
        except Exception as e:
            logger.warning(f"Plot error: {e}")

    def prepare_ai_input(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        try:
            df = ohlcv.copy()
            df = df[df['close'].between(0.1, 10000)]

            # Adaptive clipping based on z-score threshold
            returns = df['close'].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            std_dev = returns.std()
            z_limit = 4 * std_dev if std_dev > 0 else 1
            df['return'] = returns.clip(lower=-z_limit, upper=z_limit)

            df['volatility'] = df['return'].rolling(10).std().clip(0, 1).fillna(0)
            df['spread'] = (df['high'] - df['low']).clip(0, 100).fillna(0)

            return df[['open', 'high', 'low', 'close', 'volume', 'return', 'volatility', 'spread']].dropna()
        except Exception as e:
            logger.error(f"AI feature prep failed: {e}")
            return pd.DataFrame()

    def publish_to_redis(self, df: pd.DataFrame, symbol: str = "EURUSD"):
        if not self.redis_enabled:
            return

        stream_name = f"ticks:{symbol.lower()}"
        pubsub_channel = f"ticks:{symbol.lower()}:live"

        try:
            with self.redis_conn.pipeline() as pipe:
                for timestamp, row in df.iterrows():
                    tick_data = {
                        'timestamp': timestamp.isoformat(),
                        'bid': float(row['bid']),
                        'ask': float(row['ask']),
                        'volume': float(row.get('volume', 0))
                    }
                    pipe.xadd(stream_name, tick_data, id='*', maxlen=10000)
                    pipe.publish(pubsub_channel, json.dumps(tick_data))
                    if hasattr(self.redis_conn, 'ts'):
                        ts = int(timestamp.timestamp() * 1000)
                        pipe.ts().add(f"{stream_name}:bid", ts, float(row['bid']))
                        pipe.ts().add(f"{stream_name}:ask", ts, float(row['ask']))
                pipe.execute()
            logger.info(f"Published {len(df)} ticks to Redis")
        except Exception as e:
            logger.error(f"Redis publish error: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting CoreFlow Processor")
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, resource.RLIM_INFINITY))

        # 🔐 .env.enc entschlüsseln und env erhalten
        env = load_env("/opt/coreflow/config/.env.enc", "/opt/coreflow/config/encryption.key")

        processor = TickDataProcessor()

        redis_config = {
            "host": env.get("REDIS_HOST", "10.10.10.50"),
            "port": int(env.get("REDIS_PORT", 6380)),
            "password": env.get("REDIS_PASSWORD", None),
            "timeout": int(env.get("REDIS_TIMEOUT", 10))
        }
        processor.enable_redis(**redis_config)

        ticks, ohlcv = processor.process_all_files("EURUSD")

        if not ohlcv.empty:
            processor.plot_ohlcv(ohlcv.tail(120))

            if processor.redis_enabled:
                processor.publish_to_redis(ticks.head(10000))

            features = processor.prepare_ai_input(ohlcv)
            logger.info(f"AI Features Sample:\n{features.describe()}")

    except MemoryError:
        logger.critical("Memory limit exceeded")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        logger.info("Processing completed")
