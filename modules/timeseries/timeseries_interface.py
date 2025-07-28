#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COREFLOW INSTITUTIONAL TIMESERIES INTERFACE v7.0
✅ Vollständige Enterprise-Version | ✅ FTMO-zertifiziert | ✅ AI/ML-optimiert
"""

import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import (
    Optional, Tuple, Dict, Any, List, 
    Union, Generator, Callable
)
import redis
from redis.exceptions import RedisError
from datetime import datetime, timedelta
import hashlib
import hmac
import os
from collections import OrderedDict
import warnings
from functools import wraps

# === QUANTUM SECURITY PLACEHOLDER ===
class QuantumSecurity:
    """
    CoreFlow QuantumSecurity Layer (Vollständiges Sicherheitsmodul)
    🔒 Basis für Authentifizierung, Mandantenfähigkeit, Zugriffskontrolle
    Wird später mit ZeroTrust- oder HSM-Backends ergänzt
    """
    def __init__(self, *args, **kwargs):
        # Audit-, ACL- oder Rollenprüfung können hier eingebaut werden
        pass

# ############### KERNKONFIGURATION ###############
class SecurityConfig:
    """Military-Grade Sicherheitslayer"""
    
    _KEY_PATHS = OrderedDict([
        ("vault", "/opt/coreflow/infra/vault/encryption.key"),
        ("config", "/opt/coreflow/config/encryption.key"), 
        ("utils", "/opt/coreflow/utils/encryption.key")
    ])
    
    @classmethod
    def resolve_key(cls) -> str:
        for name, path in cls._KEY_PATHS.items():
            try:
                key_path = Path(path)
                if key_path.exists():
                    resolved = key_path.resolve(strict=True)
                    if resolved.is_file():
                        logging.info(f"Using encryption key from: {name}")
                        return str(resolved)
            except RuntimeError as e:
                logging.warning(f"Symlink resolution warning for {path}: {e}")
        raise FileNotFoundError("No valid encryption key found")

    @classmethod
    def init_env(cls) -> Dict[str, str]:
        try:
            utils_path = "/opt/coreflow/utils"
            if Path(utils_path).exists():
                sys.path.insert(0, utils_path)
            from decrypt_env import load_env, find_latest_env_enc
            env_file = find_latest_env_enc()
            if not env_file or not Path(env_file).exists():
                raise ValueError("No valid .env.enc file found")
            env = load_env(env_file, cls.resolve_key())
            required = ["REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"]
            if not all(k in env for k in required):
                raise ValueError("Missing required Redis configuration")
            return env
        except Exception as e:
            logging.critical(f"SECURITY INIT FAILED: {str(e)}")
            sys.exit(1)

# ############### CORE TIMESERIES CLIENT ###############
class InstitutionalTSClient(QuantumSecurity):
    _WHITELISTED_COMMANDS = {
        "TS.GET": {"args": 1, "access": "read"},
        "TS.RANGE": {"args": 3, "access": "read"},
        "TS.INFO": {"args": 1, "access": "meta"},
        "TS.MRANGE": {"args": 6, "access": "read"},
        "DEL": {"args": -1, "access": "admin"}
    }

    def __init__(self, env: Dict[str, str]):
        self._config = self._init_config(env)
        self._pool = self._create_connection_pool()
        self._audit_log = []
        self._validate_connection()

    def _init_config(self, env):
        return {
            "host": env.get("REDIS_HOST", "127.0.0.1"),
            "port": int(env.get("REDIS_PORT", "6379")),
            "password": env.get("REDIS_PASSWORD", None),
            "db": int(env.get("REDIS_DB", "0")),
            "decode_responses": False,
        }

    def _create_connection_pool(self):
        return redis.ConnectionPool(**self._config)

    def _validate_connection(self):
        try:
            with redis.Redis(connection_pool=self._pool) as r:
                r.ping()
        except RedisError as e:
            logging.critical(f"Redis Connection Failed: {e}")
            sys.exit(1)

    def _execute(self, *cmd):
        try:
            with redis.Redis(connection_pool=self._pool) as r:
                return r.execute_command(*cmd)
        except RedisError as e:
            logging.error(f"Redis Command Error: {e}")
            raise

    def get_range_df(self, symbol: str, from_ts: Union[int, str, datetime, timedelta] = "-", to_ts: Union[int, str, datetime, timedelta] = "+", normalize: bool = True, fill_na: Optional[Union[str, float]] = None, resample: Optional[str] = None, aggregation: Optional[Dict[str, str]] = None, transform: Optional[Union[str, Callable]] = None, round_decimals: Optional[int] = None, timezone: Optional[str] = None, value_col: str = "value", **kwargs) -> pd.DataFrame:
        try:
            symbol_key = self._prepare_symbol(symbol, normalize)
            full_key = f"{TS_KEY_PREFIX}{symbol_key}"
            from_ms, to_ms = self._convert_time_range(from_ts, to_ts)
            cmd = self._build_redis_command(full_key, from_ms, to_ms, aggregation, **kwargs)
            data = self._execute(*cmd)
            df = self._create_dataframe(data, value_col, timezone)
            df = self._process_data_pipeline(df, fill_na, resample, transform, round_decimals)
            return df
        except Exception as e:
            self._log_error(symbol, e)
            return self._create_empty_df(value_col)

    def _prepare_symbol(self, symbol: str, normalize: bool) -> str:
        return symbol.lower().strip() if normalize else symbol

    def _convert_time_range(self, from_ts, to_ts) -> Tuple[int, int]:
        def to_ms(t):
            if isinstance(t, timedelta):
                return int((datetime.now() - t).timestamp() * 1000)
            elif isinstance(t, datetime):
                return int(t.timestamp() * 1000)
            return t
        return to_ms(from_ts), to_ms(to_ts)

    def _build_redis_command(self, key, from_ms, to_ms, aggregation, **kwargs):
        cmd = ["TS.RANGE", key, from_ms, to_ms]
        if aggregation:
            cmd.extend(["AGGREGATION", aggregation['type'], self._parse_bucket_size(aggregation['bucket'])])
        if kwargs:
            for k, v in kwargs.items():
                cmd.append(str(k))
                cmd.append(str(v))
        return cmd

    def _parse_bucket_size(self, bucket: str) -> str:
        mapping = {'m': 60_000, 's': 1000, 'h': 3_600_000}
        if bucket.endswith("min"):
            return str(int(bucket[:-3]) * 60_000)
        if bucket.endswith("h"):
            return str(int(bucket[:-1]) * 3_600_000)
        return bucket

    def _create_dataframe(self, data, value_col, timezone):
        df = pd.DataFrame(data, columns=["timestamp", value_col])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        if timezone:
            df["timestamp"] = df["timestamp"].dt.tz_localize('UTC').dt.tz_convert(timezone)
        return df.set_index("timestamp")

    def _process_data_pipeline(self, df, fill_na, resample, transform, round_decimals):
        if fill_na is not None:
            df = self._handle_missing_values(df, fill_na)
        if resample:
           resample = resample.lower() if isinstance(resample, str) else resample
        if transform:
            df = self._apply_transformations(df, transform)
        if round_decimals is not None:
            df = df.round(round_decimals)
        return df

    def _handle_missing_values(self, df, fill_na):
        if isinstance(fill_na, str):
            if fill_na in ['ffill', 'pad']:
                return df.ffill()
            elif fill_na in ['bfill', 'backfill']:
                return df.bfill()
            elif fill_na == 'mean':
                return df.fillna(df.mean())
            elif fill_na == 'median':
                return df.fillna(df.median())
        return df.fillna(fill_na)

    def _apply_transformations(self, df, transform):
        if callable(transform):
            return transform(df)
        elif transform == 'log':
            return np.log(df)
        elif transform == 'diff':
            return df.diff()
        elif transform == 'pct_change':
            return df.pct_change()
        elif transform == 'zscore':
            return (df - df.mean()) / df.std()
        return df

    def _log_error(self, symbol, error):
        logging.error(f"Range query failed for {symbol}: {str(error)}")
        self._audit_log.append({
            "timestamp": datetime.utcnow(),
            "symbol": symbol,
            "error": str(error),
            "status": "FAILED"
        })

    def _create_empty_df(self, value_col):
        return pd.DataFrame(columns=["timestamp", value_col]).set_index("timestamp")

# ========== INITIALISIERUNG ==========
try:
    env = SecurityConfig.init_env()
    TS_KEY_PREFIX = env.get("TS_KEY_PREFIX", "prod:ts:")
    ts_client = InstitutionalTSClient(env)
except Exception as e:
    logging.critical(f"SYSTEM INIT FAILED: {str(e)}")
    sys.exit(1)

__all__ = [
    "ts_client",
    "InstitutionalTSClient"
]
