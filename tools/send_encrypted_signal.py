#!/usr/bin/env python3
"""
Secure Trading Signal Sender - Enterprise Edition
CoreFlow Redis-ZMQ Bridge Utility
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
import redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
import yaml

# === Configuration Loader ===
class Config:
    _instance = None
    
    def __init__(self):
        self.config_path = Path(os.getenv('CORE_FLOW_CONFIG', '/opt/coreflow/config/bridge_config.yaml'))
        self._load_config()
        
    def _load_config(self):
        """Load YAML config with environment variable substitution"""
        try:
            with open(self.config_path) as f:
                raw_config = os.path.expandvars(f.read())
                self._config = yaml.safe_load(raw_config)
        except Exception as e:
            raise RuntimeError(f"Config load failed: {str(e)}")

    @property
    def redis(self) -> Dict[str, Any]:
        return self._config['infrastructure']['redis']
    
    @property
    def crypto(self) -> Dict[str, Any]:
        return self._config['security']['cryptography']
    
    @property
    def logging_cfg(self) -> Dict[str, Any]:
        return self._config['observability']['logging']

# === Secure Fernet Manager ===
class FernetManager:
    def __init__(self, key_path: str):
        self.key_path = Path(key_path)
        self._validate_key_path()
        
    def _validate_key_path(self):
        if not self.key_path.exists():
            raise FileNotFoundError(f"Fernet key not found at {self.key_path}")
        if self.key_path.stat().st_mode & 0o077:
            raise PermissionError("Insecure key file permissions")
            
    def load_key(self) -> Fernet:
        """Securely load Fernet key with validation"""
        with open(self.key_path, 'rb') as f:
            key = f.read()
            if len(key) != 44:
                raise ValueError("Invalid Fernet key length")
            return Fernet(key)

# === Redis Connector ===
class RedisSignalSender:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._conn = None
        
    @property
    def connection(self) -> redis.Redis:
        """Lazy-loaded Redis connection with health check"""
        if not self._conn or not self._conn.ping():
            self._conn = redis.Redis(
                host=self.config['host'],
                port=self.config['port'],
                password=os.getenv('REDIS_PASSWORD'),
                ssl=self.config['use_ssl'],
                socket_timeout=self.config['timeout_seconds'],
                health_check_interval=30
            )
        return self._conn
    
    def send(self, encrypted_data: bytes) -> str:
        """Send encrypted signal with retry logic"""
        try:
            return self.connection.xadd(
                self.config['stream'],
                {'encrypted': encrypted_data},
                maxlen=1000  # Prevent stream overgrowth
            )
        except redis.RedisError as e:
            logging.error(f"Redis send failed: {str(e)}")
            raise

# === Signal Processor ===
class TradingSignal:
    def __init__(self, fernet: Fernet):
        self.fernet = fernet
        
    def create(self, symbol: str, action: str, volume: float) -> Dict[str, Any]:
        """Validate and format trading signal"""
        if volume <= 0:
            raise ValueError("Volume must be positive")
        return {
            'symbol': symbol.upper(),
            'action': action.upper(),
            'volume': round(volume, 2),
            'meta': {
                'source': 'coreflow_bridge',
                'version': os.getenv('APP_VERSION', '1.0')
            }
        }
    
    def encrypt(self, signal: Dict[str, Any]) -> bytes:
        """Secure encryption with data validation"""
        try:
            signal_json = json.dumps(signal, separators=(',', ':'))
            return self.fernet.encrypt(signal_json.encode('utf-8'))
        except (TypeError, ValueError) as e:
            logging.error(f"Signal validation failed: {str(e)}")
            raise

# === Main Execution ===
def main():
    # Initialize systems
    config = Config()
    
    # Setup logging
    logging.basicConfig(
        level=config.logging_cfg['level'],
        format=config.logging_cfg['format'],
        datefmt=config.logging_cfg['datefmt']
    )
    logger = logging.getLogger("SignalSender")
    
    try:
        # Parse command line
        parser = argparse.ArgumentParser(
            description="CoreFlow Secure Signal Sender",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument("--symbol", required=True, 
                          help="Trading symbol (e.g. EURUSD)")
        parser.add_argument("--action", required=True, 
                          choices=["BUY", "SELL", "CLOSE"],
                          help="Trade action")
        parser.add_argument("--volume", type=float, default=0.1,
                          help="Trade volume in lots")
        parser.add_argument("--dry-run", action="store_true",
                          help="Validate without sending")
        args = parser.parse_args()

        # Process signal
        fernet = FernetManager(config.crypto['fernet']['key_path']).load_key()
        signal = TradingSignal(fernet).create(args.symbol, args.action, args.volume)
        encrypted = TradingSignal(fernet).encrypt(signal)
        
        if args.dry_run:
            logger.info(f"DRY RUN | Valid signal: {signal}")
            return

        # Send to Redis
        sender = RedisSignalSender(config.redis)
        msg_id = sender.send(encrypted)
        logger.info(f"Signal sent | ID: {msg_id.decode()} | Symbol: {args.symbol}")

    except Exception as e:
        logger.critical(f"Critical failure: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
