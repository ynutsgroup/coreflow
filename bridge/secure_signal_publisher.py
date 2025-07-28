#!/usr/bin/env python3
"""
CoreFlow Institutional Signal Publisher v6.2
FTMO & Hedge Fund Compliant Trading Signal Dispatcher
"""

import os
import sys
import json
import argparse
import logging
import socket
from datetime import datetime, timedelta, UTC
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv
import redis
import zmq

# === CONSTANTS ===
DEFAULT_TTL = 60  # seconds
MAX_SIZE = 100.0
MIN_SIZE = 0.01

# === ENVIRONMENT SETUP ===
ENV_PATH = "/opt/coreflow/.env"
if not Path(ENV_PATH).exists():
    print(f"❌ Config nicht gefunden: {ENV_PATH}")
    sys.exit(1)

try:
    load_dotenv(ENV_PATH, override=True)
except Exception as e:
    print(f"❌ Fehler beim Laden der .env Datei: {e}")
    sys.exit(1)

# === LOGGING CONFIG ===
log_dir = Path(os.getenv("CF_LOG_DIR", "/opt/coreflow/logs"))
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_dir/"institutional_trading.log",
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s | %(levelname)-8s | %(host)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S%z'
)
logger = logging.getLogger("CFPublisher")
logger = logging.LoggerAdapter(logger, {'host': socket.gethostname()})

# === ARGUMENT PARSING ===
def validate_size(value):
    try:
        fval = float(value)
        if not MIN_SIZE <= fval <= MAX_SIZE:
            raise ValueError
        return round(fval, 2)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Size must be {MIN_SIZE}-{MAX_SIZE}, got {value}")

parser = argparse.ArgumentParser(
    description='Institutional Trading Signal Publisher',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--asset", required=True,
    choices=["forex", "metals", "crypto", "indices"],
    help="Asset class"
)
parser.add_argument(
    "--symbol", required=True,
    help="Trading symbol (e.g. EURUSD, XAUUSD)"
)
parser.add_argument(
    "--action", required=True,
    choices=["BUY", "SELL", "CLOSE"],
    help="Trade action"
)
parser.add_argument(
    "--size", required=True, type=validate_size,
    help=f"Position size ({MIN_SIZE}-{MAX_SIZE} lots)"
)
parser.add_argument(
    "--strategy", default="VALG_UNKNOWN",
    help="Strategy identifier"
)
args = parser.parse_args()

# === SECURE CONNECTIONS ===
class SecureConnections:
    _redis_pool = None
    _zmq_socket = None

    @classmethod
    def get_redis(cls):
        if not cls._redis_pool:
            try:
                redis_auth = Path(os.getenv("REDIS_AUTH_FILE")).read_text().strip()
                ssl_config = {
                    'ssl_cert_reqs': 'required',
                    'ssl_ca_certs': '/etc/ssl/certs/ca-certificates.crt'
                } if os.getenv("REDIS_SSL", "false").lower() == "true" else {}
                
                cls._redis_pool = redis.ConnectionPool(
                    host=os.getenv("REDIS_HOST"),
                    port=int(os.getenv("REDIS_PORT", "6380")),
                    password=redis_auth,
                    **ssl_config,
                    max_connections=10,
                    health_check_interval=30
                )
            except Exception as e:
                logger.critical(f"Redis connection failed: {e}")
                raise SystemExit(1)
        return redis.Redis(connection_pool=cls._redis_pool)

    @classmethod
    def get_zmq(cls):
        if os.getenv("USE_ZMQ", "false").lower() == "true":
            if not cls._zmq_socket:
                try:
                    ctx = zmq.Context.instance()
                    cls._zmq_socket = ctx.socket(zmq.PUB)
                    cls._zmq_socket.bind(
                        f"tcp://{os.getenv('SEND_SIGNAL_IP')}:"
                        f"{os.getenv('ZMQ_PORT', '5555')}"
                    )
                except Exception as e:
                    logger.error(f"ZMQ setup failed: {e}")
            return cls._zmq_socket
        return None

# === ENCRYPTION SERVICE ===
class EncryptionService:
    _fernet = None

    @classmethod
    def get_fernet(cls):
        if not cls._fernet:
            try:
                key_path = Path(os.getenv("FERNET_KEY_FILE"))
                cls._fernet = Fernet(key_path.read_bytes())
            except Exception as e:
                logger.critical(f"Encryption init failed: {e}")
                raise SystemExit(1)
        return cls._fernet

    @classmethod
    def encrypt_payload(cls, payload: dict) -> str:
        try:
            return cls.get_fernet().encrypt(
                json.dumps(payload).encode()
            ).decode()
        except Exception as e:
            logger.critical(f"Encryption failed: {e}")
            raise SystemExit(1)

# === SIGNAL PUBLISHER ===
class InstitutionalSignalPublisher:
    def __init__(self):
        self.redis = SecureConnections.get_redis()
        self.zmq = SecureConnections.get_zmq()
        self.stream = os.getenv("REDIS_STREAM", "institutional_signals")
        self.ttl = int(os.getenv("MESSAGE_TTL_SECONDS", DEFAULT_TTL))

    def _create_payload(self, asset, symbol, action, size, strategy):
        return {
            "protocol": "CF6.2",
            "asset": asset.lower(),
            "symbol": symbol.upper().replace("/", ""),
            "action": action.upper(),
            "size": size,
            "strategy": strategy.upper(),
            "timestamp": datetime.now(UTC).isoformat(),
            "valid_until": (datetime.now(UTC) + timedelta(seconds=self.ttl)).isoformat(),
            "source": socket.gethostname(),
            "signature": os.getenv("SIGNATURE", "unsigned")
        }

    def publish(self, asset, symbol, action, size, strategy):
        payload = self._create_payload(asset, symbol, action, size, strategy)
        try:
            encrypted = EncryptionService.encrypt_payload(payload)
            
            # Redis Publication
            stream_id = self.redis.xadd(
                self.stream,
                {"payload": encrypted},
                maxlen=1000
            )
            
            # ZMQ Publication if configured
            if self.zmq:
                self.zmq.send_json({
                    "redis_id": stream_id.decode(),
                    "payload": encrypted
                })
            
            logger.info(
                f"Published {asset}/{symbol} {action} {size} "
                f"(ID: {stream_id.decode()})"
            )
            print(f"✅ Signal published → ID: {stream_id.decode()}")
            
            return stream_id.decode()
            
        except Exception as e:
            logger.critical(f"Publishing failed: {e}")
            print(f"❌ Publication error: {e}")
            raise SystemExit(1)

# === MAIN EXECUTION ===
if __name__ == "__main__":
    try:
        publisher = InstitutionalSignalPublisher()
        publisher.publish(
            args.asset,
            args.symbol,
            args.action,
            args.size,
            args.strategy
        )
    except KeyboardInterrupt:
        print("\n🚨 Operation cancelled")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        print(f"💀 Critical failure: {e}")
        sys.exit(1)
