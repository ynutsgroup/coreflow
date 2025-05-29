
#!/usr/bin/env python3
# CoreFlow Signal Emitter – Plattformunabhängig

import os
import json
import logging
import redis
from pathlib import Path
from cryptography.fernet import Fernet
import hmac
import hashlib
from datetime import datetime
import socket

# ==== Umgebungsvariablen ====
ENV_PATH = "/opt/coreflow/.env"
KEY_PATH = "/opt/coreflow/.env.key"
LOG_DIR = Path("/opt/coreflow/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==== .env manuell laden ====
def load_env(path):
    with open(path) as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, val = line.strip().split('=', 1)
                os.environ[key] = val

load_env(ENV_PATH)

# ==== Logging ====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "emitter.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CoreFlowEmitter")

# ==== Fernet + HMAC ====
fernet = Fernet(os.getenv("FERNET_KEY").encode())
hmac_secret = os.getenv("HMAC_SECRET").encode()

# ==== Redis ====
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=False
)

# ==== Signal erzeugen ====
signal_data = {
    "symbol": "BTCUSD",
    "action": "BUY",
    "volume": 0.1,
    "timestamp": datetime.utcnow().isoformat(),
    "source": socket.gethostname()
}

encrypted = fernet.encrypt(json.dumps(signal_data).encode())
signature = hmac.new(hmac_secret, encrypted, hashlib.sha256).hexdigest()

try:
    r.publish(os.getenv("REDIS_CHANNEL", "trading_signals"), encrypted)
    logger.info("✅ Signal gesendet")
except Exception as e:
    logger.error(f"❌ Fehler beim Senden: {str(e)}")
