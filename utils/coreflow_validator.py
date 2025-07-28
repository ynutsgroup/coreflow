#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow Redis Validator
✅ Testet ENV-Ladung + Redis-Verbindung
"""

import os
import sys
import redis
import socket
import logging
from pathlib import Path
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Pfad-Konfiguration
ENV_DIR = Path("/opt/coreflow/")
ENCRYPTION_KEY = Path("/opt/coreflow/infra/vault/encryption.key")
TMP_ENV = Path("/tmp/coreflow.env.decrypted")

# Required Variables
REQUIRED_VARS = {
    'REDIS_HOST': None,
    'REDIS_PORT': 6380,
    'REDIS_PASSWORD': ''
}

def load_encrypted_env() -> bool:
    """Lädt die verschlüsselte .env.enc Datei"""
    try:
        env_files = list(ENV_DIR.rglob('.env.enc'))
        if not env_files:
            raise FileNotFoundError("Keine .env.enc Datei gefunden")
        env_file = max(env_files, key=lambda f: f.stat().st_mtime)
        key = ENCRYPTION_KEY.read_bytes()
        decrypted = Fernet(key).decrypt(env_file.read_bytes())
        TMP_ENV.write_bytes(decrypted)
        load_dotenv(TMP_ENV, override=True)
        TMP_ENV.unlink(missing_ok=True)
        logger.info(f"ENV geladen aus {env_file}")
        return True
    except Exception as e:
        logger.error(f"ENV-Fehler: {str(e)}")
        return False

def test_redis_connection() -> bool:
    """Testet die Redis-Verbindung"""
    try:
        conf = {
            'host': os.getenv('REDIS_HOST'),
            'port': int(os.getenv('REDIS_PORT', 6380)),
            'password': os.getenv('REDIS_PASSWORD') or None,
            'socket_timeout': 3,
            'socket_connect_timeout': 3
        }
        with socket.socket() as s:
            s.settimeout(3)
            if s.connect_ex((conf['host'], conf['port'])) != 0:
                raise ConnectionError(f"Port {conf['port']} nicht erreichbar")
        r = redis.Redis(**conf)
        if not r.ping():
            raise RuntimeError("Redis nicht responsiv")
        test_key = "coreflow:healthcheck"
        r.set(test_key, "1", ex=5)
        if r.get(test_key) != b"1":
            raise RuntimeError("Dateninkonsistenz")
        logger.info(f"Redis {conf['host']}:{conf['port']} ✔️ Verbindung erfolgreich")
        return True
    except Exception as e:
        logger.error(f"Redis-Fehler: {str(e)}")
        return False

def main():
    if not load_encrypted_env():
        sys.exit(1)
    missing_vars = [k for k, v in REQUIRED_VARS.items() if not os.getenv(k) and v is None]
    if missing_vars:
        logger.error(f"Fehlende Variablen: {', '.join(missing_vars)}")
        sys.exit(2)
    if not test_redis_connection():
        sys.exit(3)
    logger.info("✅ Alle Systemchecks erfolgreich")
    sys.exit(0)

if __name__ == "__main__":
    main()
