#!/usr/bin/env python3
# /opt/coreflow/utils/test_pubsub.py
# Redis Pub/Sub Test – mit Entschlüsselung und Logging

import os
import redis
import logging
import json
from cryptography.fernet import Fernet, InvalidToken
from dotenv import load_dotenv

# === ENV laden ===
load_dotenv("/opt/coreflow/.env")

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('/opt/coreflow/logs/test_pubsub.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PubSubTest")

# === Fernet laden ===
try:
    with open("/opt/coreflow/.env.key", "rb") as f:
        fernet = Fernet(f.read())
except Exception as e:
    logger.critical(f"❌ Fernet-Key-Fehler: {e}")
    exit(1)

# === Redis verbinden ===
try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        password=os.getenv("REDIS_PASSWORD"),
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        decode_responses=False,
        socket_timeout=15
    )
    r.ping()
    logger.info("✅ Redis-Verbindung erfolgreich")
except Exception as e:
    logger.critical(f"❌ Redis-Verbindung fehlgeschlagen: {e}")
    exit(1)

# === Kanal abonnieren ===
channel = os.getenv("REDIS_CHANNEL")
pubsub = r.pubsub()
pubsub.subscribe(channel)
logger.info(f"📡 Lausche auf Redis-Kanal: {channel}")

# === Warten auf Nachricht ===
try:
    for message in pubsub.listen():
        if message['type'] == 'message':
            logger.info("📥 Signal erhalten – versuche zu entschlüsseln...")
            try:
                decrypted = fernet.decrypt(message['data'])
                signal = json.loads(decrypted.decode('utf-8'))
                logger.info(f"✅ Entschlüsselt: {json.dumps(signal, indent=2)}")
            except InvalidToken:
                logger.error("❌ Entschlüsselung fehlgeschlagen – ungültiger Schlüssel")
            except Exception as e:
                logger.error(f"❌ Allgemeiner Fehler bei Entschlüsselung: {e}")
except KeyboardInterrupt:
    logger.info("🛑 Beendet durch Benutzer")
