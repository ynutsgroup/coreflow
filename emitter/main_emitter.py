#!/usr/bin/env python3
# CoreFlow SignalEmitter – Redis + ZMQ + .env.secure

import sys

# 🔐 .env entschlüsseln
from emitter.env_loader import decrypt_env
ENV_PATH = "/opt/coreflow/.env.enc"
KEY_PATH = "/opt/coreflow/infra/vault/encryption.key"

try:
    env_vars = decrypt_env(ENV_PATH, KEY_PATH)
except Exception as e:
    print(f"❌ .env entschlüsseln fehlgeschlagen: {e}")
    sys.exit(1)

# ✅ DEBUG: ENV-VARS ausgeben
print("🚀 CoreFlow Emitter gestartet")
print("🔍 ENV-VARS geladen:")
for k, v in env_vars.items():
    if "REDIS" in k or "ZMQ" in k:
        print(f"  {k} = {v}")

# Jetzt Module laden (abhängig von ENV)
from emitter.logger import setup_logger
from emitter.redis_sender import send_signal
from emitter.zmq_sender import send_zmq_signal

logger = setup_logger()

# === Eingabevalidierung ===
if len(sys.argv) < 5:
    logger.error("❌ Nutzung: main_emitter.py SYMBOL ACTION LOT CONFIDENCE")
    sys.exit(1)

symbol = sys.argv[1].upper()
action = sys.argv[2].upper()

try:
    lot = float(sys.argv[3])
    confidence = float(sys.argv[4])
except ValueError:
    logger.error("❌ Ungültiger LOT oder CONFIDENCE")
    sys.exit(1)

if action not in {"BUY", "SELL"} or not (0 < lot <= 100) or not (0 <= confidence <= 1):
    logger.error("❌ Ungültige Parameter")
    sys.exit(1)

# === Signalstruktur ===
signal = {
    "symbol": symbol,
    "action": action,
    "lot": lot,
    "confidence": confidence
}

# === Redis senden ===
try:
    redis_channel = send_signal(env_vars, signal)
    logger.info(f"📤 Redis → {redis_channel}: {signal}")
except Exception as e:
    logger.error(f"❌ Redis Fehler: {e}")
    sys.exit(1)

# === ZMQ senden (optional) ===
if "ZMQ_ENDPOINT" in env_vars:
    try:
        send_zmq_signal(env_vars, signal)
        logger.info("📤 ZMQ-Signal gesendet")
    except Exception as e:
        logger.warning(f"⚠️ ZMQ-Fehler: {e}")
else:
    logger.info("ℹ️ Kein ZMQ-Endpunkt definiert – nur Redis verwendet.")

sys.exit(0)
