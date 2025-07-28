#!/usr/bin/env python3
# CoreFlow ZMQ Emitter – Linux → Windows MT5 Bridge (.env-basiert)

import zmq
import argparse
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

# === .env laden ===
ENV_PATH = "/opt/coreflow/.env"
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
else:
    print(f"❌ .env nicht gefunden: {ENV_PATH}")
    exit(1)

# === Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger("ZMQEmitter")

def send_signal(ip, port, message):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    address = f"tcp://{ip}:{port}"
    logger.info(f"Verbinde zu {address} ...")

    try:
        socket.connect(address)
        socket.send_string(message)
        logger.info(f"✅ Signal gesendet: {message}")
    except Exception as e:
        logger.error(f"❌ Fehler beim Senden: {e}")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoreFlow ZMQ Signal Emitter")
    parser.add_argument("--message", type=str, required=True, help="Signalnachricht (z. B. BUY EURUSD 1.00)")
    args = parser.parse_args()

    ip = os.getenv("SEND_SIGNAL_IP")
    port = int(os.getenv("SEND_SIGNAL_PORT", "5555"))

    if not ip:
        logger.error("❌ SEND_SIGNAL_IP nicht gesetzt in .env")
        exit(1)

    send_signal(ip, port, args.message)
