#!/usr/bin/env python3
import os
import time
import json
import zmq
import logging
from pathlib import Path

SIGNAL_DIR = Path("/opt/coreflow/signals")
PROCESSED_DIR = SIGNAL_DIR / "sent"
ZMQ_TARGET = "tcp://62.171.165.69:5555"  # <-- HIER IP ERSETZEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/opt/coreflow/logs/zmq_bridge.log"),
        logging.StreamHandler()
    ]
)

def send_to_zmq(data: dict):
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(ZMQ_TARGET)
    socket.send_json(data)
    socket.close()
    context.term()

def run_bridge():
    logging.info("ZMQ Bridge gestartet ✅")
    SIGNAL_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

    while True:
        for file in SIGNAL_DIR.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    payload = json.load(f)

                signal_data = payload.get("data")
                if signal_data and all(k in signal_data for k in ("action", "symbol", "volume")):
                    send_to_zmq(signal_data)
                    logging.info(f"📤 Gesendet: {signal_data}")
                else:
                    logging.warning(f"⚠️ Datei ignoriert (kein gültiger Trade): {file.name}")

                file.rename(PROCESSED_DIR / file.name)
            except Exception as e:
                logging.error(f"Fehler bei Datei {file.name}: {e}")
        time.sleep(1)

if __name__ == "__main__":
    try:
        run_bridge()
    except KeyboardInterrupt:
        logging.info("ZMQ Bridge gestoppt ❌")
