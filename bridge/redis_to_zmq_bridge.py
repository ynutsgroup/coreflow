#!/usr/bin/env python3
import os
import json
import time
import logging
from pathlib import Path
from cryptography.fernet import Fernet
import redis
import zmq

# ========= Konfiguration =========
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_STREAM = os.getenv("REDIS_STREAM", "trading_signals")
REDIS_GROUP = os.getenv("REDIS_GROUP", "coreflow_group")
REDIS_CONSUMER = os.getenv("REDIS_CONSUMER", "bridge_consumer")
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"
FERNET_KEY_PATH = os.getenv("FERNET_KEY_PATH", "/opt/coreflow/.env.key")

ZMQ_TARGET = os.getenv("ZMQ_TARGET", "tcp://localhost:5556")
LOG_DIR = Path(os.getenv("LOG_DIR", "/opt/coreflow/logs"))

# ========= Logging =========
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "redis_to_zmq_bridge.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RedisToZMQBridge")

# ========= Fernet laden =========
try:
    with open(FERNET_KEY_PATH, "rb") as f:
        fernet = Fernet(f.read())
except Exception as e:
    logger.critical(f"❌ Fernet-Schlüssel konnte nicht geladen werden: {str(e)}")
    exit(1)

def setup_redis_connection():
    while True:
        try:
            r = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                ssl=REDIS_SSL,
                decode_responses=False
            )
            r.ping()
            try:
                r.xgroup_create(REDIS_STREAM, REDIS_GROUP, id='0', mkstream=True)
            except redis.exceptions.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
            logger.info(f"✅ Redis verbunden – Stream: {REDIS_STREAM}, Gruppe: {REDIS_GROUP}")
            return r
        except Exception as e:
            logger.error(f"❌ Redis-Verbindungsfehler: {str(e)}")
            time.sleep(5)

def setup_zmq_connection():
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, 5000)
    socket.setsockopt(zmq.SNDHWM, 1000)
    try:
        socket.connect(ZMQ_TARGET)
        logger.info(f"✅ ZMQ verbunden – Ziel: {ZMQ_TARGET}")
        return context, socket
    except Exception as e:
        logger.critical(f"❌ ZMQ-Verbindung fehlgeschlagen: {str(e)}")
        context.term()
        raise

def process_messages(r, socket):
    while True:
        try:
            messages = r.xreadgroup(
                groupname=REDIS_GROUP,
                consumername=REDIS_CONSUMER,
                streams={REDIS_STREAM: '>'},
                count=10,
                block=5000
            )
            if not messages:
                continue

            for stream, message_list in messages:
                for msg_id, msg_data in message_list:
                    try:
                        if b"encrypted" in msg_data:
                            decrypted = fernet.decrypt(msg_data[b"encrypted"])
                            signal = json.loads(decrypted.decode("utf-8"))

                            if not all(k in signal for k in ("symbol", "action", "volume")):
                                logger.warning(f"⚠️ Ungültiges Signalformat: {signal}")
                                r.xack(REDIS_STREAM, REDIS_GROUP, msg_id)
                                continue

                            socket.send_json(signal, flags=zmq.NOBLOCK)
                            logger.info(f"📤 Signal weitergeleitet: {signal['symbol']} {signal['action']} {signal['volume']} Lot")
                            r.xack(REDIS_STREAM, REDIS_GROUP, msg_id)

                        else:
                            logger.warning(f"⚠️ Kein 'encrypted'-Feld vorhanden: {msg_data}")
                            r.xack(REDIS_STREAM, REDIS_GROUP, msg_id)

                    except Exception as e:
                        logger.error(f"❌ Fehler bei Verarbeitung von {msg_id}: {str(e)}")
                        time.sleep(1)

        except redis.exceptions.ConnectionError:
            logger.warning("🔌 Redis-Verbindung verloren, reconnect...")
            r = setup_redis_connection()
        except Exception as e:
            logger.error(f"⚠️ Unbekannter Fehler: {str(e)}")
            time.sleep(1)

def main():
    logger.info("🚀 Starte Redis → ZMQ Bridge")
    try:
        r = setup_redis_connection()
        context, socket = setup_zmq_connection()
        process_messages(r, socket)
    except KeyboardInterrupt:
        logger.info("🛑 Manuell gestoppt")
    except Exception as e:
        logger.critical(f"🔥 Kritischer Fehler: {str(e)}")
    finally:
        socket.close()
        context.term()
        logger.info("🔌 Verbindungen sauber beendet")

if __name__ == "__main__":
    main()
