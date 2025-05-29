#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CoreFlow Redis Stream SignalEmitter – Zeitzonenbewusste Version

import os
import json
import redis
import logging
import time
from datetime import datetime, timezone  # Geändert für Zeitzonenunterstützung
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from typing import Dict, Optional

# === Konstanten ===
MAX_VERSUCHE = 3
WIEDERHOLUNG_VERZOEGERUNG = 1.0
SIGNAL_TTL = 3600

# === .env laden ===
load_dotenv()

# === Redis Konfiguration ===
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORT = os.getenv("REDIS_PASSWORD")
REDIS_STREAM = os.getenv("REDIS_CHANNEL", "trading_signals")
REDIS_VERBINDUNG_TIMEOUT = 5

# === Logging ===
LOG_VERZEICHNIS = os.getenv("LOG_DIR", "/opt/coreflow/logs")
os.makedirs(LOG_VERZEICHNIS, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_VERZEICHNIS, "signal_emitter_stream.log"),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SignalEmitterStream")

# === Fernet-Schlüssel ===
FERNET_SCHLUESSEL_PFAD = os.getenv("FERNET_KEY_PATH", "/opt/coreflow/.env.key")

def lade_fernet_schluessel() -> Fernet:
    for versuch in range(MAX_VERSUCHE):
        try:
            with open(FERNET_SCHLUESSEL_PFAD, "rb") as f:
                schluessel = f.read().strip()
                if len(schluessel) != 44:
                    raise ValueError("Ungültige Fernet-Schlüssellänge")
                return Fernet(schluessel)
        except Exception as e:
            if versuch == MAX_VERSUCHE - 1:
                logger.critical(f"Fernet-Schlüssel Fehler (Letzter Versuch): {e}")
                raise
            logger.warning(f"Fernet-Schlüssel Fehler (Versuch {versuch + 1}): {e}")
            time.sleep(WIEDERHOLUNG_VERZOEGERUNG)
    raise RuntimeError("Fernet-Schlüssel konnte nicht geladen werden")

fernet = lade_fernet_schluessel()

# === Redis-Verbindung ===
def erstelle_redis_verbindung() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORT,
        socket_timeout=REDIS_VERBINDUNG_TIMEOUT,
        socket_connect_timeout=REDIS_VERBINDUNG_TIMEOUT,
        health_check_interval=30,
        decode_responses=False
    )

# Globale Variable
global r
r = erstelle_redis_verbindung()

# === Signalverarbeitung ===
def erstelle_signal(symbol: str, aktion: str, volumen: float = 0.1, 
                  metadaten: Optional[Dict] = None) -> bytes:
    signal = {
        "zeitstempel": datetime.now(timezone.utc).isoformat(),  # UTC-Zeitstempel
        "symbol": symbol.upper(),
        "aktion": aktion.upper(),
        "volumen": round(volumen, 2),
        "metadaten": metadaten or {}
    }
    return fernet.encrypt(json.dumps(signal).encode())

def sende_signal(symbol: str, aktion: str, volumen: float = 0.1,
                metadaten: Optional[Dict] = None) -> bool:
    global r
    for versuch in range(MAX_VERSUCHE):
        try:
            verschluesseltes_signal = erstelle_signal(symbol, aktion, volumen, metadaten)
            
            nachrichten_id = r.xadd(
                name=REDIS_STREAM,
                fields={"verschluesseltes_signal": verschluesseltes_signal},
                maxlen=1000,
                approximate=True
            )
            
            r.expire(REDIS_STREAM, SIGNAL_TTL)
            
            log_nachricht = (f"📤 Stream-Signal gesendet | ID: {nachrichten_id.decode()} | "
                           f"Symbol: {symbol.upper()} | Aktion: {aktion.upper()}")
            logger.info(log_nachricht)
            print(log_nachricht)
            return True
            
        except redis.RedisError as e:
            if versuch == MAX_VERSUCHE - 1:
                fehlermeldung = f"❌ Signal konnte nicht gesendet werden (Letzter Versuch): {e}"
                logger.error(fehlermeldung)
                print(fehlermeldung)
                return False
            
            logger.warning(f"Wiederhole Signalversand (Versuch {versuch + 1}): {e}")
            time.sleep(WIEDERHOLUNG_VERZOEGERUNG)
            if isinstance(e, (redis.ConnectionError, redis.TimeoutError)):
                r = erstelle_redis_verbindung()

def pruefe_redis_verbindung() -> bool:
    global r
    try:
        return r.ping()
    except redis.RedisError as e:
        logger.error(f"Redis Gesundheitsprüfung fehlgeschlagen: {e}")
        return False

if __name__ == "__main__":
    print("=== CoreFlow SignalEmitter ===")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"Stream: {REDIS_STREAM}")
    
    if not pruefe_redis_verbindung():
        print("❌ Redis-Verbindung fehlgeschlagen")
        exit(1)
    
    print("✅ Redis-Verbindung erfolgreich")
    
    signale = [
        ("EURUSD", "KAUFEN", 0.2),
        ("GBPUSD", "VERKAUFEN", 0.15),
        ("XAUUSD", "KAUFEN", 0.05, {"kommentar": "Gold-Position"})
    ]
    
    for signal in signale:
        if len(signal) == 3:
            sende_signal(*signal)
        else:
            sende_signal(signal[0], signal[1], signal[2], signal[3])
if __name__ == "__main__":
    print("✅ SignalEmitter bereit. Warte auf externe Trigger...")
    try:
        while True:
            time.sleep(3600)  # Einfach aktiv bleiben
    except KeyboardInterrupt:
        print("🛑 Manuell beendet")

