#!/usr/bin/env python3
import os
import csv
import time
import redis
import json
import logging
import socket
import sys
from pathlib import Path

# --- CoreFlow-Basispfad einfügen ---
sys.path.insert(0, "/opt/coreflow")  # <--- Wichtig für 'core.*'-Importe

# --- Sicherheitsgrenze erhöhen (gegen RecursionError) ---
sys.setrecursionlimit(1000)

# --- Environment laden ---
from core.utils.env_loader import EnvLoader
env = EnvLoader("/opt/coreflow/.env.replay.clean")
env.load()

SYMBOL = env.get("SYMBOL", "XAUUSD")
REPLAY_FILE = env.get("REPLAY_FILE")
SLEEP = env.get("SLEEP_INTERVAL", 0.2)
MARKET_PHASE_THRESHOLD = env.get("MARKET_PHASE", "trend")
CONFIDENCE_THRESHOLD = env.get("CONFIDENCE", 0.8)
MODE = env.get("MODE", "TEST")
REDIS_PASSWORD = env.get("REDIS_PASSWORD")
REDIS_HOST = env.get("REDIS_HOST", "127.0.0.1")
REDIS_PORT = env.get("REDIS_PORT", 6379)

# --- Logverzeichnis sicherstellen ---
Path("/opt/coreflow/logs").mkdir(parents=True, exist_ok=True)

# --- Logging konfigurieren ---
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s [REPLAY-{SYMBOL}] %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(f"/opt/coreflow/logs/replay_{SYMBOL}.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# --- DNS validieren ---
try:
    ip = socket.gethostbyname(REDIS_HOST)
    log.info(f"Redis-Host aufgelöst: {REDIS_HOST} → {ip}")
except Exception as e:
    log.critical(f"DNS-Auflösung fehlgeschlagen: {REDIS_HOST} → {e}")
    exit(1)

# --- Redis-Verbindung initialisieren ---
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,
    socket_connect_timeout=3,
    retry_on_timeout=False,
    health_check_interval=10
)

try:
    r.ping()
except redis.ConnectionError as e:
    log.critical(f"Redis-PING fehlgeschlagen: {e}")
    exit(1)

# --- Watchdog initialisieren ---
r.set(f'coreflow:status:replay_{SYMBOL}', 'alive', ex=10)

# --- Dynamische KI-Module laden ---
def load_ai_modules():
    global detect_market_phase, predict_signal
    from core.valg_engine import detect_market_phase
    from core.lstm_predictor import predict_signal

load_ai_modules()

# --- Replay starten ---
with open(REPLAY_FILE) as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        if i % 10 == 0:
            r.set(f'coreflow:status:replay_{SYMBOL}', 'alive', ex=15)

        bar = {
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row.get('tick_volume', row.get('volume', 0)))
        }

        if detect_market_phase(bar) != MARKET_PHASE_THRESHOLD:
            continue

        prediction = predict_signal(bar)
        if prediction['confidence'] < CONFIDENCE_THRESHOLD:
            continue

        payload = {
            'symbol': SYMBOL,
            'action': prediction['signal'],
            'confidence': prediction['confidence'],
            'timestamp': row['time'],
            'source': 'replay'
        }

        if MODE == 'LIVE':
            r.publish('coreflow_signals', json.dumps(payload))
            log.info(f"LIVE SIGNAL: {payload}")
        else:
            log.info(f"TEST SIGNAL: {payload}")

        time.sleep(SLEEP)
