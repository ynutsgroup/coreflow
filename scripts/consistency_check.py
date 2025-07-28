#!/usr/bin/env python3
"""
CoreFlow Consistency Check
Validiert Redis-, ZMQ-, Telegram- und Modulverfügbarkeit im Linux-Node.
"""

import os
import sys
import zmq
import importlib

# === ENV laden (vor Redis!) ===
try:
    from core.config.settings import load_env
    env = load_env()
except Exception as e:
    print(f"[ERROR] .env.enc konnte nicht geladen werden: {e}")
    sys.exit(1)

# === Redis prüfen ===
try:
    import redis
    r = redis.StrictRedis(
        host=env.get("REDIS_HOST", "127.0.0.1"),
        port=int(env.get("REDIS_PORT", "6380")),
        password=env.get("REDIS_PASSWORD", ""),
        db=0,
        decode_responses=True
    )
    pong = r.ping()
    print(f"[OK] Redis verbunden (PING): {pong}")
    signals_exist = r.exists("coreflow:signals")
    print(f"[OK] Redis Key 'coreflow:signals' vorhanden: {bool(signals_exist)}")
except Exception as e:
    print(f"[ERROR] Redis-Verbindung fehlgeschlagen: {e}")

# === ZMQ prüfen ===
try:
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    zmq_port = env.get("ZMQ_PORT", "5555")
    socket.connect(f"tcp://127.0.0.1:{zmq_port}")
    print(f"[OK] ZMQ-Port {zmq_port} ist erreichbar")
except Exception as e:
    print(f"[ERROR] ZMQ-Verbindung fehlgeschlagen: {e}")

# === Modul-Importe prüfen ===
required_modules = [
    "core.risk_manager",
    "core.zmq_bridge",
    "interfaces.mt5_adapter"
]
for module in required_modules:
    try:
        importlib.import_module(module)
        print(f"[OK] Modul geladen: {module}")
    except ImportError as e:
        print(f"[ERROR] Modul fehlgeschlagen: {module} → {e}")

# === Telegram-Nachricht senden ===
try:
    from utils.telegram_notifier import send_telegram_message
    send_telegram_message("✅ CoreFlow Consistency Check erfolgreich.")
    print(f"[OK] Telegram-Testnachricht gesendet.")
except Exception as e:
    print(f"[ERROR] Telegram konnte nicht senden: {e}")
