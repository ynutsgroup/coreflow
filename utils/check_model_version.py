#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoreFlow Modellprüfer – GPT-Kompatibilitätsprüfung mit Telegram-Warnung bei Modellwechsel
"""

import os
import sys
import time
import logging
from dotenv import load_dotenv

# === Projektpfad für Telegram-Funktion aktivieren ===
sys.path.insert(0, "/opt/coreflow")

try:
    from core.telegram_bot.alerts.notify import notify_telegram
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("⚠️ Telegram-Modul nicht gefunden – keine Benachrichtigung möglich")

# === .env laden ===
ENV_PATH = "/opt/coreflow/.env"
load_dotenv(ENV_PATH)

# === Logging konfigurieren ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger("ModelChecker")

# === Erlaubte Modelle (FTMO-konform) ===
ALLOWED_MODELS = {
    "gpt-4.5",
    "gpt-4o",
    "gpt-4-turbo"
}

# === Status-Datei zur Erkennung von Änderungen
LAST_MODEL_PATH = "/opt/coreflow/.model_version.cache"

def load_last_model() -> str:
    if os.path.exists(LAST_MODEL_PATH):
        with open(LAST_MODEL_PATH, "r") as f:
            return f.read().strip().lower()
    return ""

def save_last_model(model: str):
    with open(LAST_MODEL_PATH, "w") as f:
        f.write(model)

def validate_model():
    logger.info("🧠 CoreFlow Modellprüfung gestartet...")

    current_model = os.getenv("COREFLOW_MODEL", "").strip().lower()
    system_mode = os.getenv("SYSTEM_MODE", "TEST").strip().upper()
    last_model = load_last_model()

    if not current_model:
        logger.warning("⚠️ Keine COREFLOW_MODEL Variable in .env gefunden – Abbruch")
        return

    if current_model not in ALLOWED_MODELS:
        logger.critical(f"❌ Unzulässiges Modell: {current_model}")
        logger.info("🔐 Erlaubt sind nur: gpt-4.5, gpt-4o, gpt-4-turbo")
        return

    logger.info(f"✅ Erlaubtes Modell erkannt: {current_model}")
    logger.info(f"🌍 SYSTEM_MODE = {system_mode}")
    if system_mode == "LIVE":
        logger.info("🔒 Live-Modus aktiviert – nur sichere Modelle erlaubt.")

    if current_model != last_model:
        logger.info(f"🔁 Modelländerung erkannt – {last_model or 'unbekannt'} ➜ {current_model}")
        save_last_model(current_model)

        if TELEGRAM_AVAILABLE:
            notify_telegram(f"🧠 CoreFlow Modellwechsel erkannt:\n`{last_model or 'unbekannt'}` ➜ `{current_model}`")
            logger.info("📬 Telegram-Nachricht gesendet.")
        else:
            logger.warning("⚠️ Telegram nicht verfügbar – keine Benachrichtigung gesendet.")
    else:
        logger.info("🔄 Kein Modellwechsel seit letzter Prüfung.")

    logger.info("✅ Modellprüfung abgeschlossen.")

if __name__ == "__main__":
    while True:
        try:
            validate_model()
            time.sleep(3600)  # Alle 60 Minuten prüfen
        except KeyboardInterrupt:
            logger.info("🛑 Manuell beendet")
            break
