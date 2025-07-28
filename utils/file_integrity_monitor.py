#!/usr/bin/env python3
# VALG-Dateiwächter – 6h Telegram-Cooldown + 5min Prüfintervall + robuster Timestamp

import os
import sys
import time
import hashlib
import asyncio
from datetime import datetime

sys.path.insert(0, "/opt/coreflow/")
from utils.telegram_notifier import send_telegram_alert

VALG_PATH = "/opt/coreflow/core/ki/valg_engine_cpu.py"
COOLDOWN_HOURS = 6
CHECK_INTERVAL_SECONDS = 300  # 5 Minuten
TIMESTAMP_FILE = "/opt/coreflow/tmp/valg_last_alert.timestamp"

os.makedirs(os.path.dirname(TIMESTAMP_FILE), exist_ok=True)

def get_file_hash(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        print(f"❌ Datei-Hashing-Fehler: {e}")
        return None

def last_alert_time() -> float:
    try:
        with open(TIMESTAMP_FILE, 'r') as f:
            return float(f.read().strip())
    except FileNotFoundError:
        return 0.0
    except Exception as e:
        print(f"❌ Fehler beim Lesen des Timestamps: {e}")
        return 0.0

def update_alert_time():
    now = time.time()
    try:
        with open(TIMESTAMP_FILE, 'w') as f:
            f.write(str(now))
        print(f"📝 Cooldown aktualisiert: {now}")
    except Exception as e:
        print(f"❌ Fehler beim Schreiben des Timestamps: {e}")

def should_alert() -> bool:
    diff = time.time() - last_alert_time()
    print(f"🔍 Letzter Alert vor {int(diff)} Sekunden ({diff/3600:.2f} h)")
    return diff >= COOLDOWN_HOURS * 3600

async def alert(message):
    await send_telegram_alert(message, alert_type="WARNING")
    update_alert_time()

def main():
    print(f"📡 VALG-Dateiwächter gestartet | Prüfe alle {CHECK_INTERVAL_SECONDS//60} Minuten | Cooldown: {COOLDOWN_HOURS} Stunden")
    last_hash = get_file_hash(VALG_PATH)

    while True:
        time.sleep(CHECK_INTERVAL_SECONDS)
        current_hash = get_file_hash(VALG_PATH)

        if current_hash != last_hash:
            print("🟠 Änderung erkannt.")
            if should_alert():
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                message = f"⚠️ VALG-Datei verändert:\n{VALG_PATH}\n🕒 {timestamp}"
                asyncio.run(alert(message))
                print(f"📤 Telegram-Alert gesendet um {timestamp}")
            else:
                print("⏳ Änderung erkannt – innerhalb Cooldown, kein Telegram.")
            last_hash = current_hash
        else:
            print("✅ Keine Änderung erkannt.")

if __name__ == "__main__":
    main()
