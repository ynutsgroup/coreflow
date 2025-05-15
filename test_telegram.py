#!/usr/bin/env python3
import sys
from pathlib import Path

# Pfadanpassung für korrekten Import
sys.path.insert(0, str(Path(__file__).parent))

from core.watchdog import send_telegram

if __name__ == "__main__":
    message = "✅ Testnachricht vom CoreFlow Watchdog!"
    if send_telegram(message):
        print("Telegram-Nachricht erfolgreich gesendet!")
    else:
        print("Fehler beim Senden!", file=sys.stderr)
        sys.exit(1)
