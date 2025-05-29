# coreflow_structure_check.py – Systemstruktur-Scan für CoreFlow
# Autor: GPT CoreFlow Assist, Mai 2025

import os
from pathlib import Path

BASE_PATH = Path("/opt/coreflow")

CHECKLIST = {
    "Virtuelle Umgebung": BASE_PATH / ".venv/bin/activate",
    ".env Datei": BASE_PATH / ".env",
    ".env.key (optional)": BASE_PATH / ".env.key",
    "Decryptor Script": BASE_PATH / "utils/decrypt_env.py",
    "Signal Emitter": BASE_PATH / "utils/signal_emitter.py",
    "Main CoreFlow": BASE_PATH / "coreflow_main.py",
    "Watchdog": BASE_PATH / "coreflow_watchdog.py",
    "FTMO AI Trader": BASE_PATH / "core/ai/ftmo_ai_trader.py",
    "Strategy Engine (Entry)": BASE_PATH / "core/strategy/entry_decision_engine.py",
    "Backup Folder": BASE_PATH / "backup/",
    "Telegram Notifier": BASE_PATH / "utils/telegram_notifier.py",
    "Git Sync Script": BASE_PATH / "utils/git_sync.sh",
}

def check_file(path: Path) -> str:
    return "✅ Vorhanden" if path.exists() else "❌ Fehlt"

def main():
    print("\n📦 CoreFlow Systemstruktur Check – Stand:", BASE_PATH)
    print("--------------------------------------------------------")
    for name, path in CHECKLIST.items():
        print(f"{name:<35}: {check_file(path)}")
    print("--------------------------------------------------------\n")

if __name__ == "__main__":
    main()
