# /opt/coreflow/core/telegram_bot/alerts/notify.py
# Telegram-Benachrichtigungssystem für CoreFlow (FTMO-konform)

import os
import requests
from datetime import datetime
from cryptography.fernet import Fernet

# ========== 🔐 Encrypted .env laden ==========
def load_encrypted_env(enc_path="/opt/coreflow/.env.enc", key_path="/opt/coreflow/infra/vault/encryption.key"):
    try:
        with open(key_path, "rb") as f:
            key = f.read()
        fernet = Fernet(key)

        with open(enc_path, "rb") as f:
            decrypted = fernet.decrypt(f.read()).decode()

        for line in decrypted.splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

        return True
    except Exception as e:
        print(f"[notify.py] Fehler beim .env-Entschlüsseln: {e}")
        return False

# ========== 📤 Telegram-Nachricht senden ==========
def send_telegram_message(message: str):
    try:
        if not os.getenv("TELEGRAM_TOKEN") or not os.getenv("TELEGRAM_CHAT_ID"):
            load_encrypted_env()

        token = os.getenv("TELEGRAM_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }

        response = requests.post(url, data=payload, timeout=10)

        if response.status_code == 200:
            print(f"📨 Telegram: ✅ {message}")
        else:
            print(f"❌ Telegram-Fehler ({response.status_code}): {response.text}")

    except Exception as e:
        print(f"❌ Telegram-Benachrichtigung fehlgeschlagen: {e}")

# ========== 📦 Beispiel-Aufruf ==========
if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    send_telegram_message(f"🟢 CoreFlow Testmeldung {now}")
