import requests

def send_telegram_message(message: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.warning("⚠️ TELEGRAM_BOT_TOKEN oder TELEGRAM_CHAT_ID nicht gesetzt")
        return

    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        response = requests.post(url, data=payload, timeout=10)

        if response.status_code == 200:
            logger.info("📬 Telegram-Nachricht erfolgreich gesendet")
        else:
            logger.error(f"❌ Fehler beim Senden der Telegram-Nachricht: {response.text}")
    except Exception as e:
        logger.error(f"❌ Telegram-Sendefehler: {str(e)}")

# ... innerhalb der sync_git() Funktion, nach erfolgreichem Push:
if changes:
    logger.info("✅ Git Sync erfolgreich abgeschlossen")
    send_telegram_message("✅ Git Backup erfolgreich abgeschlossen – Änderungen synchronisiert.")
else:
    logger.info("🟢 Keine Änderungen zum Synchronisieren gefunden")
    send_telegram_message("🟢 Git Backup abgeschlossen – keine Änderungen gefunden.")
