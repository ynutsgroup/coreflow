#!/bin/bash
# CoreFlow Watchdog Reset – institutionell mit Telegram
# Fix Windows line endings (falls vorhanden)
[ -f /opt/coreflow/.env ] && dos2unix /opt/coreflow/.env 2>/dev/null

WATCHDOG_SCRIPT="/opt/coreflow/watchdogs/watchdog_async_nohup.py"
LOG_FILE="/opt/coreflow/logs/watchdog_async_nohup.out"
HISTORY_FILE="/opt/coreflow/logs/restart_history.json"
ENV_FILE="/opt/coreflow/.env"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# === .env sicher laden ===
set -a
. "$ENV_FILE"
set +a

echo "$TIMESTAMP | 🛑 Watchdog wird gestoppt ..."
pkill -f "$WATCHDOG_SCRIPT"

echo "$TIMESTAMP | 🧹 Restart-History wird gelöscht ..."
echo "[]" > "$HISTORY_FILE"

echo "$TIMESTAMP | 🚀 Starte Watchdog neu ..."
nohup python3 "$WATCHDOG_SCRIPT" > "$LOG_FILE" 2>&1 &

echo "$TIMESTAMP | ✅ Watchdog wurde erfolgreich zurückgesetzt."

# === Telegram-Benachrichtigung ===
if [ "$TELEGRAM_ENABLED" = "True" ] && [ -n "$TELEGRAM_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
  curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
    -d chat_id="${TELEGRAM_CHAT_ID}" \
    -d text="🔁 *CoreFlow Watchdog wurde zurückgesetzt und neu gestartet* – $TIMESTAMP" \
    -d parse_mode="Markdown"
fi
