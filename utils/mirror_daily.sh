#!/bin/bash

# ============================================================================
# 📦 CoreFlow Backup-Skript – mirror_daily.sh
# 🔁 Automatisches Spiegel-Backup von /opt/coreflow nach /mnt/mirror
# 📨 Mit Telegram-Benachrichtigung bei Erfolg & Fehler
# 📅 Geplant für: 23:00, 00:00, 02:00, 03:00 via Cron
# ============================================================================

TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
LOGFILE="/mnt/mirror/cron_backup.log"
SIGNATURE="/mnt/mirror/backup_signatur.txt"
SRC="/opt/coreflow/"
DST="/mnt/mirror/"

# 🔐 Telegram Konfiguration aus .env entschlüsseln
source /opt/coreflow/.env.decrypted 2>/dev/null || true

TELEGRAM_TOKEN="${TELEGRAM_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"

function send_telegram() {
    local msg="$1"
    if [[ -n "$TELEGRAM_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
            -d chat_id="${TELEGRAM_CHAT_ID}" \
            -d text="${msg}" \
            -d parse_mode="Markdown" >/dev/null 2>&1
    fi
}

echo "🔁 Starte Backup um $TIMESTAMP" >> "$LOGFILE"

# 🧭 Mountpoint prüfen
if mountpoint -q "$DST"; then
    echo "✅ Mountpoint $DST gefunden." >> "$LOGFILE"
else
    echo "❌ Fehler: $DST ist nicht eingehängt." >> "$LOGFILE"
    send_telegram "❌ *CoreFlow Backup Fehler*: Mountpoint \`$DST\` nicht gefunden um $TIMESTAMP."
    exit 1
fi

# 🧹 Backup durchführen
rsync -a --delete --info=progress2 \
    --exclude='lost+found' \
    --exclude='*.DISABLED' \
    "$SRC" "$DST" >> "$LOGFILE" 2>&1

# 📊 Prüfung Ergebnis
if [ $? -eq 0 ]; then
    echo "✅ Backup erfolgreich abgeschlossen um $TIMESTAMP" >> "$LOGFILE"
    echo "✅ Letztes Backup: $TIMESTAMP" > "$SIGNATURE"
    send_telegram "✅ *CoreFlow Backup abgeschlossen* um\n\`$TIMESTAMP\`"
else
    echo "⚠️ Fehler beim Backup um $TIMESTAMP" >> "$LOGFILE"
    send_telegram "⚠️ *CoreFlow Backup-Fehler* um\n\`$TIMESTAMP\`"
fi
