#!/bin/bash
# CoreFlow GitHub Auto-Push Script mit Telegram-Notifier
# Version: 1.3.0

# ---- Konfiguration ----
REPO_DIR="/opt/coreflow"
LOG_FILE="/var/log/coreflow_autopush.log"
MAX_LOG_SIZE=1048576
BRANCH="main"
REMOTE="origin"
ENV_FILE="$REPO_DIR/.env"

# ---- Telegram Variablen laden ----
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "[WARN] .env nicht gefunden – Telegram nicht aktiviert"
fi

# ---- Initialisierung ----
exec > >(tee -a "$LOG_FILE") 2>&1
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
echo "===== Auto-Push Started at $TIMESTAMP ====="

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

rotate_logs() {
    if [ -f "$LOG_FILE" ] && [ $(stat -c%s "$LOG_FILE") -gt $MAX_LOG_SIZE ]; then
        mv "$LOG_FILE" "${LOG_FILE}.1"
        log "📦 Logfile rotiert"
    fi
}

send_telegram() {
    if [[ "$TELEGRAM_ENABLED" == "True" && -n "$TELEGRAM_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
        -d chat_id="$TELEGRAM_CHAT_ID" \
        -d text="✅ CoreFlow GitHub Auto-Push erfolgreich: $TIMESTAMP"
    else
        log "⚠️ Telegram-Konfiguration fehlt"
    fi
}

# ---- Hauptlogik ----
rotate_logs

log "📁 Wechsle zu $REPO_DIR"
cd "$REPO_DIR" || { log "❌ Fehler beim Verzeichniswechsel"; exit 1; }

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "❌ Kein Git-Repository"
    exit 1
fi

if git diff --quiet && git diff --cached --quiet; then
    log "✅ Keine Änderungen gefunden"
    exit 0
fi

log "🔍 Änderungen gefunden:"
git status --short

log "⬆️ Staging aller Änderungen"
git add .

COMMIT_MSG="🔄 Auto-Commit: $TIMESTAMP"
log "💾 Commit: $COMMIT_MSG"
git commit -m "$COMMIT_MSG" || { log "❌ Commit fehlgeschlagen"; exit 1; }

log "🔃 Rebase mit Remote ($REMOTE/$BRANCH)"
git pull --rebase "$REMOTE" "$BRANCH" || { log "❌ Rebase fehlgeschlagen"; exit 1; }

log "🚀 Push nach $REMOTE/$BRANCH"
git push "$REMOTE" "$BRANCH" || { log "❌ Push fehlgeschlagen"; exit 1; }

log "✅ Erfolgreich gepusht"
send_telegram

echo "===== Auto-Push Completed at $(date +'%Y-%m-%d %H:%M:%S') ====="
