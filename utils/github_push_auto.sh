#!/bin/bash
# CoreFlow GitHub Auto-Push Script – Telegram + Log Rotation Edition
# Version: 1.3.0
# Autor: CoreFlow KI Commander
# Speicherort: /opt/coreflow/utils/github_push_auto.sh

# ---- Konfiguration ----
REPO_DIR="/opt/coreflow"
LOG_FILE="/var/log/coreflow_autopush.log"
MAX_LOG_SIZE=1048576 # 1MB
BRANCH="main"
REMOTE="origin"

# ---- .env laden ----
if [ -f "$REPO_DIR/.env" ]; then
    source "$REPO_DIR/.env"
fi

# ---- Initialisierung ----
exec > >(tee -a "$LOG_FILE") 2>&1
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
HOSTNAME=$(hostname)

echo "===== Auto-Push Started at $TIMESTAMP ====="

# ---- Hilfsfunktionen ----
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

rotate_logs() {
    if [ -f "$LOG_FILE" ] && [ $(stat -c%s "$LOG_FILE") -gt $MAX_LOG_SIZE ]; then
        mv "$LOG_FILE" "${LOG_FILE}.1"
        log "🔁 Rotated log file"
    fi
}

send_telegram() {
    local MSG="$1"
    local TOKEN="$TELEGRAM_BOT_TOKEN"
    local CHAT_ID="$TELEGRAM_CHAT_ID"

    if [ -z "$TOKEN" ] || [ -z "$CHAT_ID" ]; then
        log "⚠️ Telegram-Konfiguration fehlt"
        return
    fi

    curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendMessage" \
        -d chat_id="${CHAT_ID}" \
        -d text="$MSG" \
        -d parse_mode="Markdown" >/dev/null
}

# ---- Ausführung ----
rotate_logs

log "📁 Wechsle zu $REPO_DIR"
cd "$REPO_DIR" || {
    log "❌ Fehler: Kein Zugriff auf $REPO_DIR"
    send_telegram "❌ *GitHub Push fehlgeschlagen* (`cd error`) auf \`$HOSTNAME\` um \`$TIMESTAMP\`"
    exit 1
}

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "❌ Kein Git-Repository"
    send_telegram "❌ *GitHub Push fehlgeschlagen* (`not a git repo`) auf \`$HOSTNAME\` um \`$TIMESTAMP\`"
    exit 1
fi

if git diff --quiet && git diff --cached --quiet; then
    log "✅ Keine Änderungen erkannt"
    send_telegram "ℹ️ *Keine Änderungen zu pushen* auf \`$HOSTNAME\` um \`$TIMESTAMP\`"
    exit 0
fi

log "🔍 Änderungen gefunden:"
git status --short

log "⬆️ Staging aller Änderungen"
git add . || {
    log "❌ git add fehlgeschlagen"
    send_telegram "❌ *git add fehlgeschlagen* auf \`$HOSTNAME\` um \`$TIMESTAMP\`"
    exit 1
}

COMMIT_MSG="🔄 Auto-Commit: $TIMESTAMP"
log "💾 Commit: $COMMIT_MSG"
git commit -m "$COMMIT_MSG" || {
    log "❌ Commit fehlgeschlagen"
    send_telegram "❌ *Commit fehlgeschlagen* auf \`$HOSTNAME\` um \`$TIMESTAMP\`"
    exit 1
}

log "🔃 Rebase mit Remote ($REMOTE/$BRANCH)"
git pull --rebase "$REMOTE" "$BRANCH" || {
    log "❌ Rebase fehlgeschlagen"
    send_telegram "❌ *Rebase fehlgeschlagen* auf \`$HOSTNAME\` um \`$TIMESTAMP\`"
    exit 1
}

log "🚀 Push nach $REMOTE/$BRANCH"
git push "$REMOTE" "$BRANCH" || {
    log "❌ Push fehlgeschlagen"
    send_telegram "❌ *Push fehlgeschlagen* auf \`$HOSTNAME\` um \`$TIMESTAMP\`"
    exit 1
}

log "✅ Erfolgreich gepusht"
send_telegram "✅ *CoreFlow Push erfolgreich* auf \`$HOSTNAME\` um \`$TIMESTAMP\`"

echo "===== Auto-Push Completed at $(date +'%Y-%m-%d %H:%M:%S') ====="
