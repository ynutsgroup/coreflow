#!/bin/bash
# /opt/coreflow/utils/git_backup_push.sh
# 🔄 Secure Git Backup with Telegram Notifications

# === Configuration ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.gitbackup"

[ -f "$ENV_FILE" ] && source "$ENV_FILE" || {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | ❌ Konfigurationsdatei nicht gefunden: $ENV_FILE" | tee -a "${GIT_LOG_FILE:-/opt/coreflow/logs/git_backup.log}"
    exit 1
}

# Set defaults
REPO_DIR="${GIT_REPO_DIR:-/opt/coreflow}"
BRANCH="${GIT_BRANCH:-main}"
LOG_FILE="${GIT_LOG_FILE:-/opt/coreflow/logs/git_backup_$(date +%Y%m%d).log}"
MAX_LOG_DAYS="${MAX_LOG_DAYS:-30}"

# Security settings
umask "${UMASK:-0077}"
GIT_SSH_COMMAND="ssh -i '${GIT_IDENTITY:-~/.ssh/id_rsa}' -o IdentitiesOnly=yes"

# === Functions ===
init_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    find "$(dirname "$LOG_FILE")" -name "git_backup_*.log" -mtime +$MAX_LOG_DAYS -delete
}

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "$LOG_FILE"
}

send_notification() {
    [ "${TELEGRAM_ENABLED:-false}" = "true" ] || return 0
    
    local status="$1"
    local message="*Git Backup Report*\n\n"
    message+="• Status: $status\n"
    message+="• Time: $(date '+%Y-%m-%d %H:%M:%S')\n"
    message+="• Branch: $BRANCH\n"
    
    if [ "$status" = "SUCCESS" ]; then
        message+="• Changes:\n\`\`\`$(git -C "$REPO_DIR" status --short)\`\`\`"
    else
        message+="• Error: Siehe Log-Datei"
    fi

    curl -s -X POST "${TELEGRAM_API_URL:-https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage}" \
        -d chat_id="$TELEGRAM_CHAT_ID" \
        -d text="$message" \
        -d parse_mode="Markdown" >> "$LOG_FILE" 2>&1
}

git_backup() {
    cd "$REPO_DIR" || { log "❌ Repository-Verzeichnis nicht erreichbar"; return 1; }
    
    if [ -z "$(git status --porcelain)" ]; then
        log "ℹ️ Keine Änderungen für Backup"
        return 0
    fi

    log "🔄 Starte Git Backup"
    git add . || { log "❌ git add fehlgeschlagen"; return 1; }
    git commit -m "🔄 Auto-Backup: $(date '+%Y-%m-%d %H:%M:%S')" || { log "❌ git commit fehlgeschlagen"; return 1; }
    git push origin "$BRANCH" || { log "❌ git push fehlgeschlagen"; return 1; }
    
    log "✅ Backup erfolgreich"
    return 0
}

# === Main Execution ===
main() {
    init_logging
    log "=== STARTE GIT BACKUP ==="
    
    if git_backup; then
        send_notification "✅ ERFOLGREICH"
        log "=== BACKUP ABGESCHLOSSEN ==="
        exit 0
    else
        send_notification "❌ FEHLGESCHLAGEN"
        log "=== BACKUP FEHLGESCHLAGEN ==="
        exit 1
    fi
}

main "$@" check
