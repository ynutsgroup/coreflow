#!/bin/bash
# /opt/coreflow/utils/master_sync.sh
# 🔄 Hybrid Sync-Skript für CoreFlow-Umgebungsdateien (Linux → Windows)

CONFIG_DIR="$(dirname "$(realpath "$0")")"
CONFIG_FILE="${CONFIG_DIR}/.env.sync"
LOG_FILE="${LOG_FILE:-/opt/coreflow/logs/master_sync.log}"
SYNC_FILES=(".env" ".env.key" ".env.telegram")

init_logging() {
    mkdir -p "$(dirname "$LOG_FILE")"
    exec 3>&1 4>&2
    exec > >(tee -a "$LOG_FILE") 2>&1
}

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1"
}

validate_environment() {
    local required_vars=("WIN_IP" "WIN_USER" "SSH_KEY_PATH" "WIN_PATH" "LINUX_TARGET_PATH")
    local missing_vars=()
    for var in "${required_vars[@]}"; do [ -z "${!var}" ] && missing_vars+=("$var"); done
    [ ${#missing_vars[@]} -gt 0 ] && { log "❌ Fehlende Konfiguration: ${missing_vars[*]}"; return 1; }
    [[ "$WIN_PATH" != */ ]] && log "⚠️  WIN_PATH sollte mit / enden"
    [ -f "$SSH_KEY_PATH" ] || { log "❌ SSH-Key fehlt: $SSH_KEY_PATH"; return 1; }
    return 0
}

sync_file() {
    local file="$1"
    local src="${LINUX_TARGET_PATH%/}/${file}"
    local dst="${WIN_USER}@${WIN_IP}:${WIN_PATH%/}/${file}"
    local ssh_cmd=("scp" "${SSH_OPTS[@]}" "$src" "$dst")
    log "🔄 Sync: $file → $dst"
    if "${ssh_cmd[@]}"; then
        log "✅ $file erfolgreich übertragen"
    else
        log "⚠️  Fehler bei $file"
        return 1
    fi
}

send_notification() {
    [ "$TELEGRAM_ENABLED" = "True" ] || return 0
    local message="$1"
    curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
        -d chat_id="$TELEGRAM_CHAT_ID" \
        -d text="$message" \
        -d parse_mode="Markdown" && log "📨 Telegram-Benachrichtigung gesendet"
}

perform_backup() {
    [ "$BACKUP_ENABLED" = "True" ] || return 0
    mkdir -p "$BACKUP_DIR"
    local ts
    ts=$(date '+%Y%m%d_%H%M%S')
    local archive="${BACKUP_DIR}/coreflow_env_backup_${ts}.tar.gz"
    tar czf "$archive" -C "$LINUX_TARGET_PATH" "${SYNC_FILES[@]}" && log "💾 Backup erstellt: $archive"
    find "$BACKUP_DIR" -type f -mtime +${BACKUP_RETENTION:-7} -delete
}

rotate_logs() {
    find "$(dirname "$LOG_FILE")" -name "$(basename "$LOG_FILE")" -type f -mtime +${MAX_LOG_DAYS:-30} -delete
}

main() {
    init_logging
    log "=== START CROSS-PLATFORM SYNC ==="

    [ -f "$CONFIG_FILE" ] || { log "❌ $CONFIG_FILE fehlt"; send_notification "❌ Config fehlt"; exit 1; }
    source "$CONFIG_FILE" || { log "❌ Fehler beim Laden"; send_notification "❌ Fehler .env.sync"; exit 1; }
    validate_environment || { send_notification "❌ Invalid .env.sync"; exit 1; }

    SSH_OPTS=("-i" "$SSH_KEY_PATH" "-P" "${SSH_PORT:-22}" "${SSH_OPTIONS[@]}")

    local errors=0
    for file in "${SYNC_FILES[@]}"; do
        [ -f "${LINUX_TARGET_PATH%/}/${file}" ] || { log "⚠️  Datei fehlt: $file"; ((errors++)); continue; }
        sync_file "$file" || ((errors++))
    done

    perform_backup
    rotate_logs

    if [ $errors -eq 0 ]; then
        log "✅ ALLE DATEIEN ÜBERTRAGEN"
        send_notification "✅ Sync erfolgreich $(date '+%H:%M')"
    else
        log "⚠️  $errors Fehler aufgetreten"
        send_notification "⚠️  Sync mit $errors Fehler(n)"
    fi

    log "⏹️  Beendet mit Code: $errors"
    log "=== SYNC DONE ==="
    exit $errors
}

main "$@"
