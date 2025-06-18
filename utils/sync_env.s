#!/bin/bash
# /opt/coreflow/utils/sync_env.sh
# 🔐 Sync .env & .env.key Linux → Windows via SCP (passwordless)

# === KONFIGURATION ===
WIN_USER="${WIN_USER:-Administrator}"
WIN_IP="${WIN_IP:-192.168.178.20}"
WIN_PATH="${WIN_PATH:-/c:/CoreFlow/}"
LINUX_ENV_PATH="${LINUX_ENV_PATH:-/opt/coreflow/}"
SSH_KEY="${SSH_KEY:-/home/coreadmin/.ssh/id_rsa_sync}"

# === LOGGING ===
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1"
}

# === VORBEDINGUNGEN ===
check_prerequisites() {
    if [ ! -d "$LINUX_ENV_PATH" ]; then
        log "❌ Quellverzeichnis fehlt: $LINUX_ENV_PATH"
        exit 1
    fi
    if [ ! -f "$SSH_KEY" ]; then
        log "❌ SSH Key fehlt: $SSH_KEY"
        exit 1
    fi
    for file in .env .env.key; do
        if [ ! -f "$LINUX_ENV_PATH/$file" ]; then
            log "❌ Datei fehlt: $file"
            exit 1
        fi
    done
}

# === BACKUP ===
create_backups() {
    cd "$LINUX_ENV_PATH" || exit 1
    for file in .env .env.key; do
        cp "$file" "$file.bak" && log "📦 Backup erstellt: $file.bak"
    done
}

# === SYNC ===
sync_files() {
    cd "$LINUX_ENV_PATH" || exit 1
    for file in .env .env.key; do
        log "🔄 Sync $file → $WIN_USER@$WIN_IP:$WIN_PATH"
        if scp -i "$SSH_KEY" "$file" "$WIN_USER@$WIN_IP:$WIN_PATH$file"; then
            log "✅ $file synchronisiert"
        else
            log "❌ Fehler beim Sync von $file"
        fi
    done
}

# === MAIN ===
main() {
    check_prerequisites
    create_backups
    sync_files
}

main
