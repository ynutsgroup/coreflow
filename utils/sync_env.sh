#!/bin/bash
# /opt/coreflow/utils/sync_env.sh
# 🔐 Synchronisiert .env & .env.key zwischen Linux und Windows

WIN_USER="Administrator"
WIN_IP="os.getenv('REDIS_HOST')"
WIN_PATH="/C:/CoreFlow/"
LINUX_ENV_PATH="/opt/coreflow/"
SSH_KEY="$HOME/.ssh/id_rsa_sync"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1"
}

check_prerequisites() {
    [ -d "$LINUX_ENV_PATH" ] || {
        log "❌ Fehler: Verzeichnis $LINUX_ENV_PATH existiert nicht"
        exit 1
    }

    [ -f "$SSH_KEY" ] || {
        log "❌ Fehler: SSH-Key $SSH_KEY nicht gefunden"
        exit 1
    }

    for file in .env .env.key; do
        [ -f "$LINUX_ENV_PATH/$file" ] || {
            log "❌ Fehler: $LINUX_ENV_PATH/$file nicht gefunden"
            exit 1
        }
    done
}

create_backups() {
    cd "$LINUX_ENV_PATH" || exit 1
    for file in .env .env.key; do
        cp "$file" "$file.bak" && log "🔁 Backup von $file erstellt"
    done
}

sync_files() {
    local status=0

    # Verbindung prüfen
    if ! ssh -i "$SSH_KEY" "$WIN_USER@$WIN_IP" "whoami" >/dev/null 2>&1; then
        log "❌ SSH-Verbindungstest fehlgeschlagen"
        return 1
    fi

    # Zielverzeichnis sicherstellen
    ssh -i "$SSH_KEY" "$WIN_USER@$WIN_IP" "mkdir -p $WIN_PATH"

    # Dateiübertragung
    for file in .env .env.key; do
        log "🔄 Synchronisiere $file → Windows"
        if scp -i "$SSH_KEY" "$LINUX_ENV_PATH$file" "$WIN_USER@$WIN_IP:$WIN_PATH$file"; then
            log "✅ $file erfolgreich synchronisiert"
        else
            log "❌ Fehler bei $file"
            status=1
        fi
    done

    return $status
}

main() {
    log "=== Starte Synchronisation ==="
    check_prerequisites
    create_backups

    if sync_files; then
        log "✨ Synchronisation erfolgreich abgeschlossen"
    else
        log "⚠️  Synchronisation mit Fehlern abgeschlossen"
        exit 1
    fi
}

main
