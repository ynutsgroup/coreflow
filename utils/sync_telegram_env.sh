#!/bin/bash
# /opt/coreflow/utils/sync_telegram_env.sh
# 🔄 Synchronisiert .env.telegram von Windows → Linux via SSH
# Speichert unter ${LINUX_TARGET_PATH}/.env.telegram

# === Konfiguration ===
CONFIG_DIR="$(dirname "$(realpath "$0")")"
CONFIG_FILE="${CONFIG_DIR}/.env.sync"

# === Logging ===
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') | $1" | tee -a "${LOG_FILE:-/dev/null}"
}

# === Konfigurationsprüfung ===
validate_config() {
    local missing_vars=()
    local required_vars=("WIN_IP" "WIN_USER" "SSH_KEY_PATH" "WIN_PATH" "LINUX_TARGET_PATH")

    for var in "${required_vars[@]}"; do
        [ -z "${!var}" ] && missing_vars+=("$var")
    done

    if [ ${#missing_vars[@]} -gt 0 ]; then
        log "❌ Fehlende Konfiguration: ${missing_vars[*]}"
        return 1
    fi

    [ -f "$SSH_KEY_PATH" ] || {
        log "❌ SSH-Key nicht gefunden: $SSH_KEY_PATH"
        return 1
    }

    return 0
}

# === Hauptlogik ===
main() {
    if [ ! -f "$CONFIG_FILE" ]; then
        log "❌ Konfigurationsdatei nicht gefunden: $CONFIG_FILE"
        exit 1
    fi

    # shellcheck disable=SC1090
    source "$CONFIG_FILE" || {
        log "❌ Fehler beim Laden der Konfiguration"
        exit 1
    }

    validate_config || exit 1

    local source_file="${WIN_PATH}.env.telegram"
    local target_file="${LINUX_TARGET_PATH}/.env.telegram"
    local ssh_opts=("-i" "$SSH_KEY_PATH" "-P" "${SSH_PORT:-22}" "-o" "ConnectTimeout=10")

    mkdir -p "$LINUX_TARGET_PATH" || {
        log "❌ Kann Zielverzeichnis nicht erstellen: $LINUX_TARGET_PATH"
        exit 1
    }

    log "🔄 Starte Sync: ${WIN_USER}@${WIN_IP}:${source_file} → ${target_file}"

    if scp "${ssh_opts[@]}" "${WIN_USER}@${WIN_IP}:${source_file}" "$target_file"; then
        log "✅ Erfolgreich synchronisiert"
        chmod 600 "$target_file" && log "🔒 Dateiberechtigungen angepasst"
    else
        log "❌ Sync fehlgeschlagen (Code: $?)"
        exit 1
    fi
}

main "$@"
