#!/bin/bash

# CoreFlow CF4 Migrationsskript (Fehlerresistent)
set -e

echo "📦 Starte Migration in CoreFlow_CF4 Struktur..."

mkdir -p /opt/coreflow/core/{ai/inference,ai/training,ai/validation,ai/registry}
mkdir -p /opt/coreflow/core/{emitter,bridge,watchdog/healthchecks,watchdog/autoheal}
mkdir -p /opt/coreflow/core/{risk_manager,telegram_bot/alerts,telegram_bot/admin}
mkdir -p /opt/coreflow/core/{utils,infra/docker,infra/systemd,infra/vault}
mkdir -p /opt/coreflow/{monitoring/grafana,monitoring/prometheus,monitoring/mlflow,monitoring/drift_reports}
mkdir -p /opt/coreflow/{backup/incremental,backup/snapshots,logs,signals/sent,signals/archive}

# Funktion: move_if_exists
move_if_exists() {
  [ -f "$1" ] && mv "$1" "$2" && echo "✅ Verschoben: $1 → $2"
}

# Migration
move_if_exists /opt/coreflow/coreflow_main.py               /opt/coreflow/core/
move_if_exists /opt/coreflow/coreflow_watchdog.py           /opt/coreflow/core/watchdog/
move_if_exists /opt/coreflow/signal_emitter.py              /opt/coreflow/core/emitter/
move_if_exists /opt/coreflow/telegram_bot.py                /opt/coreflow/core/telegram_bot/admin/
move_if_exists /opt/coreflow/notify.py                      /opt/coreflow/core/telegram_bot/alerts/
move_if_exists /opt/coreflow/encryption.py                  /opt/coreflow/core/utils/
move_if_exists /opt/coreflow/gpu_monitor.py                 /opt/coreflow/core/watchdog/healthchecks/
move_if_exists /opt/coreflow/diagnostic_commander.py        /opt/coreflow/core/utils/
move_if_exists /opt/coreflow/reorganize_coreflow.py         /opt/coreflow/core/utils/
move_if_exists /opt/coreflow/requirements.txt               /opt/coreflow/core/infra/
move_if_exists /opt/coreflow/ai_trading.py                  /opt/coreflow/core/ai/inference/
move_if_exists /opt/coreflow/message_sender.py              /opt/coreflow/core/utils/
move_if_exists /opt/coreflow/send_telegram_message.py       /opt/coreflow/core/telegram_bot/alerts/
move_if_exists /opt/coreflow/git_sync.py                    /opt/coreflow/backup/

# Batch-Migration
mv /opt/coreflow/*.service        /opt/coreflow/core/infra/systemd/ 2>/dev/null || true
mv /opt/coreflow/*.log            /opt/coreflow/logs/ 2>/dev/null || true
mv /opt/coreflow/*.tar.gz         /opt/coreflow/backup/ 2>/dev/null || true
mv /opt/coreflow/risk_management/* /opt/coreflow/core/risk_manager/ 2>/dev/null || true

echo "🎉 Migration abgeschlossen. Struktur jetzt aktiv."
