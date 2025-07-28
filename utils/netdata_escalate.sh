#!/bin/bash
METRIC="$1"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "[NETDATA ESCALATION] Triggered by: $METRIC at $TIMESTAMP" >> /opt/coreflow/logs/escalation.log

# FTMO-konforme Maßnahme: TEST-Modus erzwingen
/opt/coreflow/commander_control.py set-mode TEST

# Telegram-Alarm senden
/opt/coreflow/telegram_bot/send_alert.py "🚨 CoreFlow Risk Alert [$METRIC]\nSystem overloaded.\nSwitched to MODE_TEST at $TIMESTAMP."
