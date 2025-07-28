#!/bin/bash
# 🧹 CoreFlow Log Cleaner – Watchdog

cd /opt/coreflow/logs || exit

echo "⏳ Entferne leere Watchdog-Dateien..."
find . -name 'watchdog_2025*.log' -size 0 -delete

echo "🧹 Entferne alte komprimierte Watchdog-Logs..."
rm -f watchdog.log.[2-9].gz watchdog.log.1[0-9].gz 2>/dev/null

echo "✅ Bereinigt. Aktuelle Logs bleiben erhalten."
