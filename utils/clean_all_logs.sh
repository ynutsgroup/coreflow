#!/bin/bash
# 🧹 CoreFlow Log Cleaner – Full Log Maintenance

LOG_DIR="/opt/coreflow/logs"
cd "$LOG_DIR" || exit

echo "📦 CoreFlow Log Cleanup gestartet..."

# Leere Log-Dateien löschen
echo "⏳ Entferne leere Log-Dateien..."
find . -name '*.log' -size 0 -exec rm -v {} \;

# Alte komprimierte Logs löschen (.gz außer letzte 2)
echo "📦 Lösche komprimierte Logs, außer letzte 2:"
for logbase in $(ls *.log.*.gz 2>/dev/null | cut -d'.' -f1 | sort -u); do
    ls -t ${logbase}.log.*.gz | tail -n +3 | xargs -r rm -v
done

# Optionale Cleanup-Meldung
echo "✅ CoreFlow Logs bereinigt am $(date '+%Y-%m-%d %H:%M')"
