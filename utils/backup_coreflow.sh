#!/bin/bash

# ===== CoreFlow Linux Backup Script =====
# Erstellt ein ZIP-Archiv des gesamten Projekts mit Zeitstempel

SRC="/opt/coreflow"
BACKUP_DIR="/opt/coreflow/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M")
DEST="$BACKUP_DIR/coreflow_$TIMESTAMP.zip"
LOGFILE="$BACKUP_DIR/backup_log.txt"

mkdir -p "$BACKUP_DIR"

echo "🕒 Starte Backup: $TIMESTAMP" | tee -a "$LOGFILE"
zip -r "$DEST" "$SRC" -x "*.venv/*" "*.pyc" "__pycache__/*" > /dev/null

if [[ $? -eq 0 ]]; then
    echo "✅ Backup erfolgreich: $DEST" | tee -a "$LOGFILE"
    echo "✅ Backup abgeschlossen: $DEST"
else
    echo "❌ Backup FEHLGESCHLAGEN: $DEST" | tee -a "$LOGFILE"
    echo "❌ Fehler beim Backup!"
fi
