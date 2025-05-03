#!/bin/bash
# /opt/coreflow/scripts/backup_coreflow.sh

BACKUP_DIR="/opt/backups"
TIMESTAMP=$(date +%Y%m%d)
ZIPFILE="$BACKUP_DIR/coreflow-project-$TIMESTAMP.zip"
BUNDLEFILE="/opt/coreflow/coreflow-backup-$TIMESTAMP.bundle"

mkdir -p "$BACKUP_DIR"

echo "🔄 Erstelle Git-Bundle..."
cd /opt/coreflow || exit 1
git bundle create "$BUNDLEFILE" --all

echo "📦 Erstelle ZIP-Archiv..."
cd /opt || exit 1
zip -r "$ZIPFILE" coreflow -x "*.git*" "*/venv/*" "*/__pycache__/*"

echo "🧹 Entferne alte Backups..."
find "$BACKUP_DIR" -name "coreflow-project-*.zip" -mtime +7 -delete

echo "✅ Backup abgeschlossen: $ZIPFILE"
