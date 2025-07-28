#!/bin/bash
DATE=$(date +%Y-%m-%d_%H-%M)
DEST="/opt/coreflow/_shutdown/coreflow_shutdown_$DATE.zip"
SOURCE="/opt/coreflow"

mkdir -p "$(dirname "$DEST")"
zip -r "$DEST" "$SOURCE/src" "$SOURCE/logs" "$SOURCE/.env" "$SOURCE/notes" > /dev/null 2>&1

echo "[✔] Shutdown-Backup erstellt: $DEST"
