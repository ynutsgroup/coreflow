#!/bin/bash
DATE=$(date +%Y-%m-%d)
DEST="/opt/coreflow/_daily/coreflow_sync_$DATE.zip"
SOURCE="/opt/coreflow"

mkdir -p "$(dirname "$DEST")"
find "$(dirname "$DEST")" -type f -name "coreflow_sync_*.zip" -mtime +7 -delete
zip -r "$DEST" "$SOURCE/src" "$SOURCE/logs" "$SOURCE/.env" "$SOURCE/notes" > /dev/null 2>&1

echo "[Reminder @ $DATE] CoreFlow Snapshot gespeichert: $DEST"
