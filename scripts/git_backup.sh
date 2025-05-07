#!/bin/bash
TIMESTAMP=$(date +%Y%m%d)
DEST="/opt/backups/coreflow-backup-${TIMESTAMP}.bundle"

echo "🔄 Erstelle Git-Bundle: $DEST"
git -C /opt/coreflow bundle create "$DEST" --all
