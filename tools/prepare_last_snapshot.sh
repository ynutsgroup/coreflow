#!/bin/bash
UPLOAD_DIR="/opt/coreflow/_upload_ready"
DAILY="/opt/coreflow/_daily"
SHUTDOWN="/opt/coreflow/_shutdown"

mkdir -p "$UPLOAD_DIR"

# Letztes ZIP (egal ob daily oder shutdown)
LAST_FILE=$(ls -t $DAILY/coreflow_sync_*.zip $SHUTDOWN/coreflow_shutdown_*.zip 2>/dev/null | head -n 1)

if [[ -f "$LAST_FILE" ]]; then
    cp "$LAST_FILE" "$UPLOAD_DIR/"
    echo "✅ Letztes CoreFlow-Snapshot bereitgestellt für Upload:"
    echo "→ $UPLOAD_DIR/$(basename "$LAST_FILE")"
else
    echo "⚠️ Kein Backup gefunden."
fi
