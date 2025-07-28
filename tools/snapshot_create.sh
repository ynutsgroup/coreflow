#!/bin/bash
DATE=$(date +%Y-%m-%d_%H-%M)
DEST="/opt/coreflow/_daily/coreflow_cfm_sync_$DATE.zip"

mkdir -p /opt/coreflow/_daily

/opt/coreflow/tools/make_manifest.sh

zip -r "$DEST" \
  /opt/coreflow/structure_manifest.json \
  /opt/coreflow/notes/todo.md \
  /opt/coreflow/logs/watchdog.log > /dev/null 2>&1

echo "📦 Snapshot erstellt: $DEST"

