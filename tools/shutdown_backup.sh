#!/bin/bash
DATE=$(date +%Y-%m-%d_%H-%M)
DEST="/opt/coreflow/_shutdown/coreflow_cfm_shutdown_$DATE.zip"

mkdir -p /opt/coreflow/_shutdown

sudo /opt/coreflow/tools/make_manifest.py

zip -r "$DEST" \
  /opt/coreflow/structure_manifest.json \
  /opt/coreflow/notes/todo.md \
  /opt/coreflow/logs/watchdog.log > /dev/null 2>&1

echo "Backup beim Shutdown erstellt: $DEST"
