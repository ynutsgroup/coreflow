#!/bin/bash
# CFM Secure Snapshot – Kompakte Freigabekette

DATE=$(date +%Y-%m-%d_%H-%M)
DEST="/opt/coreflow/_upload_ready/coreflow_cfm_sync_$DATE.zip"
SOURCE="/opt/coreflow"
MANIFEST="$SOURCE/structure_manifest.json"

echo "🔄 [1/4] Erzeuge Strukturbaum..."
/opt/coreflow/tools/make_manifest.sh
if [[ ! -f "$MANIFEST" ]]; then
  echo "❌ Manifest fehlt – Freigabekette gestoppt"
  exit 1
fi
echo "✔️ Strukturbaum erstellt"

echo "📥 [2/4] Prüfe Status und Logs..."
NOTES="$SOURCE/notes/todo.md"
[[ ! -f "$NOTES" ]] && echo "⚠️ Hinweis: notes/todo.md fehlt"

LOGFILE="$SOURCE/logs/watchdog.log"
[[ ! -f "$LOGFILE" ]] && echo "⚠️ Hinweis: logs/watchdog.log fehlt"

echo "📦 [3/4] Erzeuge Snapshot ZIP..."
mkdir -p "$(dirname "$DEST")"
zip -r "$DEST" \
  "$MANIFEST" \
  "$NOTES" \
  "$LOGFILE" \
  > /dev/null 2>&1

echo "📤 [4/4] Bereit zur Übertragung:"
echo "✅ ZIP: $DEST"
