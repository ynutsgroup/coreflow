#!/bin/bash
# CoreFlow – Unkomprimiertes, selektives Backup

SOURCE="/opt/coreflow"
TARGET="/mnt/mirror/coreflow_custom_backup"
DATE=$(date +%Y-%m-%d_%H-%M)
LOG="/var/log/coreflow_custom_backup_${DATE}.log"

mkdir -p "$TARGET/$DATE"
mkdir -p "$(dirname "$LOG")"

MY_FILES=(
  "tools/make_manifest.sh"
  "tools/cfm_secure_snapshot.sh"
  "tools/cfm_show_tree.py"
  "tools/shutdown_backup.sh"
  "notes/todo.md"
  "structure_manifest.json"
)

echo "🔁 Starte unkomprimiertes Backup..." | tee -a "$LOG"

for file in "${MY_FILES[@]}"; do
  src_path="$SOURCE/$file"
  if [[ -e "$src_path" ]]; then
    rsync -a --relative "$SOURCE/./$file" "$TARGET/$DATE/" | tee -a "$LOG"
  else
    echo "⚠️ Datei fehlt: $file" | tee -a "$LOG"
  fi
done

echo "✅ Backup abgeschlossen: $TARGET/$DATE" | tee -a "$LOG"
tree "$TARGET/$DATE" | tee -a "$LOG"

exit 0
