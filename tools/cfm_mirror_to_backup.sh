#!/bin/bash
# CoreFlow – Selektives Skript-Backup

SOURCE="/opt/coreflow"
TARGET="/mnt/mirror/coreflow_custom_backup"
DATE=$(date +%Y-%m-%d_%H-%M)
LOG="/var/log/coreflow_custom_backup_${DATE}.log"

# 🔒 Sicherstellen, dass log-Verzeichnis existiert
mkdir -p "$(dirname "$LOG")"

# 📋 Wichtige Dateien definieren (nur deine Kernskripte)
MY_FILES=(
  "tools/make_manifest.sh"
  "tools/cfm_secure_snapshot.sh"
  "tools/cfm_show_tree.py"
  "tools/shutdown_backup.sh"
  "notes/todo.md"
  "structure_manifest.json"
)

echo "🔁 Starte selektives Backup..." | tee -a "$LOG"

# 📁 Zielstruktur vorbereiten
mkdir -p "$TARGET/$DATE"

# 🔍 Dateisicherung
for file in "${MY_FILES[@]}"; do
  src_path="$SOURCE/$file"
  if [[ -e "$src_path" ]]; then
    rsync -a --relative "$SOURCE/./$file" "$TARGET/$DATE/" | tee -a "$LOG"
  else
    echo "⚠️ Fehlt: $file" | tee -a "$LOG"
  fi
done

# 📜 Strukturübersicht
echo "✅ Struktur im Backup:" | tee -a "$LOG"
tree "$TARGET/$DATE" | tee -a "$LOG"

# 🗜️ Komprimieren
TAR_NAME="coreflow_custom_${DATE}.tar.gz"
tar -czf "$TARGET/$TAR_NAME" -C "$TARGET/$DATE" ./
echo "📦 Archiv erzeugt: $TARGET/$TAR_NAME" | tee -a "$LOG"

exit 0
