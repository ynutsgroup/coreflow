#!/bin/bash
# 🔍 ZMQ-Suchskript für CoreFlow – durchsucht alle .py-Dateien nach ZMQ-Nutzung

BASE_DIR="/opt/coreflow"
OUTPUT_LOG="/opt/coreflow/logs/zmq_search_results.log"

echo "🔍 Suche nach ZMQ-Importen und -Nutzung in: $BASE_DIR"
echo "Ergebnisse werden gespeichert in: $OUTPUT_LOG"
echo "=== [Start] $(date) ===" > "$OUTPUT_LOG"

# Durchsuche alle Python-Dateien
find "$BASE_DIR" -type f -name "*.py" | while read -r file; do
    if grep -q -iE 'import zmq|zmq\.Context|socket\.(bind|connect)' "$file"; then
        echo -e "\n📄 Datei: $file" >> "$OUTPUT_LOG"
        grep -iE 'import zmq|zmq\.Context|socket\.(bind|connect)' "$file" >> "$OUTPUT_LOG"
    fi
done

echo "✅ ZMQ-Suche abgeschlossen – siehe: $OUTPUT_LOG"



