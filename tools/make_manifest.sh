#!/bin/bash
TARGET="/opt/coreflow/structure_manifest.json"
echo "{" > "$TARGET"
for DIR in src logs .env notes; do
  echo "  \"$DIR/\": [" >> "$TARGET"
  FILES=$(find /opt/coreflow/$DIR -maxdepth 1 -type f -exec basename {} \; | sort)
  COUNT=$(echo "$FILES" | wc -l)
  INDEX=0
  for FILE in $FILES; do
    INDEX=$((INDEX + 1))
    if [ "$INDEX" -lt "$COUNT" ]; then
      echo "    \"$FILE\"," >> "$TARGET"
    else
      echo "    \"$FILE\"" >> "$TARGET"
    fi
  done
  echo "  ]," >> "$TARGET"
done
# Entferne das letzte Komma vor der schließenden geschweiften Klammer
sed -i '$ s/,\s*$//' "$TARGET"
echo "}" >> "$TARGET"
echo "✅ Manifest mit sauberem JSON erstellt: $TARGET"
