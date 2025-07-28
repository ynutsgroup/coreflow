#!/bin/bash
# Entfernt alle 0-Byte-Dateien innerhalb von /opt/coreflow (keine Symlinks oder Devices)

TARGET="/opt/coreflow"

echo "🧹 Starte Bereinigung leerer Dateien in: $TARGET"

find "$TARGET" -type f -size 0 -print -delete
echo "✅ Bereinigung abgeschlossen."
