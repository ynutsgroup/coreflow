#!/bin/bash
echo "⛔ Stoppe signal_emitter.py ..."

# Prozess killen
pkill -f signal_emitter.py

# Datei blockieren
echo "# BLOCKIERT VOM SYSTEM" > /opt/coreflow/emitter/signal_emitter.py
chmod -x /opt/coreflow/emitter/signal_emitter.py

# Log löschen (optional)
rm -f /opt/coreflow/logs/emitter.log

echo "✅ Spam-Emitter deaktiviert"
