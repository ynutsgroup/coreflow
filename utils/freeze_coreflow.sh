#!/bin/bash
echo "🧊 CoreFlow wird eingefroren..."

# Prozesse beenden
pkill -f emitter
pkill -f signal
pkill -f test_signal
pkill -f trade
pkill -f coreflow
pkill -f zmq
pkill -f mqtt

# Redis stoppen
systemctl stop redis-server

# Dateien blockieren
chmod -x /opt/coreflow/**/*.py 2>/dev/null
chmod -x /opt/coreflow/core/**/*.py 2>/dev/null
chmod -x /opt/coreflow/emitter/*.py 2>/dev/null
chmod -x /opt/coreflow/utils/*.py 2>/dev/null

# Umbenennen kritischer Scripte
mv /opt/coreflow/emitter/signal_emitter.py /opt/coreflow/emitter/signal_emitter.DISABLED 2>/dev/null
mv /opt/coreflow/test_signal_plain.py /opt/test_signal_plain.DISABLED 2>/dev/null
mv /opt/coreflow/signals /opt/coreflow/signals_DISABLED 2>/dev/null

# Log löschen
rm -rf /opt/coreflow/logs/*

echo "✅ CoreFlow eingefroren & gestoppt."
