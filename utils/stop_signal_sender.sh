#!/bin/bash
echo "🔧 Stoppe test_signal_plain.py ..."

# Prozess suchen und beenden
PID=$(ps aux | grep '[t]est_signal_plain.py' | awk '{print $2}')
if [ -n "$PID" ]; then
    kill -9 "$PID"
    echo "✅ Prozess $PID beendet"
else
    echo "ℹ️ Kein laufender Prozess gefunden"
fi

# Datei deaktivieren
if [ -f /opt/test_signal_plain.py ]; then
    mv /opt/test_signal_plain.py /opt/test_signal_plain.DISABLED
    echo "📁 test_signal_plain.py deaktiviert"
fi

# Redis-Key löschen (optional)
if command -v redis-cli >/dev/null 2>&1; then
    redis-cli DEL trading_signals >/dev/null
    echo "🧹 Redis-Channel 'trading_signals' gelöscht"
fi

echo "✅ Signal-Sender vollständig gestoppt"
