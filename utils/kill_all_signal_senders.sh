#!/bin/bash
echo "🛑 Beende alle Signal-Sender... $(date)"
SIGNAL_PATTERNS=("test_signal" "signal_emitter" "signal_aggregator" "emitter" "debug_plain_signal" "debug_plain_signal+")

for pattern in "${SIGNAL_PATTERNS[@]}"; do
    pids=$(pgrep -f "$pattern")
    if [[ ! -z "$pids" ]]; then
        echo "🔻 $pattern → stoppe PID(s): $pids"
        kill -9 $pids
    fi
done

echo "🧹 Lösche Redis Queue (LIST & STREAM)..."
/usr/bin/redis-cli -a 'NeuesSicheresPasswort#2024!' del trading_signals >/dev/null 2>&1
/usr/bin/redis-cli -a 'NeuesSicheresPasswort#2024!' xtrim trading_signals MAXLEN 0 >/dev/null 2>&1

echo "✅ Signal-Sender vollständig gestoppt."
