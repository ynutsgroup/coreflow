#!/bin/bash
# Überwacht alle CoreFlow-Komponenten

check_process() {
    local name=$1
    local cmd=$2
    if ! pgrep -f "$cmd" >/dev/null; then
        echo "❌ $name ist NICHT aktiv (Befehl: $cmd)"
        return 1
    else
        echo "✅ $name läuft (PID: $(pgrep -f "$cmd"))"
        return 0
    fi
}

# 1. Bridge-Process
check_process "Linux-Bridge" "python institutional_bridge.py" || {
    echo "Starten der Bridge..."
    /opt/coreflow/venv/bin/python /opt/coreflow/institutional_bridge.py &
}

# 2. Redis-Server
check_process "Redis-Server" "redis-server" || {
    echo "Starten von Redis..."
    sudo systemctl start redis-server
}

# 3. ZMQ-Verbindung
if ! nc -z localhost 5555; then
    echo "❌ ZMQ-Port 5555 nicht erreichbar"
    echo "Problemanalyse:"
    sudo lsof -i :5555 || echo "Keine Prozesse gebunden"
else
    echo "✅ ZMQ-Verbindung aktiv"
fi

# 4. Windows-Empfänger prüfen (nur wenn Windows erreichbar)
if ping -c 1 10.10.10.40 &>/dev/null; then
    if ! nc -z 10.10.10.40 5555; then
        echo "⚠️ Windows-Empfänger nicht erreichbar"
    else
        echo "✅ Windows-Empfänger verbunden"
    fi
fi
