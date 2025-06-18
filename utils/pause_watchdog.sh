#!/bin/bash
# CoreFlow - Vollständiges Watchdog-Stopp-Skript
# Stellt sicher, dass keine Watchdog- oder Autorestart-Prozesse mehr laufen

echo "🛑 Watchdog-Stopp wird ausgeführt..."

# 1. Dienste stoppen & deaktivieren
SERVICES=("coreflow-watchdog" "watchdog" "coreflow_watchdog")
for SERVICE in "${SERVICES[@]}"; do
  systemctl stop "$SERVICE.service" 2>/dev/null
  systemctl disable "$SERVICE.service" 2>/dev/null
  rm -f "/etc/systemd/system/${SERVICE}.service"
done
systemctl daemon-reload

# 2. Prozesse killen
pkill -f watchdog.py
pkill -f coreflow_watchdog.py
pkill -f coreflow-watchdog
pkill -f telegram
pkill -f zmq
pkill -f mqtt

# 3. Crontab-Einträge entfernen
crontab -l | grep -v 'watchdog' | grep -v 'coreflow' | crontab -
sudo crontab -l | grep -v 'watchdog' | grep -v 'coreflow' | crontab -

# 4. Tmux / Screen Sessions
tmux kill-server 2>/dev/null
screen -wipe 2>/dev/null

# 5. Watchdog-Dateien deaktivieren
find /opt/coreflow -iname "*watchdog*" -exec chmod -x {} \;
echo "✅ Alle Watchdog-Prozesse wurden gestoppt und deaktiviert."
