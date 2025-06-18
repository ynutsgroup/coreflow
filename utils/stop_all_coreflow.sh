#!/bin/bash
echo "🛑 CoreFlow Total Shutdown gestartet: $(date)"

# 1. Stoppe systemd-Dienste
echo "⛔️ Stoppe systemd-Dienste..."
sudo systemctl stop redis-server
sudo systemctl stop coreflow-watchdog.service
sudo systemctl stop coreflow-bridge.service
sudo systemctl stop coreflow-emitter.service
sudo systemctl stop coreflow-watchdog-full.service
sudo systemctl stop coreflow_watchdog_async.service
sudo systemctl stop watchdog_full_monitor.service
sudo systemctl stop watchdog_full_subprocess.service

# 2. Deaktiviere systemd-Dienste
echo "📛 Deaktiviere systemd Autostarts..."
sudo systemctl disable redis-server
sudo systemctl disable coreflow-watchdog.service
sudo systemctl disable coreflow-bridge.service
sudo systemctl disable coreflow-emitter.service
sudo systemctl disable coreflow-watchdog-full.service
sudo systemctl disable coreflow_watchdog_async.service
sudo systemctl disable watchdog_full_monitor.service
sudo systemctl disable watchdog_full_subprocess.service

# 3. Kill alle Python- und nohup-Prozesse im CoreFlow
echo "🔪 Beende CoreFlow Python-Prozesse..."
pkill -f watchdog
pkill -f coreflow
pkill -f signal_emitter
pkill -f signal_receiver
pkill -f debug_plain
pkill -f mqtt
pkill -f zmq
pkill -f telegram
pkill -f ftmo_ai_trader
pkill -f nohup

# 4. Redis-Keys löschen (optional)
echo "🧹 Bereinige Redis-Keys..."
redis-cli -a 'NeuesSicheresPasswort#2024!' flushall

# 5. nohup.out löschen
echo "🧹 Lösche nohup.out Files..."
find /opt/coreflow -name "nohup.out" -delete

# 6. Netzwerkschnittstelle deaktivieren (optional)
# echo "🌐 Deaktiviere Netzwerk (eth0)..."
# ip link set eth0 down

echo "✅ CoreFlow Infrastruktur vollständig gestoppt."
