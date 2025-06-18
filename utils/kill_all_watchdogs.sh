#!/bin/bash
echo "$(date) - ⛔️ Starte Watchdog-Killer"

# 1. Stoppe bekannte systemd-Dienste
sudo systemctl stop coreflow-watchdog.service 2>/dev/null
sudo systemctl disable coreflow-watchdog.service 2>/dev/null
sudo systemctl stop redis-server 2>/dev/null

# 2. Beende Prozesse hart
pkill -f watchdog_async_nohup.py
pkill -f watchdog_full_monitor.py
pkill -f coreflow_watchdog.py
pkill -f watchdog
sleep 1

# 3. Lösche nohup-Ausgabe
rm -f /opt/coreflow/watchdogs/nohup.out

# 4. Blockiere weitere Starts durch chmod/remove
chmod -x /opt/coreflow/watchdogs/watchdog_async_nohup.py 2>/dev/null
chmod -x /opt/coreflow/utils/watchdog_full_monitor.py 2>/dev/null

mv /opt/coreflow/watchdogs/watchdog_async_nohup.py /opt/coreflow/watchdogs/watchdog_async_nohup.DISABLED 2>/dev/null
mv /opt/coreflow/utils/watchdog_full_monitor.py /opt/coreflow/utils/watchdog_full_monitor.DISABLED 2>/dev/null

# 5. Prüfe, ob noch Reste laufen
echo " "
echo "🔍 Laufende Prozesse nach Kill:"
ps aux | grep -i watchdog | grep -v grep

echo "$(date) - ✅ Alle CoreFlow Watchdogs gestoppt"
