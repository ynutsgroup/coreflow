#!/bin/bash
echo "🔍 Watchdog Autostart-Diagnose startet..."

echo -e "\n📁 Suche nach Watchdog in laufenden Prozessen:"
ps aux | grep -i watchdog | grep -v grep

echo -e "\n🧠 Suche in Crontab (root):"
crontab -l 2>/dev/null | grep -i watchdog

echo -e "\n🧠 Suche in /etc/crontab:"
grep -i watchdog /etc/crontab 2>/dev/null

echo -e "\n🧠 Suche in systemd Autostart:"
find /etc/systemd/system /lib/systemd/system -type f -name "*.service" -exec grep -il watchdog {} \;

echo -e "\n🧠 Suche in .bashrc, .profile, rc.local:"
grep -i watchdog ~/.bashrc ~/.profile /etc/rc.local 2>/dev/null

echo -e "\n🧪 Suche nach nohup-Instanz:"
ps -ef | grep nohup | grep -i watchdog

echo -e "\n📦 Überprüfung abgeschlossen."
