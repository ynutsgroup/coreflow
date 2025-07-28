#!/bin/bash

echo "🚨 UFW wird bereinigt und restriktiv neu konfiguriert..."

# Reset – alle Regeln entfernen
ufw --force reset

# Default Regeln
ufw default deny incoming
ufw default allow outgoing

# CoreFlow interne Verbindungen
ufw allow from 10.10.10.40 to any port 6380 proto tcp  # MT5-ZMQ Windows
ufw allow from 10.10.10.50 to any port 6380 proto tcp  # Redis Publisher
ufw allow from 10.10.10.50 to any port 5555 proto tcp  # ZMQ Push
ufw allow from 10.10.10.40 to any port 5555 proto tcp  # Optional Rückkanal

# SSH Zugang (falls gewünscht)
ufw allow OpenSSH

# Aktivieren
ufw --force enable

echo "✅ CoreFlow Firewall-Konfiguration abgeschlossen."
ufw status numbered
