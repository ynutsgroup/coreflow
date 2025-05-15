#!/bin/bash
set -e

echo "📦 Starte Migration zu Python 3.10..."

# Installiere Python 3.10 falls nicht vorhanden
if ! command -v python3.10 &> /dev/null; then
    echo "🔧 Installiere Python 3.10..."
    apt update && apt install -y python3.10 python3.10-venv
fi

cd /opt/coreflow || { echo "❌ Fehler: /opt/coreflow nicht gefunden!"; exit 1; }

# Backup bestehendes venv
if [ -d "venv" ]; then
    echo "🗃️  Backup des alten venv als venv_backup..."
    mv venv venv_backup_$(date +%Y%m%d_%H%M%S)
fi

# Neues venv mit Python 3.10 erstellen
echo "🐍 Erstelle neue virtuelle Umgebung mit Python 3.10..."
python3.10 -m venv venv

# Aktivieren und Abhängigkeiten installieren
source venv/bin/activate
echo "📚 Installiere Requirements..."
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "⚠️ Keine requirements.txt gefunden. Manuelle Installation notwendig!"
fi

# Prüfen ob MT5 jetzt korrekt importiert werden kann
echo "🔍 Prüfe MetaTrader5-Installation..."
python3 -c "import MetaTrader5 as mt5; print('✅ MetaTrader5 Import erfolgreich!')" || echo "❌ MT5 Import fehlgeschlagen!"

echo "✅ Migration abgeschlossen! Starte dein Projekt jetzt neu."
