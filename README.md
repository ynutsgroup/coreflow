# CoreFlow – KI-gestütztes Auto-Trading-System 🧠📈

Ein autonomes Trading-Framework für MetaTrader 5, FTMO und Telegram – mit KI, Watchdog, Verschlüsselung und Backup.

## 🔑 Hauptfunktionen

- 📡 MT5-Verbindung (über Login in .env.enc)
- 🧠 AI Trading Agent (FTMOAgent mit PyTorch)
- 📈 SpreadScanner für Marktdaten
- 🔄 Auto-Restart durch Watchdog
- ⏸️ AutoPause bei Leerlauf
- 📨 Telegram Alerts (Start, Uptime, Fehler)
- 🔐 .env-Verschlüsselung via Fernet
- 💾 GitHub Actions für Backup & Snapshot

## 🧪 Wiederherstellung

```bash
git clone https://github.com/ynutsgroup/coreflow.git
cd coreflow
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 watchdog.py
