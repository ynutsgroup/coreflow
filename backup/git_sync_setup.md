# CoreFlow Git Auto-Sync Setup

**Stand:** 2025-05-29  🌐

## 🎯 Ziel

Automatisches tägliches Git-Syncing für `/opt/coreflow` um 20:00 Uhr mit Logging und Fehlerbehandlung.

---

## 🔧 Git-Sync-Skript

**Pfad:** `/opt/coreflow/utils/git_sync.py`  
**Log-Datei:** `/opt/coreflow/logs/git_sync.log`

Dieses Skript prüft Änderungen im Repository, stasht lokale Änderungen, führt ein `git rebase`, commitet und pusht.

---

## 🛠️ systemd Service

**Pfad:** `/etc/systemd/system/coreflow-git-sync.service`

```ini
[Unit]
Description=CoreFlow Git Auto-Sync Service
After=network.target

[Service]
Type=oneshot
ExecStart=/opt/coreflow/utils/git_sync.py
WorkingDirectory=/opt/coreflow
StandardOutput=journal
StandardError=journal
