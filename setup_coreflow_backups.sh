#!/bin/bash

echo "📦 [1/5] Verzeichnisse vorbereiten..."
mkdir -p /opt/coreflow/tools /opt/coreflow/_daily /opt/coreflow/_shutdown /opt/coreflow/logs

echo "📝 [2/5] Schreibe daily_coreflow_sync.sh..."
cat << 'EOF' > /opt/coreflow/tools/daily_coreflow_sync.sh
#!/bin/bash
DATE=$(date +%Y-%m-%d)
DEST="/opt/coreflow/_daily/coreflow_sync_$DATE.zip"
SOURCE="/opt/coreflow"

mkdir -p "$(dirname "$DEST")"
find "$(dirname "$DEST")" -type f -name "coreflow_sync_*.zip" -mtime +7 -delete
zip -r "$DEST" "$SOURCE/src" "$SOURCE/logs" "$SOURCE/.env" "$SOURCE/notes" > /dev/null 2>&1

echo "[Reminder @ $DATE] CoreFlow Snapshot gespeichert: $DEST"
EOF

chmod +x /opt/coreflow/tools/daily_coreflow_sync.sh

echo "⏰ [3/5] Cronjob für 23:00 Uhr eintragen..."
(crontab -l 2>/dev/null; echo "0 23 * * * /opt/coreflow/tools/daily_coreflow_sync.sh") | crontab -

echo "🛑 [4/5] Schreibe backup_on_shutdown.sh..."
cat << 'EOF' > /opt/coreflow/tools/backup_on_shutdown.sh
#!/bin/bash
DATE=$(date +%Y-%m-%d_%H-%M)
DEST="/opt/coreflow/_shutdown/coreflow_shutdown_$DATE.zip"
SOURCE="/opt/coreflow"

mkdir -p "$(dirname "$DEST")"
zip -r "$DEST" "$SOURCE/src" "$SOURCE/logs" "$SOURCE/.env" "$SOURCE/notes" > /dev/null 2>&1

echo "[✔] Shutdown-Backup erstellt: $DEST"
EOF

chmod +x /opt/coreflow/tools/backup_on_shutdown.sh

echo "🧩 [5/5] systemd-Hook erstellen..."
cat << 'EOF' > /etc/systemd/system/coreflow-shutdown.service
[Unit]
Description=CoreFlow Shutdown Snapshot
DefaultDependencies=no
Before=shutdown.target reboot.target halt.target

[Service]
Type=oneshot
ExecStart=/opt/coreflow/tools/backup_on_shutdown.sh
RemainAfterExit=true

[Install]
WantedBy=halt.target reboot.target shutdown.target
EOF

systemctl daemon-reexec
systemctl daemon-reload
systemctl enable coreflow-shutdown.service

echo "✅ CoreFlow Backup-System aktiv."
