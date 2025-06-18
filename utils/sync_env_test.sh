#!/bin/bash
# /opt/coreflow/utils/sync_env_test.sh

echo "🔎 Prüfe Verfügbarkeit von $WIN_IP ..."
ping -c 1 "$WIN_IP" &> /dev/null || {
  echo "❌ Zielhost nicht erreichbar: $WIN_IP"
  exit 1
}

echo "🔐 Teste SSH-Verbindung ..."
ssh -i ~/.ssh/id_rsa -o BatchMode=yes -o ConnectTimeout=5 "$WIN_USER@$WIN_IP" "echo ✅ SSH Zugriff funktioniert" || {
  echo "❌ SSH Verbindung fehlgeschlagen"
  exit 1
}

echo "📁 Teste Schreibzugriff im Zielpfad ..."
ssh -i ~/.ssh/id_rsa "$WIN_USER@$WIN_IP" "echo test > $WIN_PATH/sync_test.txt && del $WIN_PATH/sync_test.txt" || {
  echo "❌ Schreibtest fehlgeschlagen"
  exit 1
}

echo "✅ Alle Tests erfolgreich bestanden"
