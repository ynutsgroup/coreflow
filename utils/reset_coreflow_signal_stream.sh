#!/bin/bash

# Konfiguration
REDIS_HOST="127.0.0.1"
REDIS_PORT="6380"
REDIS_PASS="ncasInitHW!"
REDIS_KEY="coreflow:signals"

# Prüfe Typ des Schlüssels
key_type=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASS" TYPE "$REDIS_KEY" | tr -d '\r')

echo "📦 Aktueller Typ von '$REDIS_KEY': $key_type"

if [[ "$key_type" != "stream" && "$key_type" != "none" ]]; then
  echo "⚠️  '$REDIS_KEY' ist kein Stream. Lösche..."
  redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASS" DEL "$REDIS_KEY"
  echo "🧼 GESÄUBERT: '$REDIS_KEY' gelöscht."
elif [[ "$key_type" == "stream" ]]; then
  echo "✅ Bereits korrekt als Stream angelegt. Kein Eingriff notwendig."
  exit 0
fi

# Anlegen eines neuen Stream mit Testsignal
echo "🛠️  Lege neuen Stream '$REDIS_KEY' an..."
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASS" XADD "$REDIS_KEY" "*" symbol "EURUSD" action "BUY"

echo "🎉 Stream angelegt mit erstem Eintrag!"
