#!/bin/bash
# CoreFlow Redis Startscript – Port 6380 mit Passwort

REDIS_PORT=6380
REDIS_PASS="ncasInitHW!"
REDIS_CONF="/opt/coreflow/config/coreflow_redis.conf"

# Redis Config dynamisch erzeugen, falls nicht vorhanden
if [ ! -f "$REDIS_CONF" ]; then
  cat <<EOF > "$REDIS_CONF"
port $REDIS_PORT
requirepass $REDIS_PASS
bind 127.0.0.1
daemonize yes
logfile "/var/log/redis_coreflow.log"
dir "/var/lib/redis/"
EOF
fi

# Starte Redis mit dieser Config
redis-server "$REDIS_CONF"

# Warte 1 Sekunde, dann Testverbindung
sleep 1
redis-cli -h 127.0.0.1 -p $REDIS_PORT -a "$REDIS_PASS" ping
