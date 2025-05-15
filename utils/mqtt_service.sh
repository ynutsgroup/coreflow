#!/bin/bash
case "$1" in
  start) sudo systemctl start coreflow_mqtt.service ;;
  stop) sudo systemctl stop coreflow_mqtt.service ;;
  restart) sudo systemctl restart coreflow_mqtt.service ;;
  status) sudo systemctl status coreflow_mqtt.service ;;
  log) tail -f /opt/coreflow/logs/mqtt_subscriber.log ;;
  *) echo "Usage: $0 {start|stop|restart|status|log}" ;;
esac
