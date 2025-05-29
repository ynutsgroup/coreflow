#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import logging
import time
import paho.mqtt.client as mqtt
import subprocess
import socket
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# --- Configuration ---
load_dotenv("/opt/coreflow/.env")

# MQTT Settings
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER")
MQTT_PASS = os.getenv("MQTT_PASS")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "gpu/monitor")
CLIENT_ID = f"ftmo-gpu-{socket.gethostname()}"

# Alert Thresholds
GPU_TEMP_CRITICAL = int(os.getenv("GPU_TEMP_CRITICAL", "90"))
GPU_TEMP_WARNING = int(os.getenv("GPU_TEMP_WARNING", "80"))
GPU_USAGE_LIMIT = int(os.getenv("GPU_USAGE_LIMIT", "95"))

# Telegram Alerts
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# FTMO Compliance
FTMO_ALERT_INTERVAL = 3600  # 1 hour cooldown for non-critical alerts

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("/var/log/ftmo_gpu_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GPUMonitor:
    def __init__(self):
        self.last_alert_time = 0
        self.mqtt_client = self._init_mqtt()
        self.hostname = socket.gethostname()

    def _init_mqtt(self) -> mqtt.Client:
        """Initialize and configure MQTT client"""
        client = mqtt.Client(client_id=CLIENT_ID, clean_session=False)
        
        if MQTT_USER and MQTT_PASS:
            client.username_pw_set(MQTT_USER, MQTT_PASS)
        
        client.on_connect = self._on_mqtt_connect
        client.on_disconnect = self._on_mqtt_disconnect
        return client

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        else:
            logger.error(f"Connection failed with code {rc}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        logger.warning(f"Disconnected from MQTT broker (code: {rc})")
        if rc != 0:
            time.sleep(5)
            self._reconnect_mqtt()

    def _reconnect_mqtt(self):
        """Attempt MQTT reconnection"""
        logger.info("Attempting MQTT reconnection...")
        try:
            self.mqtt_client.reconnect()
        except Exception as e:
            logger.error(f"Reconnection failed: {str(e)}")
            time.sleep(10)
            self._reconnect_mqtt()

    def _get_gpu_metrics(self) -> Optional[Dict]:
        """Collect GPU metrics using nvidia-smi"""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True
            )
            
            usage, temp, mem_used, mem_total = map(float, result.stdout.strip().split(', '))
            
            return {
                "host": self.hostname,
                "gpu_usage": usage,
                "gpu_temp": temp,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "mem_percent": (mem_used / mem_total) * 100,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        except subprocess.TimeoutExpired:
            logger.error("GPU metrics query timed out")
        except Exception as e:
            logger.error(f"Failed to get GPU metrics: {str(e)}")
        return None

    def _send_telegram_alert(self, message: str, priority: str = "normal") -> bool:
        """Send alert with rate limiting"""
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram credentials not configured")
            return False

        current_time = time.time()
        if priority == "normal" and (current_time - self.last_alert_time) < FTMO_ALERT_INTERVAL:
            logger.debug("Alert suppressed due to rate limiting")
            return False

        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            self.last_alert_time = current_time
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {str(e)}")
            return False

    def _check_critical_conditions(self, metrics: Dict) -> bool:
        """Check for critical GPU conditions"""
        critical = False
        alert_msg = f"⚠️ <b>GPU ALERT - {self.hostname}</b> ⚠️\n"

        if metrics['gpu_temp'] >= GPU_TEMP_CRITICAL:
            alert_msg += f"🔥 <b>CRITICAL TEMP</b>: {metrics['gpu_temp']}°C\n"
            critical = True
        elif metrics['gpu_temp'] >= GPU_TEMP_WARNING:
            alert_msg += f"🌡️ High Temp: {metrics['gpu_temp']}°C\n"

        if metrics['gpu_usage'] >= GPU_USAGE_LIMIT:
            alert_msg += f"⚡ High Usage: {metrics['gpu_usage']}%\n"
            critical = True

        if metrics['mem_percent'] >= 90:
            alert_msg += f"💾 High Memory: {metrics['mem_percent']:.1f}%\n"
            critical = True

        if critical or "CRITICAL" in alert_msg:
            alert_msg += f"\n🕒 {metrics['timestamp']}"
            self._send_telegram_alert(alert_msg, "high")
            return True
        elif "High" in alert_msg:
            alert_msg += f"\n🕒 {metrics['timestamp']}"
            self._send_telegram_alert(alert_msg)
        
        return False

    def run(self):
        """Main monitoring loop"""
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.mqtt_client.loop_start()

            while True:
                metrics = self._get_gpu_metrics()
                if metrics:
                    # Publish metrics to MQTT
                    self.mqtt_client.publish(
                        MQTT_TOPIC,
                        payload=json.dumps(metrics),
                        qos=1
                    )
                    
                    # Check for critical conditions
                    self._check_critical_conditions(metrics)
                    
                    logger.info(
                        f"GPU: {metrics['gpu_usage']:.1f}% "
                        f"Temp: {metrics['gpu_temp']}°C "
                        f"Mem: {metrics['mem_percent']:.1f}%"
                    )

                time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
        except Exception as e:
            logger.critical(f"Fatal error: {str(e)}")
        finally:
            self.mqtt_client.disconnect()
            self.mqtt_client.loop_stop()

if __name__ == "__main__":
    monitor = GPUMonitor()
    monitor.run()
