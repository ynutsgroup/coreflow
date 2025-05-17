#!/usr/bin/env python3
import subprocess
import json
import time
import paho.mqtt.client as mqtt
import os

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_USER = "ftmo_user"
MQTT_PASS = "tchaKa6610"
MQTT_TOPIC = "gpu/status"

def get_gpu_stats():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        usage, temp = result.stdout.strip().split(', ')
        return {"gpu_usage": int(usage), "gpu_temp": int(temp), "timestamp": int(time.time())}
    except Exception as e:
        print(f"GPU-Fehler: {e}")
        return None

client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASS)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

while True:
    stats = get_gpu_stats()
    if stats:
        client.publish(MQTT_TOPIC, json.dumps(stats))
    time.sleep(5)
