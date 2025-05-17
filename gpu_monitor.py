#!/usr/bin/env python3
import subprocess
import json
import time
import paho.mqtt.client as mqtt
import os
import signal
from dotenv import load_dotenv
from datetime import datetime

# 🔧 Load Environment Variables
load_dotenv("/opt/coreflow/.env")

# Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER", "ftmo_user")
MQTT_PASS = os.getenv("MQTT_PASS", "tchaKa6610")
MQTT_TOPIC_STATUS = "gpu/status"
MQTT_TOPIC_CONTROL = "gpu/control"
MQTT_QOS = 1
INTERVAL = 5  # seconds between updates

class GPUMonitor:
    def __init__(self):
        self.running = True
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
        self.client = mqtt.Client(protocol=mqtt.MQTTv5)
        self.client.username_pw_set(MQTT_USER, MQTT_PASS)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _handle_signal(self, signum, frame):
        print(f"🛑 Signal {signum} empfangen, beende Programm...")
        self.running = False

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"✅ Erfolgreich verbunden mit MQTT Broker {MQTT_BROKER}:{MQTT_PORT}")
            client.subscribe(MQTT_TOPIC_CONTROL, qos=MQTT_QOS)
        else:
            print(f"❌ Verbindung fehlgeschlagen mit Code {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode().strip().lower()
            print(f"📩 Kontroll-Nachricht empfangen: {payload}")
            
            if payload == "stop":
                print("🛑 NOT-AUS Befehl empfangen!")
                self.running = False
            elif payload == "restart":
                print("🔁 Neustart-Befehl empfangen")
                # Add restart logic here if needed
                
        except Exception as e:
            print(f"⚠️ Fehler bei Nachrichtenverarbeitung: {e}")

    def _on_disconnect(self, client, userdata, rc):
        print(f"🔌 Verbindung getrennt (Code: {rc})")

    def get_gpu_stats(self):
        """Collect comprehensive GPU statistics"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,clocks.current.graphics", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            
            # Parse output: usage%, temp°C, used_mem_MB, total_mem_MB, power_W, clock_MHz
            stats = [x.strip() for x in result.stdout.strip().split(', ')]
            
            return {
                "gpu_usage": int(stats[0]),
                "gpu_temp": int(stats[1]),
                "memory_used": int(stats[2]),
                "memory_total": int(stats[3]),
                "memory_usage": round((int(stats[2]) / int(stats[3])) * 100, 2),
                "power_draw": float(stats[4]),
                "clock_speed": int(stats[5]),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "hostname": os.uname().nodename
            }
        except subprocess.CalledProcessError as e:
            print(f"⚠️ nvidia-smi Fehler: {e}")
        except Exception as e:
            print(f"⚠️ Unerwarteter GPU-Abfragefehler: {e}")
        return None

    def run(self):
        try:
            print(f"🚀 Starte GPU Monitor auf {os.uname().nodename}")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()

            while self.running:
                stats = self.get_gpu_stats()
                if stats:
                    self.client.publish(
                        MQTT_TOPIC_STATUS,
                        payload=json.dumps(stats),
                        qos=MQTT_QOS,
                        retain=True
                    )
                    print(f"📊 Status gesendet: {stats['gpu_usage']}% | {stats['gpu_temp']}°C | {stats['memory_used']}/{stats['memory_total']}MB")
                time.sleep(INTERVAL)

        except Exception as e:
            print(f"❌ Kritischer Fehler: {e}")
        finally:
            print("🔌 Bereinige Ressourcen...")
            self.client.loop_stop()
            self.client.disconnect()
            print("👋 Programm beendet")

if __name__ == "__main__":
    monitor = GPUMonitor()
    monitor.run()
