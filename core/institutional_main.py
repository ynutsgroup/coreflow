#!/usr/bin/env python3
# CoreFlow Institutional v4.4 - Cross-Platform Trading Bridge

import asyncio
import logging
import zmq
import torch
import json
from datetime import datetime
from typing import Dict, Optional

# === GPU Initialisierung ===
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(f"✅ GPU aktiv: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU-Modus'}")

# === Konfiguration ===
class Config:
    ZMQ_PORT = 5555                         # Linux-Seite
    WINDOWS_IP = "10.10.10.40"              # Windows MT5 Client IP
    WINDOWS_PORT = 5555                     # Windows MT5 Port
    MAX_RETRIES = 3
    HEARTBEAT_INTERVAL = 30

# === ZeroMQ Cross-Platform Bridge ===
class CrossPlatformBridge:
    def __init__(self):
        self.context = zmq.Context()
        
        # Linux-Seite (Subscriber)
        self.receiver = self.context.socket(zmq.SUB)
        self.receiver.bind(f"tcp://*:{Config.ZMQ_PORT}")
        self.receiver.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Windows-Seite (Publisher)
        self.sender = self.context.socket(zmq.PUB)
        self.sender.connect(f"tcp://{Config.WINDOWS_IP}:{Config.WINDOWS_PORT}")
        
        # GPU-Puffer für Signalverarbeitung
        self.signal_buffer = torch.zeros(10, device=device)

    async def forward_to_mt5(self):
        """Verarbeitet Signale und sendet an Windows MT5"""
        while True:
            try:
                raw_msg = await self.receiver.recv_string()
                signal = self._process_signal(raw_msg)
                
                # GPU-beschleunigte Signalvalidierung
                with torch.no_grad():
                    self.signal_buffer[0] = torch.tensor(signal['confidence'], device=device)
                    if torch.sigmoid(self.signal_buffer[0]).item() < 0.65:
                        logging.warning(f"Signal verworfen (Confidence zu niedrig): {signal}")
                        continue
                
                # An Windows MT5 senden
                self.sender.send_json({
                    **signal,
                    "timestamp": datetime.utcnow().isoformat(),
                    "processed_on": "linux_gpu"
                })
                
            except Exception as e:
                logging.error(f"Signalverarbeitungsfehler: {str(e)}")

    def _process_signal(self, raw: str) -> Dict:
        """Signalvorverarbeitung mit Typkonvertierung"""
        signal = json.loads(raw)
        return {
            "symbol": str(signal['symbol']),
            "action": str(signal['action']).upper(),
            "price": float(signal['price']),
            "confidence": float(signal.get('confidence', 0.75))
        }

# === Hauptloop ===
async def main():
    bridge = CrossPlatformBridge()
    logging.info("🚀 Linux-Windows Trading Bridge aktiv")
    
    try:
        # Starte Signalweiterleitung
        forward_task = asyncio.create_task(bridge.forward_to_mt5())
        
        # System-Herzschlag
        async def heartbeat():
            while True:
                bridge.sender.send_json({"type": "heartbeat", "time": datetime.utcnow().isoformat()})
                await asyncio.sleep(Config.HEARTBEAT_INTERVAL)
        
        heartbeat_task = asyncio.create_task(heartbeat())
        
        await asyncio.gather(forward_task, heartbeat_task)
        
    except KeyboardInterrupt:
        logging.info("🛑 Manuelle Beendigung durch Nutzer")
    finally:
        bridge.receiver.close()
        bridge.sender.close()
        bridge.context.term()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler("/var/log/coreflow/cross_platform_bridge.log"),
            logging.StreamHandler()
        ]
    )
    
    asyncio.run(main())
