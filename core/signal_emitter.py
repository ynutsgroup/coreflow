#!/usr/bin/env python3
import os
import time
import json
import logging
from pathlib import Path

class SignalEmitter:
    def __init__(self):
        self.setup_logging()
        self.signal_dir = Path('/opt/coreflow/signals')
        self.signal_dir.mkdir(exist_ok=True)
        self.signal_count = 0

    def setup_logging(self):
        """Configure logging for Linux"""
        log_dir = Path('/opt/coreflow/logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir/'signal_emitter.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SignalEmitter')

    def create_signal(self, signal_type, data):
        """Generate a new signal file"""
        signal_id = f"{int(time.time())}_{self.signal_count}"
        signal_file = self.signal_dir / f"{signal_id}.json"
        
        signal_data = {
            'id': signal_id,
            'type': signal_type,
            'timestamp': time.time(),
            'data': data
        }
        
        with open(signal_file, 'w') as f:
            json.dump(signal_data, f)
        
        self.signal_count += 1
        self.logger.info(f"Emitted signal: {signal_id}")
        return signal_file

    def run(self):
        """Main emission loop"""
        self.logger.info("Signal Emitter started")
        while True:
            # Example: emit test signal every 10 seconds
            self.create_signal(
                signal_type='heartbeat',
                data={'status': 'ok', 'load': os.getloadavg()}
            )
            time.sleep(10)

if __name__ == "__main__":
    emitter = SignalEmitter()
    try:
        emitter.run()
    except KeyboardInterrupt:
        emitter.logger.info("Signal Emitter stopped")
