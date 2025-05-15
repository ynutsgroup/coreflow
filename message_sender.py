#!/usr/bin/env python3
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/coreflow/logs/messages.log'),
        logging.StreamHandler()
    ]
)

while True:
    logging.info("Service aktiv")
    time.sleep(60)
