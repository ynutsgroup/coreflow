#!/usr/bin/env python3
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("CoreFlow Main Process - Starting")
while True:
    try:
        logger.info("Main process running")
        time.sleep(10)
    except KeyboardInterrupt:
        logger.info("Main process shutting down")
        break
