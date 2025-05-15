#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
from pathlib import Path

# Configure logging
LOG_FILE = "/var/log/ftmo_gpu_monitor.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('FTMO_GPU_MONITOR')

def check_nvidia_smi():
    """Verify nvidia-smi is available and working"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            raise RuntimeError(f"nvidia-smi error: {result.stderr}")
        logger.info(f"Detected GPUs:\n{result.stdout.strip()}")
        return True
    except Exception as e:
        logger.critical(f"GPU check failed: {str(e)}")
        return False

def main():
    try:
        # Verify system requirements
        if not Path("/usr/bin/nvidia-smi").exists():
            raise FileNotFoundError("nvidia-smi not found at /usr/bin/nvidia-smi")
        
        if not check_nvidia_smi():
            sys.exit(1)

        logger.info("Starting FTMO GPU Monitoring Service")
        
        # Main monitoring loop
        while True:
            try:
                # Your monitoring logic here
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                logger.info(f"GPU Stats: {result.stdout.strip()}")
                time.sleep(5)
                
            except subprocess.TimeoutExpired:
                logger.warning("GPU query timeout")
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                time.sleep(10)

    except Exception as e:
        logger.critical(f"Service failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
