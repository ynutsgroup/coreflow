#!/usr/bin/env python3
import torch
import psutil

def check_hardware():
    stats = {
        "gpu_temp": torch.cuda.temperature(),
        "gpu_util": torch.cuda.utilization(),
        "cpu_load": psutil.cpu_percent(),
        "ram_free": psutil.virtual_memory().available / (1024**3)
    }
    if stats["gpu_temp"] > 80:
        print(f"WARNING: GPU-Temperatur kritisch: {stats['gpu_temp']}°C")
    return stats

if __name__ == "__main__":
    print(check_hardware())

