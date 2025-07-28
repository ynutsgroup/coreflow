#!/usr/bin/env python3
# /opt/coreflow/core/monitoring.py
import requests
import subprocess
from time import sleep

class LinuxTradingMonitor:
    def __init__(self, mt5_windows_ip="os.getenv('REDIS_HOST')", poll_interval=5):
        self.mt5_api_url = f"http://{mt5_windows_ip}:5000/mt5/stats"  # Flask-API on Windows
        self.poll_interval = poll_interval
        
    def get_gpu_stats(self):
        """Get NVIDIA GPU stats using nvidia-smi (Linux)"""
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", 
                 "--format=csv,noheader,nounits"]
            ).decode().strip()
            gpu_util, mem_used = result.split(", ")
            return {
                "gpu_util": f"{gpu_util}%",
                "vram": f"{mem_used} MB",
                "error": None
            }
        except Exception as e:
            return {"error": str(e)}

    def get_cpu_usage(self):
        """Get Linux CPU usage percentage"""
        try:
            return subprocess.getoutput(
                "top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'"
            ) + "%"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_ram_usage(self):
        """Get Linux RAM usage in MB"""
        try:
            return subprocess.getoutput(
                "free -m | grep Mem | awk '{print $3}'"
            ) + " MB"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_mt5_stats(self):
        """Get MT5 stats from Windows API"""
        try:
            response = requests.get(self.mt5_api_url, timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unavailable"}

    def check_system(self):
        """Check all system resources"""
        return {
            "linux_cpu": self.get_cpu_usage(),
            "linux_ram": self.get_ram_usage(),
            "gpu": self.get_gpu_stats(),
            "mt5": self.get_mt5_stats()
        }

    def report(self, stats=None):
        """Print system status report"""
        stats = stats or self.check_system()
        print("\n" + "="*40)
        print(f"🖥️  Linux CPU: {stats['linux_cpu']}")
        print(f"🐧 RAM Usage: {stats['linux_ram']}")
        
        if "error" in stats['gpu']:
            print(f"🎮 GPU Error: {stats['gpu']['error']}")
        else:
            print(f"🎮 GPU Usage: {stats['gpu']['gpu_util']} (VRAM: {stats['gpu']['vram']})")
        
        print(f"📊 MT5 Status: {stats['mt5'].get('status', 'unknown')}")
        print("="*40 + "\n")

    def continuous_monitoring(self):
        """Run continuous monitoring loop"""
        try:
            while True:
                self.report()
                sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")

if __name__ == "__main__":
    monitor = LinuxTradingMonitor(mt5_windows_ip="os.getenv('REDIS_HOST')")
    
    # For continuous monitoring (Ctrl+C to stop)
    monitor.continuous_monitoring()
    
    # For single report:
    # monitor.report()
