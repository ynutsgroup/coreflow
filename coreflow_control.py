import subprocess
import json
from datetime import datetime
from pathlib import Path

class CoreFlowController:
    """Systemsteuerung für CoreFlow"""
    
    def __init__(self):
        self.service_name = "coreflow-msg"
        self.log_file = "/opt/coreflow/logs/service.log"
        
    def _run_command(self, cmd: str) -> tuple:
        """Führt Shell-Befehle aus"""
        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                check=True
            )
            return (True, result.stdout.strip())
        except subprocess.CalledProcessError as e:
            return (False, e.stderr.strip())

    def get_status(self) -> dict:
        """Gibt Systemstatus zurück"""
        success, output = self._run_command(f"systemctl is-active {self.service_name}")
        
        return {
            "running": success and output == "active",
            "last_activity": self._get_last_activity(),
            "messages_today": self._count_messages_today(),
            "last_error": self._get_last_error()
        }

    def restart(self) -> bool:
        """Startet Service neu"""
        success, _ = self._run_command(f"systemctl restart {self.service_name}")
        return success

    def get_logs(self, last_n: int = 10) -> str:
        """Holt Service-Logs"""
        _, logs = self._run_command(
            f"journalctl -u {self.service_name} -n {last_n} --no-pager"
        )
        return logs

    def _get_last_activity(self) -> str:
        """Ermittelt letzte Aktivität"""
        if not Path(self.log_file).exists():
            return "N/A"
        
        with open(self.log_file, "r") as f:
            lines = f.readlines()
            return lines[-1].split(" - ")[0] if lines else "N/A"

    def _count_messages_today(self) -> int:
        """Zählt heutige Nachrichten"""
        today = datetime.now().strftime("%Y-%m-%d")
        _, output = self._run_command(
            f"grep -c '{today}.*Message sent' {self.log_file} || echo 0"
        )
        return int(output)

    def _get_last_error(self) -> str:
        """Findet letzten Fehler"""
        _, error = self._run_command(
            f"grep 'ERROR' {self.log_file} | tail -n 1 || echo ''"
        )
        return error if error else None
