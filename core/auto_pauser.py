import time
import logging

logger = logging.getLogger(__name__)

class AutoPauser:
    def __init__(self, idle_threshold_minutes=30):
        self.idle_threshold = idle_threshold_minutes * 60
        self.last_activity = time.time()

    def update_activity(self):
        self.last_activity = time.time()
        logger.debug("Aktivität erkannt – Timer zurückgesetzt.")

    def should_pause(self):
        idle_time = time.time() - self.last_activity
        if idle_time > self.idle_threshold:
            logger.info(f"⏸️ AutoPause aktiviert – Leerlauf seit {idle_time:.0f} Sekunden.")
            return True
        return False
