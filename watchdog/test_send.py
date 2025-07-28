import sys
sys.path.append("/opt/coreflow")

from utils.signal_emitter import send_signal

send_signal("EURUSD", "SELL", 0.3)
