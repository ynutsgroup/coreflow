#!/bin/bash
# send_test_signal.sh – sendet ein einfaches Testsignal an Redis & ZMQ

SYMBOL=${1:-EURUSD}
ACTION=${2:-BUY}
LOT=${3:-0.1}
CONFIDENCE=${4:-0.9}

/usr/bin/python3 /opt/coreflow/emitter/signal_emitter.py \
  --symbol "$SYMBOL" \
  --action "$ACTION" \
  --lot "$LOT" \
  --confidence "$CONFIDENCE"
