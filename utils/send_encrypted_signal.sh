#!/bin/bash
# Institutional Signal Wrapper
# Usage: ./send_signal.sh <asset_class> <symbol> <action> <size> [strategy_id]

set -eo pipefail

# Load from secure configuration
export $(grep -v '^#' /etc/trading/.env | xargs)

/usr/bin/python3 /opt/coreflow/publisher/signal_publisher.py \
    --asset "$1" \
    --symbol "$2" \
    --action "$3" \
    --size "$4" \
    ${5:+--strategy "$5"}

