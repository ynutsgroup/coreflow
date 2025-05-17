#!/bin/bash

# GPU Monitor Control Script
# Version: 1.1
# Author: FTMO
# Description: Control script for GPU monitoring service

# Configuration
SERVICE_NAME="gpu-monitor.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"
LOG_DIR="/opt/coreflow/logs"
LOG_FILE="$LOG_DIR/gpu_monitor.log"
MAX_LOG_SIZE=10485760  # 10MB
MAX_LOG_FILES=5

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Rotate logs if needed
rotate_logs() {
    if [ -f "$LOG_FILE" ] && [ $(stat -c%s "$LOG_FILE") -gt $MAX_LOG_SIZE ]; then
        echo -e "${YELLOW}Rotating log files...${NC}"
        for i in $(seq $MAX_LOG_FILES -1 2); do
            [ -f "$LOG_FILE.$((i-1))" ] && mv "$LOG_FILE.$((i-1))" "$LOG_FILE.$i"
        done
        mv "$LOG_FILE" "$LOG_FILE.1"
        touch "$LOG_FILE"
        chown $USER "$LOG_FILE"
    fi
}

case "$1" in
  start)
    rotate_logs
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}Service is already running${NC}"
    else
        sudo systemctl start "$SERVICE_NAME"
        echo -e "${GREEN}🚀 GPU Monitor Service started successfully${NC}"
    fi
    ;;
  stop)
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}Service is not running${NC}"
    else
        sudo systemctl stop "$SERVICE_NAME"
        echo -e "${RED}🛑 GPU Monitor Service stopped${NC}"
    fi
    ;;
  restart)
    rotate_logs
    sudo systemctl restart "$SERVICE_NAME"
    echo -e "${BLUE}🔄 GPU Monitor Service restarted${NC}"
    ;;
  status)
    systemctl status "$SERVICE_NAME" --no-pager
    ;;
  enable)
    sudo systemctl enable "$SERVICE_NAME"
    echo -e "${GREEN}✅ Service enabled to start on boot${NC}"
    ;;
  disable)
    sudo systemctl disable "$SERVICE_NAME"
    echo -e "${YELLOW}⚠️ Service disabled from starting on boot${NC}"
    ;;
  log)
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${RED}Log file not found: $LOG_FILE${NC}"
        exit 1
    fi
    case "$2" in
      --follow|-f)
        tail -f "$LOG_FILE"
        ;;
      --error|-e)
        grep -i "error\|fail\|warn\|critical" "$LOG_FILE"
        ;;
      --last|-l)
        tail -n ${3:-20} "$LOG_FILE"
        ;;
      *)
        cat "$LOG_FILE"
        ;;
    esac
    ;;
  install)
    if [ -f "$SERVICE_PATH" ]; then
        echo -e "${YELLOW}Service already installed at $SERVICE_PATH${NC}"
    else
        echo "Creating service file..."
        sudo bash -c "cat > $SERVICE_PATH" <<EOF
[Unit]
Description=GPU Monitoring Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/coreflow
ExecStart=/usr/bin/python3 /opt/coreflow/gpu_monitor.py
Restart=on-failure
RestartSec=5s
StandardOutput=file:$LOG_FILE
StandardError=file:$LOG_FILE

[Install]
WantedBy=multi-user.target
EOF
        sudo systemctl daemon-reload
        echo -e "${GREEN}✅ Service installed successfully${NC}"
        echo "You can now start it with: $0 start"
    fi
    ;;
  uninstall)
    if [ ! -f "$SERVICE_PATH" ]; then
        echo -e "${YELLOW}Service not found at $SERVICE_PATH${NC}"
    else
        sudo systemctl stop "$SERVICE_NAME"
        sudo systemctl disable "$SERVICE_NAME"
        sudo rm "$SERVICE_PATH"
        sudo systemctl daemon-reload
        echo -e "${GREEN}✅ Service uninstalled successfully${NC}"
    fi
    ;;
  *)
    echo -e "${BLUE}⚙️  GPU Monitor Control Script${NC}"
    echo -e "Usage: $0 {start|stop|restart|status|enable|disable|log|install|uninstall}"
    echo -e "\nAdditional log options:"
    echo -e "  log -f, --follow      Follow log output"
    echo -e "  log -e, --error       Show only errors"
    echo -e "  log -l, --last [N]    Show last N lines (default: 20)"
    exit 1
    ;;
esac

exit 0
