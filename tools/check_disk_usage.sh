#!/bin/bash
echo "🔍 Speicherplatzübersicht (df -h):"
df -h | grep '^/dev/'

echo ""
echo "📦 Geräteübersicht (lsblk):"
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT


