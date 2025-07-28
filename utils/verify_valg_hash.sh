#!/bin/bash

FILE="/opt/coreflow/core/ki/valg_engine_cpu.py"
CHECKSUM_FILE="/opt/coreflow/checksums/valg_engine_cpu.sha256"

echo "🔍 Prüfe SHA256 von $FILE..."
sha256sum -c "$CHECKSUM_FILE"

if [ $? -eq 0 ]; then
    echo "✅ File ist unverändert."
else
    echo "❌ WARNUNG: VALG-Engine wurde verändert!"
    logger -p user.warning "VALG-CPU Hash mismatch detected!"
    # Optional: Telegram-Alert hier einfügen
fi
