#!/bin/bash
echo "📊 CoreFlow Start – Modellprüfung..."
python3 /opt/coreflow/utils/check_model_version.py || exit 1
echo "✅ Modell OK – Starte Hauptsystem"
python3 /opt/coreflow/core/coreflow_main.py
