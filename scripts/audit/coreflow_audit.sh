#!/bin/bash

ROOT="/opt/coreflow"

echo "🔍 Scanne CoreFlow-Verzeichnis: $ROOT"
echo "----------------------------------------"

find "$ROOT" \( \
  -name "*.py" -o \
  -name "*.cu" -o \
  -name "*.yaml" -o \
  -name "*.yml" -o \
  -name "*.json" -o \
  -name "*.sh" -o \
  -name ".env" -o \
  -name "*.service" -o \
  -name "*.md" \
\) \
  -not -path "*/venv/*" \
  -not -path "*/__pycache__/*" \
  -not -path "*/site-packages/*" \
  -not -path "*/usr/*" \
  -not -path "*/lib/*" \
  -not -path "*/bin/*" \
  -not -path "*.egg-info/*" \
  -not -name "python*" \
  -not -name "*.so" \
  -not -name "*.dll" \
  -not -name "*.whl"

echo "✅ Fertig. Nur eigene Dateien angezeigt."
