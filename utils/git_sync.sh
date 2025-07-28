#!/bin/bash
# ============================================
# CoreFlow Git Sync Script – Auto Commit & Push
# Autor: CoreFlow CFM | Stand: 2025-07-28
# ============================================

REPO_DIR="/opt/coreflow"
BRANCH="main"
MESSAGE="🧠 AutoSync $(date '+%Y-%m-%d %H:%M:%S')"

cd "$REPO_DIR" || exit 1

echo "🔁 Git Pull & Rebase..."
git pull --rebase origin "$BRANCH"

echo "📦 Stage all changes..."
git add .

echo "📝 Commit: $MESSAGE"
git commit -m "$MESSAGE" || echo "⚠️ Keine neuen Änderungen"

echo "🚀 Push to GitHub..."
git push origin "$BRANCH"

echo "✅ Git Sync abgeschlossen."
