#!/bin/bash
set -euo pipefail

# Only run in remote/cloud environments (Open Claw)
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "=== Installing Node.js dependencies ==="
cd "$CLAUDE_PROJECT_DIR/dashboard"
npm install

echo "=== Installing Python dependencies ==="
pip install --quiet --prefer-binary yfinance numpy pandas scipy matplotlib seaborn scikit-learn

echo "=== Setup complete ==="
