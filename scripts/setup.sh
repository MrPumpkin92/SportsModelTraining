#!/bin/bash
# scripts/setup.sh — project environment setup and prerequisite checks
set -e

# ── Python virtual environment ──────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "→ Creating virtual environment..."
    python -m venv .venv
fi

if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "→ Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Python dependencies installed."

# ── Java check (required by nbainjuries → tabula-py) ───────────────────────
echo "→ Checking Java for nbainjuries (tabula-py requirement)..."
if ! command -v java &> /dev/null; then
    echo "WARNING: Java not found. nbainjuries scraper will be skipped at runtime."
    echo "   Install Java 8+ and add to PATH to enable real-time injury scraping."
    echo "   See: https://www.java.com/en/download/"
    export JAVA_AVAILABLE=false
else
    java_ver=$(java -version 2>&1 | head -1)
    echo "✅ Java found: $java_ver"
    export JAVA_AVAILABLE=true
fi

echo ""
echo "✅ Setup complete. Run './scripts/run.sh' to start the pipeline."
