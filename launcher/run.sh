#!/bin/bash
# LLM-Sim Launcher — start script
# Ensures cwd is the project root so config paths resolve correctly.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1
echo "Starting LLM-Sim Launcher from: $PROJECT_ROOT"
streamlit run launcher/app.py "$@"
