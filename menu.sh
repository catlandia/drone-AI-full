#!/bin/bash
#
# Drone AI Full - Linux/macOS Menu Launcher
# Simply launches the Python menu
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Python not found. Please install Python 3.9+"
    exit 1
fi

# Launch the menu
exec $PYTHON_CMD "$SCRIPT_DIR/menu.py"
