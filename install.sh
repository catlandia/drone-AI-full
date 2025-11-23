#!/bin/bash
#
# Drone AI Full - Linux/macOS Installer
#
# Usage:
#   ./install.sh              # Basic installation
#   ./install.sh --all        # Install with all extras
#   ./install.sh --venv       # Create virtual environment
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_banner() {
    echo ""
    echo "============================================================"
    echo "  Drone AI Full - 4-Layer Autonomous Drone System"
    echo "  Installer for Linux/macOS"
    echo "============================================================"
    echo ""
}

print_step() {
    echo -e "${YELLOW}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[+]${NC} $1"
}

print_error() {
    echo -e "${RED}[!]${NC} $1"
}

check_python() {
    print_step "Checking Python installation..."

    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.9+"
        exit 1
    fi

    # Check version
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]); then
        print_error "Python 3.9+ required, found $PYTHON_VERSION"
        exit 1
    fi

    print_success "Python $PYTHON_VERSION OK"
}

create_venv() {
    VENV_PATH="${1:-venv}"
    print_step "Creating virtual environment at $VENV_PATH..."

    $PYTHON_CMD -m venv "$VENV_PATH"

    # Activate it
    source "$VENV_PATH/bin/activate"
    PYTHON_CMD="python"

    # Upgrade pip
    pip install --upgrade pip

    print_success "Virtual environment created and activated"
}

install_package() {
    print_step "Installing drone-ai-full package..."

    EXTRAS=""
    if [ "$INSTALL_ALL" = true ]; then
        EXTRAS="[all]"
    elif [ "$INSTALL_TRAINING" = true ]; then
        EXTRAS="[training]"
    elif [ "$INSTALL_DEV" = true ]; then
        EXTRAS="[dev]"
    fi

    pip install -e "${SCRIPT_DIR}${EXTRAS}"

    print_success "Package installed"
}

install_pytorch() {
    print_step "Installing PyTorch..."

    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "  CUDA detected, installing GPU version..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  No CUDA detected, installing CPU version..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

    print_success "PyTorch installed"
}

verify_installation() {
    print_step "Verifying installation..."

    $PYTHON_CMD -c "
import drone_ai
print(f'  Version: {drone_ai.__version__}')
from drone_ai import DroneAI, MissionPlanner, PathPlanner
print('  All imports successful!')
"

    print_success "Installation verified"
}

print_usage() {
    echo ""
    echo "============================================================"
    echo "  Installation Complete!"
    echo "============================================================"
    echo ""

    if [ -n "$VENV_PATH" ]; then
        echo "To activate the virtual environment:"
        echo "  source $VENV_PATH/bin/activate"
        echo ""
    fi

    echo "Quick Start:"
    echo ""
    echo "  # Run the demo"
    echo "  python -m examples.demo"
    echo ""
    echo "  # Or use the CLI"
    echo "  drone-ai demo --deliveries 3"
    echo ""
}

# Parse arguments
USE_VENV=false
INSTALL_ALL=false
INSTALL_TRAINING=false
INSTALL_DEV=false
VENV_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --venv)
            USE_VENV=true
            VENV_PATH="${2:-venv}"
            shift
            shift
            ;;
        --all)
            INSTALL_ALL=true
            shift
            ;;
        --training)
            INSTALL_TRAINING=true
            shift
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --venv [PATH]  Create virtual environment (default: ./venv)"
            echo "  --all          Install all optional dependencies"
            echo "  --training     Install training dependencies"
            echo "  --dev          Install development dependencies"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Main installation
print_banner

check_python

if [ "$USE_VENV" = true ]; then
    create_venv "$VENV_PATH"
fi

if [ "$INSTALL_ALL" = true ] || [ "$INSTALL_TRAINING" = true ]; then
    install_pytorch
fi

install_package

verify_installation

print_usage
