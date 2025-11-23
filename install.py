#!/usr/bin/env python3
"""
Drone AI Full - Cross-Platform Installer

This script installs the Drone AI Full package and its dependencies.
Works on Windows, macOS, and Linux.

Usage:
    python install.py              # Basic installation
    python install.py --all        # Install with all extras
    python install.py --dev        # Install with development tools
    python install.py --training   # Install with training dependencies
"""

import subprocess
import sys
import os
import platform
import argparse
from pathlib import Path


def print_banner():
    """Print installation banner."""
    print()
    print("=" * 60)
    print("  Drone AI Full - 4-Layer Autonomous Drone System")
    print("  Installer")
    print("=" * 60)
    print()


def print_step(step: str):
    """Print a step message."""
    print(f"[*] {step}")


def print_success(msg: str):
    """Print success message."""
    print(f"[+] {msg}")


def print_error(msg: str):
    """Print error message."""
    print(f"[!] {msg}")


def run_command(cmd: list, check: bool = True) -> bool:
    """Run a command and return success status."""
    try:
        subprocess.run(cmd, check=check, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {' '.join(cmd)}")
        return False
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print_step("Checking Python version...")

    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}.{sys.version_info[2]}"

    if major < 3 or (major == 3 and minor < 9):
        print_error(f"Python 3.9+ required, found {version_str}")
        print("Please upgrade Python: https://www.python.org/downloads/")
        return False

    print_success(f"Python {version_str} OK")
    return True


def check_pip():
    """Check if pip is available."""
    print_step("Checking pip...")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            check=True,
            capture_output=True
        )
        print_success("pip OK")
        return True
    except subprocess.CalledProcessError:
        print_error("pip not found")
        return False


def upgrade_pip():
    """Upgrade pip to latest version."""
    print_step("Upgrading pip...")
    return run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])


def create_virtual_env(venv_path: Path) -> bool:
    """Create a virtual environment."""
    print_step(f"Creating virtual environment at {venv_path}...")

    try:
        import venv
        venv.create(venv_path, with_pip=True)
        print_success("Virtual environment created")
        return True
    except Exception as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def get_venv_python(venv_path: Path) -> str:
    """Get the Python executable path in the virtual environment."""
    if platform.system() == "Windows":
        return str(venv_path / "Scripts" / "python.exe")
    else:
        return str(venv_path / "bin" / "python")


def install_package(python_exe: str, extras: list = None):
    """Install the drone-ai-full package."""
    print_step("Installing drone-ai-full package...")

    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()

    # Build install command
    if extras:
        extras_str = ",".join(extras)
        package_spec = f"{script_dir}[{extras_str}]"
    else:
        package_spec = str(script_dir)

    cmd = [python_exe, "-m", "pip", "install", "-e", package_spec]

    if run_command(cmd):
        print_success("Package installed successfully")
        return True
    else:
        print_error("Package installation failed")
        return False


def install_pytorch(python_exe: str):
    """Install PyTorch (required for RL training)."""
    print_step("Installing PyTorch...")

    # Detect if CUDA is available
    cuda_available = False
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=True
        )
        cuda_available = True
        print("  CUDA detected, installing GPU version...")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  No CUDA detected, installing CPU version...")

    if cuda_available:
        # Install with CUDA support
        cmd = [
            python_exe, "-m", "pip", "install",
            "torch", "torchvision", "--index-url",
            "https://download.pytorch.org/whl/cu118"
        ]
    else:
        # Install CPU-only version
        cmd = [
            python_exe, "-m", "pip", "install",
            "torch", "torchvision", "--index-url",
            "https://download.pytorch.org/whl/cpu"
        ]

    return run_command(cmd)


def verify_installation(python_exe: str) -> bool:
    """Verify that the installation works."""
    print_step("Verifying installation...")

    test_code = """
import drone_ai
print(f"  Version: {drone_ai.__version__}")
from drone_ai import DroneAI, MissionPlanner, PathPlanner, PerceptionAI
from drone_ai.layers.flycontrol import DroneEnv, PPOAgent
print("  All imports successful!")
"""

    try:
        subprocess.run(
            [python_exe, "-c", test_code],
            check=True
        )
        print_success("Installation verified")
        return True
    except subprocess.CalledProcessError:
        print_error("Verification failed - some imports may not work")
        return False


def print_usage_instructions(venv_path: Path = None):
    """Print post-installation usage instructions."""
    print()
    print("=" * 60)
    print("  Installation Complete!")
    print("=" * 60)
    print()

    if venv_path:
        if platform.system() == "Windows":
            activate_cmd = f"{venv_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"

        print("To activate the virtual environment:")
        print(f"  {activate_cmd}")
        print()

    print("Quick Start:")
    print()
    print("  # Run the demo")
    print("  python -m examples.demo")
    print()
    print("  # Or use the CLI")
    print("  drone-ai demo --deliveries 3")
    print()
    print("  # In Python")
    print("  from drone_ai import DroneAI")
    print("  drone = DroneAI()")
    print("  drone.add_delivery([100, 50, 0])")
    print()
    print("Documentation: README.md")
    print()


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(
        description="Install Drone AI Full package"
    )
    parser.add_argument(
        "--venv", type=str, default=None,
        help="Create and use a virtual environment at this path"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Install all optional dependencies"
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Install development dependencies"
    )
    parser.add_argument(
        "--training", action="store_true",
        help="Install training dependencies (PyTorch, etc.)"
    )
    parser.add_argument(
        "--visualization", action="store_true",
        help="Install visualization dependencies"
    )
    parser.add_argument(
        "--no-pytorch", action="store_true",
        help="Skip PyTorch installation"
    )

    args = parser.parse_args()

    print_banner()

    # System info
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print()

    # Check prerequisites
    if not check_python_version():
        return 1

    if not check_pip():
        return 1

    # Determine Python executable to use
    python_exe = sys.executable
    venv_path = None

    # Create virtual environment if requested
    if args.venv:
        venv_path = Path(args.venv).absolute()
        if not create_virtual_env(venv_path):
            return 1
        python_exe = get_venv_python(venv_path)

        # Upgrade pip in venv
        upgrade_pip_cmd = [python_exe, "-m", "pip", "install", "--upgrade", "pip"]
        run_command(upgrade_pip_cmd, check=False)

    # Determine extras to install
    extras = []
    if args.all:
        extras = ["all"]
    else:
        if args.dev:
            extras.append("dev")
        if args.training:
            extras.append("training")
        if args.visualization:
            extras.append("visualization")

    # Install PyTorch first if training is requested
    if (args.training or args.all) and not args.no_pytorch:
        if not install_pytorch(python_exe):
            print_error("PyTorch installation failed, continuing anyway...")

    # Install the package
    if not install_package(python_exe, extras if extras else None):
        return 1

    # Verify installation
    verify_installation(python_exe)

    # Print usage instructions
    print_usage_instructions(venv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
