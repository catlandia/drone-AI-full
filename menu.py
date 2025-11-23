#!/usr/bin/env python3
"""
Drone AI Full - Interactive Menu

Cross-platform menu system for managing the Drone AI system.
Works on Windows, macOS, and Linux.

Usage:
    python menu.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def print_banner():
    """Print the main banner."""
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                                                            ║")
    print("║           DRONE AI FULL - 4-LAYER SYSTEM                   ║")
    print("║                                                            ║")
    print("║   Layer 1: Manager      - Mission Planning                 ║")
    print("║   Layer 2: PathFinder   - Path Planning                    ║")
    print("║   Layer 3: Perception   - Obstacle Detection               ║")
    print("║   Layer 4: FlyControl   - Flight Control                   ║")
    print("║                                                            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()


def print_menu():
    """Print the main menu options."""
    print("┌────────────────────────────────────────────────────────────┐")
    print("│                      MAIN MENU                             │")
    print("├────────────────────────────────────────────────────────────┤")
    print("│                                                            │")
    print("│   [1]  Install / Update Package                            │")
    print("│   [2]  Run Demo                                            │")
    print("│   [3]  Run Demo (Verbose)                                  │")
    print("│   [4]  Train RL Agent                                      │")
    print("│   [5]  Test Single Layer                                   │")
    print("│   [6]  System Information                                  │")
    print("│   [7]  Open Documentation                                  │")
    print("│   [8]  Python Interactive Shell                            │")
    print("│                                                            │")
    print("│   [0]  Exit                                                │")
    print("│                                                            │")
    print("└────────────────────────────────────────────────────────────┘")
    print()


def print_training_menu():
    """Print the training submenu."""
    print("┌────────────────────────────────────────────────────────────┐")
    print("│                   TRAINING MENU                            │")
    print("├────────────────────────────────────────────────────────────┤")
    print("│                                                            │")
    print("│   [1]  Train Hover Task         (Basic stabilization)      │")
    print("│   [2]  Train Waypoint Task      (Navigation)               │")
    print("│   [3]  Train Delivery Task      (Single delivery)          │")
    print("│   [4]  Train Delivery Route     (Multi-delivery)           │")
    print("│                                                            │")
    print("│   [0]  Back to Main Menu                                   │")
    print("│                                                            │")
    print("└────────────────────────────────────────────────────────────┘")
    print()


def print_layer_menu():
    """Print the layer testing submenu."""
    print("┌────────────────────────────────────────────────────────────┐")
    print("│                   TEST SINGLE LAYER                        │")
    print("├────────────────────────────────────────────────────────────┤")
    print("│                                                            │")
    print("│   [1]  Test Manager Layer       (Mission planning)         │")
    print("│   [2]  Test PathFinder Layer    (A*/RRT planning)          │")
    print("│   [3]  Test Perception Layer    (Obstacle detection)       │")
    print("│   [4]  Test FlyControl Layer    (Physics simulation)       │")
    print("│                                                            │")
    print("│   [0]  Back to Main Menu                                   │")
    print("│                                                            │")
    print("└────────────────────────────────────────────────────────────┘")
    print()


def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.absolute()


def run_command(cmd: list, wait: bool = True):
    """Run a command."""
    try:
        if wait:
            subprocess.run(cmd)
        else:
            subprocess.Popen(cmd)
    except FileNotFoundError:
        print(f"Error: Command not found: {cmd[0]}")
    except KeyboardInterrupt:
        print("\nInterrupted.")


def check_installed() -> bool:
    """Check if drone_ai is installed."""
    try:
        import drone_ai
        return True
    except ImportError:
        return False


def install_package():
    """Run the installer."""
    clear_screen()
    print_banner()
    print("Installing Drone AI Full...\n")

    script_dir = get_script_dir()

    if platform.system() == 'Windows':
        installer = script_dir / 'install.bat'
        if installer.exists():
            run_command(['cmd', '/c', str(installer)])
        else:
            run_command([sys.executable, str(script_dir / 'install.py')])
    else:
        installer = script_dir / 'install.sh'
        if installer.exists():
            run_command(['bash', str(installer)])
        else:
            run_command([sys.executable, str(script_dir / 'install.py')])

    input("\nPress Enter to continue...")


def run_demo(verbose: bool = False):
    """Run the demo."""
    clear_screen()
    print_banner()

    if not check_installed():
        print("Error: drone_ai is not installed. Please install first (option 1).")
        input("\nPress Enter to continue...")
        return

    print("Running Demo...\n")
    print("=" * 60)

    script_dir = get_script_dir()
    demo_path = script_dir / 'examples' / 'demo.py'

    if demo_path.exists():
        run_command([sys.executable, str(demo_path)])
    else:
        # Use CLI
        args = ['--deliveries', '3']
        if verbose:
            args.append('--verbose')
        run_command([sys.executable, '-m', 'drone_ai.cli', 'demo'] + args)

    input("\nPress Enter to continue...")


def training_menu():
    """Show training submenu."""
    tasks = {
        '1': ('hover', 50000),
        '2': ('waypoint', 100000),
        '3': ('delivery', 200000),
        '4': ('delivery_route', 500000),
    }

    while True:
        clear_screen()
        print_banner()
        print_training_menu()

        choice = input("Select option: ").strip()

        if choice == '0':
            break
        elif choice in tasks:
            task, default_steps = tasks[choice]

            print(f"\nTraining {task} task")
            steps_input = input(f"Number of steps [{default_steps}]: ").strip()
            steps = int(steps_input) if steps_input else default_steps

            clear_screen()
            print_banner()
            print(f"Training {task} for {steps} steps...\n")
            print("Press Ctrl+C to stop training early.\n")
            print("=" * 60)

            run_command([
                sys.executable, '-m', 'drone_ai.cli', 'train',
                '--task', task, '--steps', str(steps)
            ])

            input("\nPress Enter to continue...")


def layer_test_menu():
    """Show layer testing submenu."""
    while True:
        clear_screen()
        print_banner()
        print_layer_menu()

        choice = input("Select option: ").strip()

        if choice == '0':
            break
        elif choice == '1':
            test_manager()
        elif choice == '2':
            test_pathfinder()
        elif choice == '3':
            test_perception()
        elif choice == '4':
            test_flycontrol()


def test_manager():
    """Test the Manager layer."""
    clear_screen()
    print_banner()
    print("Testing Manager Layer (Mission Planning)\n")
    print("=" * 60)

    code = '''
from drone_ai import MissionPlanner, DeliveryRequest
import numpy as np

# Create planner with base at origin
planner = MissionPlanner(base_position=[0, 0, 0])
print("Created MissionPlanner at base [0, 0, 0]")

# Add some deliveries
print("\\nAdding deliveries...")
d1 = planner.create_delivery([100, 50, 0], priority=3, weight=1.5)
print(f"  Delivery {d1.id}: target=[100, 50, 0], priority=3 (critical)")

d2 = planner.create_delivery([50, 100, 0], priority=2, weight=1.0)
print(f"  Delivery {d2.id}: target=[50, 100, 0], priority=2 (urgent)")

d3 = planner.create_delivery([75, 75, 0], priority=1, weight=0.5)
print(f"  Delivery {d3.id}: target=[75, 75, 0], priority=1 (normal)")

# Select next delivery (should be highest priority)
print("\\nSelecting next delivery...")
next_delivery = planner.select_next_delivery()
print(f"  Selected: Delivery {next_delivery.id} (priority {next_delivery.priority})")

# Check range calculations
print("\\nRange calculations:")
print(f"  Max range (empty): {planner.get_max_range(0):.0f} m")
print(f"  Max range (2kg):   {planner.get_max_range(2):.0f} m")
print(f"  Safe range (1kg):  {planner.get_safe_range(1):.0f} m")

# Get mission summary
print("\\nMission Summary:")
summary = planner.get_mission_summary()
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\\n[OK] Manager layer working correctly!")
'''
    run_command([sys.executable, '-c', code])
    input("\nPress Enter to continue...")


def test_pathfinder():
    """Test the PathFinder layer."""
    clear_screen()
    print_banner()
    print("Testing PathFinder Layer (Path Planning)\n")
    print("=" * 60)

    code = '''
from drone_ai import PathPlanner
from drone_ai.core.state import Obstacle, ObstacleType
import numpy as np

# Create planner
planner = PathPlanner(bounds=(-50, 150, -50, 150, 0, 30))
print("Created PathPlanner with bounds [-50, 150] x [-50, 150] x [0, 30]")

# Add obstacles
print("\\nAdding obstacles...")
obs1 = Obstacle([50, 50, 0], [5, 15], ObstacleType.BUILDING)
obs2 = Obstacle([80, 30, 0], [3, 10], ObstacleType.TREE)
planner.update_obstacles([obs1, obs2])
print(f"  Building at [50, 50] - radius=5, height=15")
print(f"  Tree at [80, 30] - radius=3, height=10")

# Plan path with A*
print("\\nPlanning path with A*...")
start = np.array([0, 0, 5])
goal = np.array([100, 50, 5])
path = planner.plan_path_astar(start, goal)
print(f"  Start: {start}")
print(f"  Goal: {goal}")
print(f"  Path found with {len(path)} waypoints")

# Show waypoints
print("\\nWaypoints:")
for i, wp in enumerate(path):
    print(f"  {i}: [{wp[0]:.1f}, {wp[1]:.1f}, {wp[2]:.1f}]")

# Calculate total path length
total_length = sum(np.linalg.norm(path[i+1] - path[i]) for i in range(len(path)-1))
print(f"\\nTotal path length: {total_length:.1f} m")
print(f"Direct distance: {np.linalg.norm(goal - start):.1f} m")

print("\\n[OK] PathFinder layer working correctly!")
'''
    run_command([sys.executable, '-c', code])
    input("\nPress Enter to continue...")


def test_perception():
    """Test the Perception layer."""
    clear_screen()
    print_banner()
    print("Testing Perception Layer (Obstacle Detection)\n")
    print("=" * 60)

    code = '''
from drone_ai import PerceptionAI
from drone_ai.core.state import Obstacle, ObstacleType
import numpy as np

# Create perception system
perception = PerceptionAI()
print("Created PerceptionAI system")

# Create simulated obstacles
obstacles = [
    Obstacle([30, 20, 0], [3, 10], ObstacleType.TREE),
    Obstacle([50, -10, 0], [5, 15], ObstacleType.BUILDING),
    Obstacle([70, 40, 0], [2, 8], ObstacleType.POLE),
]
print(f"\\nSimulated {len(obstacles)} obstacles in environment")

# Detect from simulation
drone_pos = np.array([0, 0, 10])
print(f"\\nDrone position: {drone_pos}")
print("Running perception...")

result = perception.detect_from_simulation(
    obstacles=obstacles,
    drone_position=drone_pos,
    max_range=100.0
)

print(f"\\nDetected {len(result['obstacles'])} obstacles:")
for obs in result["obstacles"]:
    print(f"  ID {obs['id']}: {obs['type']} at {obs['position'][:2]}")

print(f"\\nNearest obstacle: {result['nearest_obstacle_distance']:.1f} m")
print(f"Ground distance: {result['ground_distance']:.1f} m")

# Convert to PathFinder format
pf_obstacles = perception.get_obstacles_for_pathfinder(result)
print(f"\\nConverted to {len(pf_obstacles)} Obstacle objects for PathFinder")

print("\\n[OK] Perception layer working correctly!")
'''
    run_command([sys.executable, '-c', code])
    input("\nPress Enter to continue...")


def test_flycontrol():
    """Test the FlyControl layer."""
    clear_screen()
    print_banner()
    print("Testing FlyControl Layer (Physics Simulation)\n")
    print("=" * 60)

    code = '''
from drone_ai.layers.flycontrol import DroneSimulation, DroneEnv, TaskType
import numpy as np

# Test simulation
print("Testing DroneSimulation...")
sim = DroneSimulation()
sim.reset(position=[0, 0, 5])
print(f"  Initial position: {sim.state.position}")

# Apply some actions
print("\\nSimulating 100 steps with hover thrust...")
for _ in range(100):
    action = np.array([0.5, 0.5, 0.5, 0.5])  # Hover-ish
    sim.step(action)

print(f"  Final position: {sim.state.position}")
print(f"  Crashed: {sim.is_crashed()}")

# Test environment
print("\\nTesting DroneEnv (Gymnasium)...")
env = DroneEnv(task=TaskType.HOVER, difficulty=0.3)
obs, info = env.reset()
print(f"  Observation shape: {obs.shape}")
print(f"  Action space: {env.action_space}")

# Run a few steps
print("\\nRunning 50 environment steps...")
total_reward = 0
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"  Total reward: {total_reward:.2f}")
print(f"  Final position: {info['position']}")

print("\\n[OK] FlyControl layer working correctly!")
'''
    run_command([sys.executable, '-c', code])
    input("\nPress Enter to continue...")


def show_system_info():
    """Show system information."""
    clear_screen()
    print_banner()
    print("System Information\n")
    print("=" * 60)

    # Python info
    print(f"\nPython: {sys.version}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")

    # Check if installed
    print("\nPackage Status:")
    try:
        import drone_ai
        print(f"  drone-ai-full: v{drone_ai.__version__} (installed)")
    except ImportError:
        print("  drone-ai-full: NOT INSTALLED")

    # Check PyTorch
    try:
        import torch
        cuda_status = "CUDA available" if torch.cuda.is_available() else "CPU only"
        print(f"  PyTorch: v{torch.__version__} ({cuda_status})")
    except ImportError:
        print("  PyTorch: NOT INSTALLED")

    # Check other dependencies
    deps = {
        'numpy': 'numpy',
        'gymnasium': 'gymnasium',
        'opencv-python': 'cv2',
    }
    for dep_name, import_name in deps.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  {dep_name}: v{version}")
        except ImportError:
            print(f"  {dep_name}: NOT INSTALLED")

    print()
    input("Press Enter to continue...")


def open_documentation():
    """Open the documentation."""
    clear_screen()
    print_banner()

    readme = get_script_dir() / 'README.md'

    if readme.exists():
        print("README.md Contents:\n")
        print("=" * 60)
        with open(readme, 'r') as f:
            content = f.read()
            # Show first part
            lines = content.split('\n')[:80]
            print('\n'.join(lines))
            if len(content.split('\n')) > 80:
                print("\n... (truncated, see README.md for full content)")
    else:
        print("README.md not found.")

    print()
    input("Press Enter to continue...")


def python_shell():
    """Open Python interactive shell with drone_ai imported."""
    clear_screen()
    print_banner()
    print("Opening Python shell with drone_ai imported...\n")
    print("Type 'exit()' or press Ctrl+D to return to menu.\n")
    print("=" * 60)

    code = '''
print("Importing drone_ai...")
try:
    from drone_ai import *
    print("Available: DroneAI, MissionPlanner, PathPlanner, PerceptionAI")
    print("           DroneSimulation, DroneEnv, PPOAgent")
    print()
except ImportError as e:
    print(f"Import error: {e}")
    print("Run option 1 to install first.")
'''

    # Start interactive Python
    run_command([sys.executable, '-i', '-c', code])


def main():
    """Main menu loop."""
    while True:
        clear_screen()
        print_banner()
        print_menu()

        choice = input("Select option: ").strip()

        if choice == '0':
            clear_screen()
            print("\nGoodbye!\n")
            break
        elif choice == '1':
            install_package()
        elif choice == '2':
            run_demo(verbose=False)
        elif choice == '3':
            run_demo(verbose=True)
        elif choice == '4':
            training_menu()
        elif choice == '5':
            layer_test_menu()
        elif choice == '6':
            show_system_info()
        elif choice == '7':
            open_documentation()
        elif choice == '8':
            python_shell()
        else:
            print("\nInvalid option. Press Enter to continue...")
            input()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!\n")
        sys.exit(0)
