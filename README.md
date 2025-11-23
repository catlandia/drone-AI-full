# Drone AI Full

Complete 4-Layer Autonomous Drone System combining mission planning, path finding, perception, and flight control.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DRONE AI FULL                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Layer 1   │    │   Layer 2   │    │   Layer 4   │         │
│  │   MANAGER   │───>│ PATHFINDER  │───>│ FLYCONTROL  │         │
│  │             │    │             │    │             │         │
│  │ What to do  │    │ How to get  │    │ How to fly  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                           ▲                                     │
│                           │                                     │
│                     ┌─────────────┐                            │
│                     │   Layer 3   │                            │
│                     │ PERCEPTION  │                            │
│                     │             │                            │
│                     │ What's there│                            │
│                     └─────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Layers

### Layer 1: Manager
High-level mission planning and task scheduling.
- Multi-delivery queue management
- Priority-based scheduling
- Battery management decisions
- Route optimization (TSP-like)

### Layer 2: PathFinder
Path planning with obstacle avoidance.
- A* algorithm for optimal paths
- RRT for complex environments
- Real-time path adjustment
- Path smoothing and interpolation

### Layer 3: Perception
Real-time obstacle detection and tracking.
- CNN-based object detection
- Depth estimation
- Camera-to-world coordinate transformation
- Multi-object tracking with Kalman filters

### Layer 4: FlyControl
Low-level flight control.
- Quadrotor physics simulation
- Gymnasium-compatible RL environment
- PPO agent for learned control
- Mathematical PD controller fallback

## Installation

```bash
# Clone the repository
git clone https://github.com/catlandia/drone-AI-full.git
cd drone-AI-full

# Install in development mode
pip install -e .

# Or install with all extras
pip install -e ".[all]"
```

## Quick Start

```python
from drone_ai import DroneAI

# Create the complete system
drone = DroneAI()

# Add delivery targets
drone.add_delivery([100, 50, 0], priority=2)
drone.add_delivery([50, 100, 0], priority=1)

# Run autonomous mission
while not drone.is_mission_complete():
    state, reward, done, info = drone.step()
    print(f"Position: {state.position}")

# Get mission summary
print(drone.get_mission_summary())
```

## CLI Usage

```bash
# Run demonstration
drone-ai demo --deliveries 5

# Train RL agent
drone-ai train --steps 100000 --task hover

# Show system info
drone-ai info
```

## Project Structure

```
drone-AI-full/
├── src/drone_ai/
│   ├── __init__.py           # Main package
│   ├── cli.py                # Command-line interface
│   ├── core/
│   │   ├── config.py         # System configuration
│   │   └── state.py          # Shared state classes
│   ├── layers/
│   │   ├── manager/          # Layer 1: Mission Planning
│   │   ├── pathfinder/       # Layer 2: Path Planning
│   │   ├── perception/       # Layer 3: Obstacle Detection
│   │   └── flycontrol/       # Layer 4: Flight Control
│   └── integration/
│       └── full_system.py    # Unified DroneAI class
├── examples/
│   └── demo.py               # Full system demo
├── pyproject.toml
└── README.md
```

## Components

### DroneAI (Integration)

The main class that integrates all layers:

```python
from drone_ai import DroneAI, SystemConfig

# Custom configuration
config = SystemConfig()
config.difficulty = 0.7
config.drone_specs.max_payload = 3.0

drone = DroneAI(
    config=config,
    use_perception=True,
    use_rl_control=False
)
```

### MissionPlanner (Layer 1)

```python
from drone_ai import MissionPlanner

planner = MissionPlanner(base_position=[0, 0, 0])
planner.create_delivery([100, 50, 0], priority=2)

delivery = planner.select_next_delivery()
planner.complete_current_delivery(success=True)
```

### PathPlanner (Layer 2)

```python
from drone_ai import PathPlanner
from drone_ai.core.state import Obstacle

planner = PathPlanner()
planner.add_obstacle(Obstacle([50, 25, 0], [3.0]))

path = planner.plan_path(
    start=[0, 0, 5],
    goal=[100, 50, 5],
    algorithm="astar"  # or "rrt"
)
```

### PerceptionAI (Layer 3)

```python
from drone_ai import PerceptionAI
import numpy as np

perception = PerceptionAI()

# With camera input
result = perception.detect(
    frame=np.zeros((480, 640, 3), dtype=np.uint8),
    drone_position=[0, 0, 10],
    drone_orientation=[0, 0, 0]
)

# Or from simulation
result = perception.detect_from_simulation(
    obstacles=known_obstacles,
    drone_position=[0, 0, 10]
)
```

### DroneEnv (Layer 4)

```python
from drone_ai.layers.flycontrol import DroneEnv, TaskType

env = DroneEnv(
    task=TaskType.DELIVERY_ROUTE,
    difficulty=0.5,
    render_mode="human"
)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## Training

Train the PPO agent for flight control:

```python
from drone_ai.layers.flycontrol import DroneEnv, PPOAgent, TaskType

env = DroneEnv(task=TaskType.HOVER)
agent = PPOAgent(obs_dim=31, action_dim=4)

for episode in range(1000):
    obs, _ = env.reset()
    done = False

    while not done:
        action, info = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        agent.store_transition(
            obs, action, reward,
            info['value'], info['log_prob'],
            terminated or truncated
        )

        obs = next_obs
        done = terminated or truncated

    # Update after each episode
    agent.update(obs)

agent.save("checkpoints/trained_agent.pt")
```

## Dependencies

- Python >= 3.9
- NumPy >= 1.21.0
- Gymnasium >= 0.28.0
- PyTorch >= 2.0.0 (optional, for RL)
- OpenCV >= 4.5.0 (optional, for perception)

## License

MIT License

## Contributing

Contributions are welcome! Please see the issues on GitHub for areas that need work.
