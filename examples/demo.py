"""
Drone AI Full - Complete System Demo

This example demonstrates how all four layers work together:
1. Manager:     Plans which deliveries to do
2. PathFinder:  Computes optimal paths avoiding obstacles
3. Perception:  Detects obstacles in real-time
4. FlyControl:  Executes flight commands

Run: python -m examples.demo
"""

import numpy as np
from drone_ai import DroneAI
from drone_ai.core.state import Obstacle, ObstacleType


def main():
    print("=" * 70)
    print("  Drone AI Full - 4-Layer Autonomous Drone System Demo")
    print("=" * 70)
    print()

    # Initialize the complete system
    print("Initializing Drone AI system...")
    drone = DroneAI(use_perception=True, use_rl_control=False)
    print(f"  Base position: {drone.base_position}")
    print(f"  Layers: Manager, PathFinder, Perception, FlyControl")
    print()

    # Add some obstacles
    print("Adding obstacles to environment...")
    obstacles = [
        Obstacle(
            position=np.array([30.0, 20.0, 0.0]),
            size=np.array([3.0, 10.0]),
            obstacle_type=ObstacleType.TREE
        ),
        Obstacle(
            position=np.array([50.0, -10.0, 0.0]),
            size=np.array([5.0, 15.0]),
            obstacle_type=ObstacleType.BUILDING
        ),
        Obstacle(
            position=np.array([70.0, 40.0, 0.0]),
            size=np.array([2.0, 8.0]),
            obstacle_type=ObstacleType.POLE
        ),
    ]
    drone.set_obstacles(obstacles)
    print(f"  Added {len(obstacles)} obstacles")
    print()

    # Add delivery requests
    print("Adding delivery requests...")
    deliveries = [
        ([80.0, 30.0, 0.0], 3, "Critical - Medical supplies"),
        ([40.0, 60.0, 0.0], 2, "Urgent - Food delivery"),
        ([100.0, -20.0, 0.0], 1, "Normal - Package"),
    ]

    for target, priority, desc in deliveries:
        delivery = drone.add_delivery(target, priority=priority)
        print(f"  Delivery {delivery.id}: {desc}")
        print(f"    Target: ({target[0]:.0f}, {target[1]:.0f})")
        print(f"    Priority: {priority}")
    print()

    # Run the mission
    print("Starting autonomous mission...")
    print("-" * 70)

    step = 0
    last_state = None

    while not drone.is_mission_complete() and step < 3000:
        state, reward, done, info = drone.step()
        step += 1

        # Print updates on state changes
        current_state = info['system_state']
        if current_state != last_state:
            print(f"\nStep {step}: State changed to {current_state.upper()}")
            status = drone.get_status()
            print(f"  Position: ({status.position[0]:.1f}, {status.position[1]:.1f}, {status.position[2]:.1f})")
            print(f"  Deliveries: {status.deliveries_completed} completed, {status.deliveries_pending} pending")
            last_state = current_state

        # Periodic progress updates
        if step % 500 == 0:
            status = drone.get_status()
            print(f"\n[Step {step}] Progress update:")
            print(f"  State: {status.state.value}")
            print(f"  Position: ({status.position[0]:.1f}, {status.position[1]:.1f})")
            print(f"  Waypoint: {status.current_waypoint}/{status.total_waypoints}")
            print(f"  Obstacles nearby: {status.obstacles_detected}")

    print("-" * 70)
    print()

    # Mission summary
    summary = drone.get_mission_summary()
    print("MISSION COMPLETE!")
    print()
    print("Results:")
    print(f"  Total steps:        {summary['total_steps']}")
    print(f"  Total distance:     {summary['total_distance_m']:.1f} m")
    print(f"  Deliveries done:    {summary['manager']['completed_count']}")
    print(f"  Deliveries failed:  {summary['manager']['failed_count']}")
    print(f"  Total score:        {summary['manager']['total_score']:.1f}")
    print(f"  Drone crashed:      {summary['crashed']}")
    print()

    print("Layer Summary:")
    print("  Manager:     Scheduled deliveries by priority")
    print("  PathFinder:  Planned paths around obstacles")
    print("  Perception:  Tracked obstacles during flight")
    print("  FlyControl:  Executed flight maneuvers")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
