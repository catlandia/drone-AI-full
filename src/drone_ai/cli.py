"""
Command-line interface for Drone AI.
"""

import argparse
import numpy as np


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Drone AI Full - Complete 4-Layer Autonomous Drone System"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demonstration")
    demo_parser.add_argument(
        "--deliveries", type=int, default=3,
        help="Number of deliveries to simulate"
    )
    demo_parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the RL agent")
    train_parser.add_argument(
        "--steps", type=int, default=100000,
        help="Total training steps"
    )
    train_parser.add_argument(
        "--task", type=str, default="hover",
        choices=["hover", "waypoint", "delivery", "delivery_route"],
        help="Training task"
    )

    # Info command
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo(args.deliveries, args.verbose)
    elif args.command == "train":
        run_training(args.steps, args.task)
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()


def run_demo(num_deliveries: int = 3, verbose: bool = False):
    """Run a demonstration of the full system."""
    from drone_ai import DroneAI

    print("=" * 60)
    print("Drone AI Full - 4-Layer Autonomous Drone System Demo")
    print("=" * 60)
    print()

    # Create system
    drone = DroneAI()
    print(f"System initialized with base at {drone.base_position}")
    print()

    # Add deliveries
    print(f"Adding {num_deliveries} delivery requests...")
    for i in range(num_deliveries):
        angle = 2 * np.pi * i / num_deliveries
        distance = 50 + np.random.randint(0, 50)
        target = [
            distance * np.cos(angle),
            distance * np.sin(angle),
            0.0
        ]
        priority = np.random.randint(1, 4)
        delivery = drone.add_delivery(target, priority=priority)
        print(f"  Delivery {delivery.id}: target={target[:2]}, priority={priority}")

    print()
    print("Running mission...")
    print("-" * 40)

    # Run episode
    result = drone.run_episode(max_steps=5000, verbose=verbose)

    print("-" * 40)
    print()
    print("Mission Results:")
    print(f"  Mission Complete: {result['mission_complete']}")
    print(f"  Total Steps: {result['steps']}")
    print(f"  Total Distance: {result['total_distance_m']:.1f} m")
    print(f"  Deliveries Completed: {result['manager']['completed_count']}")
    print(f"  Total Score: {result['manager']['total_score']:.1f}")
    print()


def run_training(total_steps: int, task: str):
    """Run RL training."""
    from drone_ai.layers.flycontrol import DroneEnv, PPOAgent, TaskType

    print("=" * 60)
    print(f"Training PPO Agent on {task} task")
    print("=" * 60)
    print()

    # Task mapping
    task_map = {
        "hover": TaskType.HOVER,
        "waypoint": TaskType.WAYPOINT,
        "delivery": TaskType.DELIVERY,
        "delivery_route": TaskType.DELIVERY_ROUTE
    }

    # Create environment and agent
    env = DroneEnv(task=task_map[task], difficulty=0.5)
    agent = PPOAgent(obs_dim=env.OBS_DIM, action_dim=4)

    print(f"Environment: {task}")
    print(f"Total steps: {total_steps}")
    print()

    # Training loop
    obs, _ = env.reset()
    episode_reward = 0
    episode_count = 0

    for step in range(total_steps):
        action, info = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        agent.store_transition(
            obs, action, reward,
            info['value'], info['log_prob'],
            terminated or truncated
        )

        obs = next_obs
        episode_reward += reward

        if terminated or truncated:
            obs, _ = env.reset()
            episode_count += 1

            if episode_count % 10 == 0:
                print(f"Episode {episode_count}: reward = {episode_reward:.1f}")

            episode_reward = 0

        # Update agent
        if (step + 1) % agent.config.n_steps == 0:
            update_info = agent.update(obs)
            print(f"Step {step + 1}: loss = {update_info['loss']:.4f}")

    # Save agent
    agent.save("checkpoints/ppo_agent.pt")
    print()
    print(f"Training complete. Agent saved to checkpoints/ppo_agent.pt")


def show_info():
    """Show system information."""
    from drone_ai import __version__

    print("=" * 60)
    print("Drone AI Full - System Information")
    print("=" * 60)
    print()
    print(f"Version: {__version__}")
    print()
    print("Layers:")
    print("  1. Manager     - Mission planning and task scheduling")
    print("  2. PathFinder  - Path planning with A* and RRT")
    print("  3. Perception  - Real-time obstacle detection")
    print("  4. FlyControl  - Low-level flight control with RL")
    print()
    print("Usage:")
    print("  drone-ai demo      - Run demonstration")
    print("  drone-ai train     - Train RL agent")
    print("  drone-ai info      - Show this information")
    print()


if __name__ == "__main__":
    main()
