"""
PPO (Proximal Policy Optimization) Agent for Drone Control

Implements a PPO agent optimized for continuous control tasks
like drone flight.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PPOConfig:
    """Configuration for PPO algorithm."""
    # Network architecture
    hidden_sizes: Tuple[int, ...] = (256, 256)
    activation: str = "tanh"

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    # Training parameters
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    normalize_advantages: bool = True

    # Action space
    action_std_init: float = 0.5
    action_std_min: float = 0.1
    action_std_decay: float = 0.95


if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """Actor-Critic neural network for PPO."""

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_sizes: Tuple[int, ...] = (256, 256),
            activation: str = "tanh",
            action_std_init: float = 0.5
        ):
            super().__init__()

            self.obs_dim = obs_dim
            self.action_dim = action_dim

            if activation == "tanh":
                act_fn = nn.Tanh
            elif activation == "relu":
                act_fn = nn.ReLU
            else:
                act_fn = nn.Tanh

            # Build shared feature extractor
            layers = []
            prev_size = obs_dim
            for hidden_size in hidden_sizes[:-1]:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(act_fn())
                prev_size = hidden_size

            self.shared = nn.Sequential(*layers) if layers else nn.Identity()

            # Actor head
            self.actor = nn.Sequential(
                nn.Linear(prev_size, hidden_sizes[-1]),
                act_fn(),
                nn.Linear(hidden_sizes[-1], action_dim),
                nn.Sigmoid()
            )

            # Critic head
            self.critic = nn.Sequential(
                nn.Linear(prev_size, hidden_sizes[-1]),
                act_fn(),
                nn.Linear(hidden_sizes[-1], 1)
            )

            # Learnable log standard deviation
            self.log_std = nn.Parameter(
                torch.ones(action_dim) * np.log(action_std_init)
            )

            self._init_weights()

        def _init_weights(self):
            """Initialize network weights."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                    nn.init.constant_(module.bias, 0)

        def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning action mean and value."""
            features = self.shared(obs)
            action_mean = self.actor(features)
            value = self.critic(features)
            return action_mean, value

        def get_action(
            self,
            obs: torch.Tensor,
            deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Sample action from policy."""
            action_mean, value = self.forward(obs)
            std = self.log_std.exp()

            if deterministic:
                action = action_mean
                log_prob = torch.zeros(obs.shape[0], device=obs.device)
            else:
                dist = Normal(action_mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            action = torch.clamp(action, 0, 1)
            return action, log_prob, value.squeeze(-1)

        def evaluate_actions(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Evaluate actions for PPO update."""
            action_mean, value = self.forward(obs)
            std = self.log_std.exp()

            dist = Normal(action_mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            return log_prob, value.squeeze(-1), entropy


class PPOAgent:
    """
    PPO Agent for drone control.

    Works with or without PyTorch installed.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None,
        device: str = "auto"
    ):
        self.config = config or PPOConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Using random policy.")
            self.policy = None
            self.device = "cpu"
            return

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize network
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            action_std_init=self.config.action_std_init
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )

        # Training stats
        self.total_steps = 0
        self.updates = 0

        # Rollout buffer
        self._buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Select action given observation."""
        if not TORCH_AVAILABLE or self.policy is None:
            # Random policy fallback
            action = np.random.uniform(0, 1, self.action_dim).astype(np.float32)
            return action, {'log_prob': 0.0, 'value': 0.0}

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(obs_tensor, deterministic)

        return (
            action.cpu().numpy().squeeze(0),
            {'log_prob': log_prob.item(), 'value': value.item()}
        )

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store a transition in the rollout buffer."""
        self._buffer['observations'].append(obs)
        self._buffer['actions'].append(action)
        self._buffer['rewards'].append(reward)
        self._buffer['values'].append(value)
        self._buffer['log_probs'].append(log_prob)
        self._buffer['dones'].append(done)
        self.total_steps += 1

    def update(self, last_obs: np.ndarray) -> Dict[str, float]:
        """Perform PPO update."""
        if not TORCH_AVAILABLE or self.policy is None:
            self._clear_buffer()
            return {'loss': 0.0}

        # Get last value
        with torch.no_grad():
            obs_tensor = torch.from_numpy(last_obs).float().unsqueeze(0).to(self.device)
            _, _, last_value = self.policy.get_action(obs_tensor)
            last_value = last_value.item()

        # Compute advantages
        advantages = self._compute_advantages(last_value)

        # Convert buffer to tensors
        observations = torch.FloatTensor(np.array(self._buffer['observations'])).to(self.device)
        actions = torch.FloatTensor(np.array(self._buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(self._buffer['log_probs']).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns = advantages_t + torch.FloatTensor(self._buffer['values']).to(self.device)

        # Normalize advantages
        if self.config.normalize_advantages:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # PPO update
        total_loss = 0.0
        n_samples = len(self._buffer['observations'])

        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, self.config.batch_size):
                end = min(start + self.config.batch_size, n_samples)
                batch_idx = indices[start:end]

                # Get batch
                obs_batch = observations[batch_idx]
                action_batch = actions[batch_idx]
                old_log_prob_batch = old_log_probs[batch_idx]
                advantage_batch = advantages_t[batch_idx]
                return_batch = returns[batch_idx]

                # Evaluate current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    obs_batch, action_batch
                )

                # Compute ratio
                ratio = torch.exp(log_probs - old_log_prob_batch)

                # Clipped surrogate objective
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * advantage_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = 0.5 * ((values - return_batch) ** 2).mean()

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()

        # Clear buffer
        self._clear_buffer()

        # Decay action std
        with torch.no_grad():
            self.policy.log_std.data = torch.clamp(
                self.policy.log_std.data - np.log(1 / self.config.action_std_decay),
                min=np.log(self.config.action_std_min)
            )

        self.updates += 1

        return {
            'loss': total_loss / (self.config.n_epochs * (n_samples // self.config.batch_size + 1)),
        }

    def _compute_advantages(self, last_value: float) -> np.ndarray:
        """Compute GAE advantages."""
        rewards = self._buffer['rewards']
        values = self._buffer['values']
        dones = self._buffer['dones']

        advantages = np.zeros(len(rewards))
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]

            delta = rewards[t] + self.config.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        return advantages

    def _clear_buffer(self):
        """Clear rollout buffer."""
        for key in self._buffer:
            self._buffer[key] = []

    def save(self, path: str):
        """Save agent to file."""
        if not TORCH_AVAILABLE or self.policy is None:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'total_steps': self.total_steps,
            'updates': self.updates
        }
        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load agent from file."""
        if not TORCH_AVAILABLE:
            return

        checkpoint = torch.load(path, map_location=self.device)

        if self.policy is not None:
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.updates = checkpoint.get('updates', 0)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto") -> 'PPOAgent':
        """Create agent from checkpoint file."""
        if not TORCH_AVAILABLE:
            return cls(31, 4, device=device)

        checkpoint = torch.load(path, map_location='cpu')

        config = PPOConfig(**checkpoint['config'])
        agent = cls(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
            config=config,
            device=device
        )
        agent.load(path)

        return agent
