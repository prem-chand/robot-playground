"""PPO networks, GAE, and the clipped update, as in robot-playground-v0."""

import torch

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
LR = 3.0e-4
N_EPOCHS = 10
ROLLOUT_STEPS = 2048
MINIBATCH_SIZE = 64
TOTAL_STEPS = 200_000
SEED = 0
ADV_EPS = 1e-8


def _mlp(inputs: int, outputs: int, hidden_sizes: tuple[int, ...]) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []
    last = inputs
    for width in hidden_sizes:
        layers.append(torch.nn.Linear(last, width))
        layers.append(torch.nn.Tanh())
        last = width
    layers.append(torch.nn.Linear(last, outputs))
    return torch.nn.Sequential(*layers)


class PolicyNet(torch.nn.Module):
    def __init__(
        self, observation_size: int, action_size: int, hidden_sizes: tuple[int, ...] = (64, 64)
    ) -> None:
        super().__init__()
        self.mean_head = _mlp(observation_size, action_size, hidden_sizes)
        self.log_std = torch.nn.Parameter(torch.zeros(action_size))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std). log_std is state-independent."""
        return self.mean_head(obs), self.log_std


class ValueNet(torch.nn.Module):
    def __init__(self, observation_size: int, hidden_sizes: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        self.net = _mlp(observation_size, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return values of shape (N, 1)."""
        return self.net(obs)


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    truncated: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, returns). All tensors have length T.

    `next_values[t]` is V(s_{t+1}) computed before any env reset.
    Terminal steps bootstrap 0. Truncated steps bootstrap `next_values[t]`.
    At a done or truncated step, zero the carried next advantage before
    combining that step. Then carry that step's advantage to t-1.
    """
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    next_adv = torch.zeros((), dtype=rewards.dtype)
    for t in reversed(range(steps)):
        bootstrap = torch.zeros((), dtype=rewards.dtype)
        if not bool(dones[t]):
            bootstrap = next_values[t]
        if bool(dones[t]) or bool(truncated[t]):
            next_adv = torch.zeros((), dtype=rewards.dtype)
        delta = rewards[t] + gamma * bootstrap - values[t]
        advantages[t] = delta + gamma * lam * next_adv
        next_adv = advantages[t]
    return advantages, advantages + values


def ppo_update(
    policy: PolicyNet,
    value_net: ValueNet,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
) -> None:
    """Clip the ratio on the log-prob of the stored (pre-clip) actions.

    `old_log_probs`, `advantages`, `returns`, and value predictions have
    shape (T,). `log_prob` is summed over the action dimension.
    """
    steps = observations.shape[0]
    params = list(policy.parameters()) + list(value_net.parameters())
    for _ in range(N_EPOCHS):
        order = torch.randperm(steps)
        for start in range(0, steps, MINIBATCH_SIZE):
            idx = order[start : start + MINIBATCH_SIZE]
            mean, log_std = policy(observations[idx])
            dist = torch.distributions.Normal(mean, log_std.exp())
            new_log_probs = dist.log_prob(actions[idx]).sum(-1)
            ratio = (new_log_probs - old_log_probs[idx]).exp()
            adv = advantages[idx]
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + ADV_EPS)
            clipped = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
            policy_loss = -torch.min(ratio * adv, clipped * adv).mean()
            v = value_net(observations[idx]).reshape(-1)
            value_loss = 0.5 * ((v - returns[idx]) ** 2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            optimizer.step()
