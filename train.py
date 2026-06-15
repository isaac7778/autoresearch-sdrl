"""
Autoresearch streaming RL training script. Single-file streaming actor-critic agent
for continuous control (MuJoCo / DeepMind Control Suite via Gymnasium).

Algorithm: Stream AC(lambda) with eligibility traces and the ObGD optimizer, ported
from stream_ac_continuous.py in https://github.com/mohmdelsayed/streaming-drl
(Elsayed, Vasan & Mahmood, "Streaming Deep Reinforcement Learning Finally Works", 2024,
arXiv:2410.14606). That repo is distributed under CC BY-NC 4.0 -- see its LICENSE.md.

Usage: uv run train.py
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym

# ---------------------------------------------------------------------------
# Sparse weight initialization
# ---------------------------------------------------------------------------

def sparse_init(tensor, sparsity):
    """LeCun-uniform init for a 2D weight matrix with a random per-row sparsity mask."""
    fan_out, fan_in = tensor.shape
    num_zeros = int(math.ceil(sparsity * fan_in))
    with torch.no_grad():
        tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
        for row in range(fan_out):
            zero_indices = torch.randperm(fan_in)[:num_zeros]
            tensor[row, zero_indices] = 0
    return tensor


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        sparse_init(m.weight, sparsity=0.9)
        m.bias.data.fill_(0.0)

# ---------------------------------------------------------------------------
# Environment wrappers: running observation/reward normalization + episode time
# ---------------------------------------------------------------------------

class SampleMeanStd:
    """Welford-style running mean/variance, used by NormalizeObservation/ScaleReward."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.p = np.ones(shape, "float64")
        self.count = 0

    def update(self, x):
        if self.count == 0:
            self.mean = x
            self.p = np.zeros_like(x)
        new_count = self.count + 1
        new_mean = self.mean + (x - self.mean) / new_count
        self.p = self.p + (x - self.mean) * (x - new_mean)
        self.var = 1 if new_count < 2 else self.p / (new_count - 1)
        self.mean, self.count = new_mean, new_count


class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Normalizes observations by a running mean/std estimated online."""

    def __init__(self, env, epsilon=1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.obs_stats = SampleMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        self.obs_stats.update(obs)
        return (obs - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + self.epsilon)


class ScaleReward(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Scales rewards by a running estimate of the discounted return's std."""

    def __init__(self, env, gamma=0.99, epsilon=1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.reward_stats = SampleMeanStd(shape=())
        self.reward_trace = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        rews = np.array([rews])
        term = terminateds or truncateds
        self.reward_trace = self.reward_trace * self.gamma * (1 - term) + rews
        self.reward_stats.update(self.reward_trace)
        rews = (rews / np.sqrt(self.reward_stats.var + self.epsilon))[0]
        return obs, rews, terminateds, truncateds, infos


class AddTimeInfo(gym.Wrapper):
    """Appends a feature in [-0.5, ...] tracking progress through the episode."""

    def __init__(self, env):
        super().__init__(env)
        self.epi_time = -0.5
        if "dm_control" in env.spec.id:
            self.time_limit = 1000
        else:
            self.time_limit = env.spec.max_episode_steps
        obs_size = self.observation_space.shape[0] + self.env.num_envs
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        obs = np.concatenate((obs, np.array([self.epi_time] * self.env.num_envs)))
        self.epi_time += 1.0 / self.time_limit
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.epi_time = -0.5
        obs = np.concatenate((obs, np.array([self.epi_time])))
        return obs, info

# ---------------------------------------------------------------------------
# Optimizer: ObGD (Overshooting-bounded Gradient Descent)
# ---------------------------------------------------------------------------

class ObGD(torch.optim.Optimizer):
    """Overshooting-bounded gradient descent with eligibility traces (Elsayed et al., 2024)."""

    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super().__init__(params, defaults)

    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        step_size = group["lr"] / dot_product if dot_product > 1 else group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                e = self.state[p]["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()

# ---------------------------------------------------------------------------
# Actor-Critic networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_size=128):
        super().__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_mu = nn.Linear(hidden_size, n_actions)
        self.linear_std = nn.Linear(hidden_size, n_actions)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        mu = self.linear_mu(x)
        std = F.softplus(self.linear_std(x))
        return mu, std


class Critic(nn.Module):
    def __init__(self, n_obs, hidden_size=128):
        super().__init__()
        self.fc_layer = nn.Linear(n_obs, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.linear_layer = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.layer_norm(x, x.size())
        x = F.leaky_relu(x)
        return self.linear_layer(x)


class StreamAC(nn.Module):
    """Streaming actor-critic with TD(lambda) eligibility traces and ObGD updates."""

    def __init__(self, n_obs, n_actions, hidden_size=128, lr=1.0, gamma=0.99, lamda=0.8,
                 kappa_policy=3.0, kappa_value=2.0):
        super().__init__()
        self.gamma = gamma
        self.policy_net = Actor(n_obs, n_actions, hidden_size)
        self.value_net = Critic(n_obs, hidden_size)
        self.optimizer_policy = ObGD(self.policy_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_policy)
        self.optimizer_value = ObGD(self.value_net.parameters(), lr=lr, gamma=gamma, lamda=lamda, kappa=kappa_value)

    def pi(self, x):
        return self.policy_net(x)

    def v(self, x):
        return self.value_net(x)

    def sample_action(self, s):
        x = torch.from_numpy(s).float()
        mu, std = self.pi(x)
        return Normal(mu, std).sample().numpy()

    def update_params(self, s, a, r, s_prime, done, entropy_coeff, overshooting_info=False):
        done_mask = 0 if done else 1
        s = torch.tensor(np.array(s), dtype=torch.float)
        a = torch.tensor(np.array(a))
        r = torch.tensor(np.array(r))
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float)
        done_mask = torch.tensor(np.array(done_mask), dtype=torch.float)

        v_s, v_prime = self.v(s), self.v(s_prime)
        td_target = r + self.gamma * v_prime * done_mask
        delta = td_target - v_s

        dist = Normal(*self.pi(s))
        log_prob_pi = -dist.log_prob(a).sum()
        value_output = -v_s
        entropy_pi = -entropy_coeff * dist.entropy().sum() * torch.sign(delta).item()

        self.optimizer_value.zero_grad()
        self.optimizer_policy.zero_grad()
        value_output.backward()
        (log_prob_pi + entropy_pi).backward()
        self.optimizer_policy.step(delta.item(), reset=done)
        self.optimizer_value.step(delta.item(), reset=done)

        if overshooting_info:
            v_s, v_prime = self.v(s), self.v(s_prime)
            delta_bar = r + self.gamma * v_prime * done_mask - v_s
            if torch.sign(delta_bar * delta).item() == -1:
                print("Overshooting detected!")

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Environment
ENV_NAME = "dm_control/humanoid-walk-v0"  # any Gymnasium continuous-control env, e.g. a MuJoCo
                             # task or "dm_control/<domain>-<task>-v0" (via shimmy)
SEED = 0
TOTAL_STEPS = 1_000_000      # number of environment steps to stream through

# Stream AC(lambda) hyperparameters
LR = 1.0                  # ObGD step size
GAMMA = 0.99              # discount factor
LAMBDA = 0.8              # eligibility trace decay
ENTROPY_COEFF = 0.01      # entropy bonus coefficient
KAPPA_POLICY = 3.0        # ObGD overshoot bound for the policy network
KAPPA_VALUE = 2.0         # ObGD overshoot bound for the value network
HIDDEN_SIZE = 128         # hidden width of the actor/critic MLPs

DEBUG = True              # print episodic returns as episodes finish
OVERSHOOTING_INFO = False # check + print whether the TD-error sign flips after the update

# ---------------------------------------------------------------------------
# Setup: environment + agent
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)
np.random.seed(SEED)

env = gym.make(ENV_NAME)
env = gym.wrappers.FlattenObservation(env)
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ClipAction(env)
env = ScaleReward(env, gamma=GAMMA)
env = NormalizeObservation(env)
env = AddTimeInfo(env)

agent = StreamAC(
    n_obs=env.observation_space.shape[0],
    n_actions=env.action_space.shape[0],
    hidden_size=HIDDEN_SIZE,
    lr=LR, gamma=GAMMA, lamda=LAMBDA,
    kappa_policy=KAPPA_POLICY, kappa_value=KAPPA_VALUE,
)

print(f"Env: {ENV_NAME} | obs_dim: {env.observation_space.shape[0]} | act_dim: {env.action_space.shape[0]}")
print(f"Total steps: {TOTAL_STEPS:,}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

returns, term_time_steps = [], []
s, _ = env.reset(seed=SEED)
for step in range(1, TOTAL_STEPS + 1):
    a = agent.sample_action(s)
    s_prime, r, terminated, truncated, info = env.step(a)
    agent.update_params(s, a, r, s_prime, terminated or truncated, ENTROPY_COEFF, OVERSHOOTING_INFO)
    s = s_prime

    if terminated or truncated:
        ep_return = info["episode"]["r"]
        returns.append(ep_return)
        term_time_steps.append(step)
        s, _ = env.reset()
        if DEBUG:
            mean_recent = np.mean(returns[-10:])
            pct_done = 100 * step / TOTAL_STEPS
            print(f"step {step:>9,}/{TOTAL_STEPS:,} ({pct_done:5.1f}%) | episode {len(returns):5d} | "
                  f"return: {ep_return:9.2f} | mean_return(last10): {mean_recent:9.2f}")

env.close()

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

total_seconds = time.time() - t_start

# Mean return over the final 5% of completed episodes (steady-state performance)
final_cutoff = 0.95 * TOTAL_STEPS
final_returns = [ret for ret, t in zip(returns, term_time_steps) if t > final_cutoff]
if not final_returns:
    final_returns = returns
mean_return = float(np.mean(final_returns)) if final_returns else 0.0

print("---")
print(f"mean_return:      {mean_return:.2f}")
print(f"num_episodes:     {len(returns)}")
print(f"total_steps:      {TOTAL_STEPS}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"env:              {ENV_NAME}")
