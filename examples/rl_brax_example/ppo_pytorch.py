# NOTE: torchscript doesn't yet support `from __future__ import annotations`. :(
from typing import Any, Dict, Iterable, NamedTuple, Callable, TypeVar, Union, Sequence, Tuple, Optional
from torch import Tensor
import torch
from torch import nn
from brax import envs
import time
from torch.nn import functional as F
import math
import tqdm
import itertools
from brax.envs import to_torch
import collections


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

PossiblyNestedDict = Dict[K, Union[V, "PossiblyNestedDict[K, V]"]]

shapes: PossiblyNestedDict[str, Tuple[int, ...]] = {
    "x": (10, 3, 3),
    "y": {
        "label": (10,),
        "bounding_box": (10, 2),
    },
}


class StepData(NamedTuple):
    observation: Tensor
    logits: Tensor
    action: Tensor
    reward: Tensor
    done: Tensor
    truncation: Tensor

    def map(self, fn: Callable[[Tensor], Tensor]):
        return type(self)(*[fn(value) for value in self])

    def _map(self, fn: Callable[[Tensor], T]) -> Dict[str, T]:
        return {key: fn(value) for key, value in self._asdict().items()}

    @property
    def shapes(self) -> PossiblyNestedDict[str, Tuple[int, ...]]:
        return self._map(lambda v: v.shape)

    @property
    def devices(self):
        return self._map(lambda v: v.device)

    @property
    def device(self) -> Union[torch.device, None]:
        devices = list(self.devices.values())
        if all(device == devices[0] for device in devices):
            return devices[0]
        return None

    def slice(self, indices: Sequence[int]) -> "StepData":
        """Take a slice of all fields across the first dimension."""
        total_length = self.observation.shape[0]
        index = torch.as_tensor(indices, dtype=torch.int64, device=self.device)

        return self.map(lambda v: v.index_select(0, index=index))
        # mask = torch.zeros(total_length, dtype=torch.bool, device=self.device)
        # mask[indices] = True
        # NOTE: Moves the index tensor to the same device as v, in case there are different devices.
        return self.map(lambda v: v[mask.to(v.device)])


import random


def batch_iter(
    td: StepData, batch_size: int, shuffle: bool = False
) -> Iterable[StepData]:
    """Creates a iterator that yields batches from `td`."""
    length = td.observation.shape[0]
    indices = list(range(length))
    if shuffle:
        random.shuffle(indices)

    for start_index in range(0, length, batch_size):
        batch_indices = indices[start_index : start_index + batch_size]
        batch = td.slice(batch_indices)
        yield batch


class Agent(nn.Module):
    """Standard PPO Agent with GAE and observation normalization."""

    def __init__(
        self,
        policy_layers: Sequence[int],
        value_layers: Sequence[int],
        entropy_cost: float,
        discounting: float,
        reward_scaling: float,
    ):
        super().__init__()

        policy: list[nn.Module] = []
        for w1, w2 in zip(policy_layers, policy_layers[1:]):
            policy.append(nn.Linear(w1, w2))
            policy.append(nn.SiLU())
        policy.pop()  # drop the final activation
        self.policy = nn.Sequential(*policy)
        value: list[nn.Module] = []
        for w1, w2 in zip(value_layers, value_layers[1:]):
            value.append(nn.Linear(w1, w2))
            value.append(nn.SiLU())
        value.pop()  # drop the final activation
        self.value = nn.Sequential(*value)

        self.register_buffer("num_steps", None)
        self.num_steps = torch.zeros(())

        self.register_buffer("running_mean", None)
        self.running_mean = torch.zeros(policy_layers[0])

        self.register_buffer("running_variance", None)
        self.running_variance: Tensor = torch.zeros(policy_layers[0])

        self.entropy_cost = entropy_cost
        self.discounting = discounting
        self.reward_scaling = reward_scaling
        self.lambda_ = 0.95
        self.epsilon = 0.3

    def forward(self, observation: Tensor) -> Tensor:
        logits, action = self.get_logits_action(observation)
        return self.dist_postprocess(action)

    @torch.jit.export
    def dist_create(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        """Normal followed by tanh.

        torch.distribution doesn't work with torch.jit, so we roll our own."""
        assert logits.shape[-1] % 2 == 0
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        scale = F.softplus(scale) + 0.001
        return loc, scale

    @torch.jit.export
    def dist_sample_no_postprocess(self, loc: Tensor, scale: Tensor) -> Tensor:
        return torch.normal(loc, scale)

    @classmethod
    def dist_postprocess(cls, x: Tensor) -> Tensor:
        return torch.tanh(x)

    @torch.jit.export
    def dist_entropy(self, loc: Tensor, scale: Tensor) -> Tensor:
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        entropy = 0.5 + log_normalized
        entropy = entropy * torch.ones_like(loc)
        dist = torch.normal(loc, scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        entropy = entropy + log_det_jacobian
        return entropy.sum(dim=-1)

    @torch.jit.export
    def dist_log_prob(self, loc: Tensor, scale: Tensor, dist: Tensor) -> Tensor:
        log_unnormalized = -0.5 * ((dist - loc) / scale).square()
        log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
        log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
        log_prob = log_unnormalized - log_normalized - log_det_jacobian
        return log_prob.sum(dim=-1)

    @torch.jit.export
    def update_normalization(self, observation: Tensor):
        # self.num_steps += observation.shape[0] * observation.shape[1]
        self.num_steps += observation.shape[0]
        input_to_old_mean = observation - self.running_mean
        # mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
        mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=0)
        self.running_mean = self.running_mean + mean_diff
        input_to_new_mean = observation - self.running_mean
        # var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
        var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=0)
        self.running_variance = self.running_variance + var_diff

    def normalize(self, observation: Tensor) -> Tensor:
        variance = self.running_variance / (self.num_steps + 1.0)
        variance = torch.clip(variance, 1e-6, 1e6)
        return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)

    def get_logits_action(self, observation: Tensor) -> Tuple[Tensor, Tensor]:
        observation = self.normalize(observation)
        logits = self.policy(observation)
        loc, scale = self.dist_create(logits)
        action = self.dist_sample_no_postprocess(loc, scale)
        return logits, action

    @torch.jit.export
    def compute_gae(
        self,
        truncation: Tensor,
        termination: Tensor,
        reward: Tensor,
        values: Tensor,
        bootstrap_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        truncation_mask: torch.Tensor = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = (
            reward + self.discounting * (1 - termination) * values_t_plus_1 - values
        )
        deltas *= truncation_mask

        acc = torch.zeros_like(bootstrap_value)
        vs_minus_v_xs = torch.zeros_like(truncation_mask)

        for ti in range(truncation_mask.shape[0]):
            ti = truncation_mask.shape[0] - ti - 1
            acc = (
                deltas[ti]
                + self.discounting
                * (1 - termination[ti])
                * truncation_mask[ti]
                * self.lambda_
                * acc
            )
            vs_minus_v_xs[ti] = acc

        # Add V(x_s) to get v_s.
        vs = vs_minus_v_xs + values
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
        advantages = (
            reward + self.discounting * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask
        return vs, advantages

    @torch.jit.export
    def loss(self, td: StepData) -> Tensor:
        observation = self.normalize(td.observation)

        policy_logits = self.policy(observation[:-1])
        baseline = self.value(observation)
        baseline = torch.squeeze(baseline, dim=-1)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = baseline[-1]
        baseline = baseline[:-1]
        reward = td.reward * self.reward_scaling
        termination = td.done * (1 - td.truncation)

        loc, scale = self.dist_create(td.logits)
        behaviour_action_log_probs = self.dist_log_prob(loc, scale, td.action)
        loc, scale = self.dist_create(policy_logits)
        target_action_log_probs = self.dist_log_prob(loc, scale, td.action)

        with torch.no_grad():
            vs, advantages = self.compute_gae(
                truncation=td.truncation,
                termination=termination,
                reward=reward,
                values=baseline,
                bootstrap_value=bootstrap_value,
            )

        rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = rho_s.clip(1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = vs - baseline
        v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

        # Entropy reward
        entropy = torch.mean(self.dist_entropy(loc, scale))
        entropy_loss = self.entropy_cost * -entropy

        return policy_loss + v_loss + entropy_loss


"""Finally, some code for unrolling and batching environment data:"""


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    for k in keys:
        items[k] = f(*[getattr(sd, k) for sd in sds])
    return StepData(**items)


from gym.vector import VectorEnv


def eval_unroll(agent: Agent, env: VectorEnv, length: int) -> Tuple[Tensor, Tensor]:
    """Return number of episodes and average reward for a single unroll."""
    observation = env.reset()
    episodes = torch.zeros((), device=observation.device)
    episode_reward = torch.zeros((), device=observation.device)
    for _ in tqdm.tqdm(range(length), desc="eval_unroll", leave=False):
        action = agent(observation)
        observation, reward, done, _ = env.step(action)
        episodes += torch.sum(done)  # type: ignore
        episode_reward += torch.sum(reward)  # type: ignore
    return episodes, episode_reward / episodes


def train_unroll(
    agent: Agent,
    env: VectorEnv,
    total_steps: int,
) -> StepData:
    """Return step data over multiple unrolls."""
    # NOTE: Since this is a vectorenv, no need to reset later.
    observation = env.reset()
    unrolls: dict[str, list[Tensor]] = collections.defaultdict(list)
    unrolls["observation"].append(observation)

    for step in tqdm.tqdm(range(total_steps), desc="Train unroll", leave=False):
        logits, action = agent.get_logits_action(observation)
        processed_action = Agent.dist_postprocess(action)
        observation, reward, done, info = env.step(processed_action)
        assert isinstance(reward, Tensor)
        assert isinstance(done, Tensor)
        truncation = info["truncation"]
        if step != total_steps - 1:  # TODO: Add this last one? or not?
            unrolls["observation"].append(observation)
        unrolls["logits"].append(logits)
        unrolls["action"].append(action)
        unrolls["reward"].append(reward)
        unrolls["done"].append(done)
        unrolls["truncation"].append(truncation)

    td = StepData(
        **{
            key: torch.atleast_2d(torch.stack(values))
            for key, values in unrolls.items()
        }  # (T, N_envs)
    )
    return td


def unroll_first(data: Tensor) -> Tensor:
    data = data.swapaxes(0, 1)
    out = data.reshape([data.shape[0], data.shape[1] * data.shape[2], *data.shape[3:]])
    return out


def train(
    env_name: str = "ant",
    num_envs: int = 2048,
    episode_length: int = 1000,
    device: str = "cuda",
    num_timesteps: int = 30_000_000,
    eval_frequency: int = 10,
    unroll_length: int = 5,
    batch_size: int = 1024,
    num_minibatches: int = 32,
    num_update_epochs: int = 4,
    reward_scaling: float = 0.1,
    entropy_cost: float = 1e-2,
    discounting: float = 0.97,
    learning_rate: float = 3e-4,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
    """Trains a policy via PPO."""
    # gym_name = f"brax-{env_name}-v0"
    # if gym_name not in gym.envs.registry.env_specs:
    #     entry_point = functools.partial(envs.create_gym_env, env_name=env_name)
    #     gym.register(gym_name, entry_point=entry_point)
    # env = gym.make(gym_name, batch_size=num_envs, episode_length=episode_length)

    env: VectorEnv = envs.create_gym_env(
        env_name=env_name, batch_size=num_envs, episode_length=episode_length
    )

    # automatically convert between jax ndarrays and torch tensors:
    env = to_torch.JaxToTorchWrapper(env, device=device)

    # env warmup
    env.reset()
    action = torch.as_tensor(env.action_space.sample(), device=device)
    env.step(action)
    from gym.spaces.utils import flatdim

    # create the agent
    policy_layers = [
        flatdim(env.single_observation_space),
        64,
        64,
        flatdim(env.single_action_space) * 2,  # .shape[-1] * 2,
    ]
    value_layers = [
        flatdim(env.single_observation_space),
        # env.observation_space.shape[-1],
        64,
        64,
        1,
    ]
    agent: Agent = Agent(
        policy_layers=policy_layers,
        value_layers=value_layers,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
    )
    agent = torch.jit.script(agent.to(device))  # type: ignore
    optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

    sps = 0
    total_steps = 0
    total_loss = 0
    for eval_i in range(eval_frequency + 1):

        # Q: Why is the evaluation hook at the start?
        if progress_fn:
            t = time.time()
            with torch.no_grad():
                episode_count, episode_reward = eval_unroll(agent, env, episode_length)
            duration = time.time() - t
            # TODO: only count stats from completed episodes
            episode_avg_length = env.num_envs * episode_length / episode_count
            eval_sps = env.num_envs * episode_length / max(duration, 1e-8)
            progress = {
                "eval/episode_reward": episode_reward.item(),
                "eval/completed_episodes": episode_count,
                "eval/avg_episode_length": episode_avg_length,
                "speed/sps": sps,
                "speed/eval_sps": eval_sps,
                "losses/total_loss": total_loss,
            }
            progress_fn(total_steps, progress)

        if eval_i == eval_frequency:
            break

        num_steps = batch_size * num_minibatches * unroll_length
        num_epochs = num_timesteps // (num_steps * eval_frequency)
        num_unrolls = batch_size * num_minibatches // env.num_envs
        total_loss = 0
        t = time.time()
        for _ in range(num_epochs):
            with torch.no_grad():
                td = train_unroll(
                    agent=agent,
                    env=env,
                    total_steps=num_unrolls * unroll_length,
                )
            # NOTE: Values in td have shape (T, N_envs)
            # Flatten the N 'envs' dimension, preserving the sequential nature of the data.
            # obs_e0_t0 = td.observation[0, 0]
            # obs_e0_t1 = td.observation[1, 0]
            # T = td.observation.shape[0]
            # obs_e1_t0 = td.observation[0, 1]
            # obs_e1_t1 = td.observation[1, 1]
            td = td.map(lambda v: v.swapaxes(0, 1).flatten(0, 1))
            # Simple checks, uncomment if you want.
            # assert (td.observation[0] == obs_e0_t0).all()
            # assert (td.observation[1] == obs_e0_t1).all()
            # assert (td.observation[T + 0] == obs_e1_t0).all()
            # assert (td.observation[T + 1] == obs_e1_t1).all()

            # update normalization statistics
            agent.update_normalization(td.observation)

            epoch_progress_bar = tqdm.tqdm(
                range(num_update_epochs),
                desc="Policy Training",
                leave=False,
                position=1,
            )

            for epoch in epoch_progress_bar:
                # shuffle and batch the data
                # NOTE: I don't know if shuffling makes sense here, since it looks like the ordering
                # within each episode needs to be preserved...
                batch_iterator: Iterable[StepData] = batch_iter(
                    td, batch_size=batch_size + 1, shuffle=False
                )
                batch_iterator = itertools.islice(batch_iterator, num_minibatches)
                batch_iterator = tqdm.tqdm(
                    batch_iterator,
                    desc=f"Train epoch {epoch}",
                    leave=False,
                    position=2,
                    total=num_minibatches,
                )
                for td_minibatch in batch_iterator:
                    # TODO: Fix the issues that are arising with the very last observation.
                    # The problem is, I've simplified some stuff, e.g. the StepData has consistent first dim
                    # now. This isn't exactly the same as they had it though.
                    td_minibatch = StepData(
                        observation=td_minibatch.observation,
                        logits=td_minibatch.logits[:-1],
                        action=td_minibatch.action[:-1],
                        reward=td_minibatch.reward[:-1],
                        done=td_minibatch.done[:-1],
                        truncation=td_minibatch.truncation[:-1],
                    )

                    loss = agent.loss(td_minibatch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.detach()

        duration = time.time() - t
        total_steps += num_epochs * num_steps
        total_loss = total_loss / max(
            num_epochs * num_update_epochs * num_minibatches, 1
        )
        sps = num_epochs * num_steps / max(duration, 1e-8)
