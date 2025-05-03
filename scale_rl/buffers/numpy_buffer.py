from typing import Tuple

import gymnasium as gym
import numpy as np

from scale_rl.buffers.base_buffer import BaseBuffer, Batch
from scale_rl.buffers.utils import SegmentTree


class NpyUniformBuffer(BaseBuffer):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        add_batch_size: int,
        sample_batch_size: int,
    ):
        super(NpyUniformBuffer, self).__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            add_batch_size,
            sample_batch_size,
        )

        self._current_idx = 0
        self._ep_timesteps = np.zeros(self._add_batch_size)
        self._mask = np.zeros(self._max_length)

    def __len__(self):
        return self._num_in_buffer

    def reset(self) -> None:
        m = self._max_length

        # for pixel-based environments, we would prefer uint8 dtype.
        observation_shape = (self._observation_space.shape[-1],)
        observation_dtype = self._observation_space.dtype

        action_shape = (self._action_space.shape[-1],)
        action_dtype = self._action_space.dtype

        # for float64, we enforce it to be float32
        if observation_dtype == "float64":
            observation_dtype = np.float32

        if action_dtype == "float64":
            action_dtype = np.float32

        self._observations = np.empty((m,) + observation_shape, dtype=observation_dtype)
        self._actions = np.empty((m,) + action_shape, dtype=action_dtype)
        self._rewards = np.empty((m,), dtype=np.float32)
        self._terminateds = np.empty((m,), dtype=np.float32)
        self._truncateds = np.empty((m,), dtype=np.float32)
        self._next_observations = np.empty(
            (m,) + observation_shape, dtype=observation_dtype
        )

        self._num_in_buffer = 0

    def multi_step_reward(self, sample_idxs: int) -> Tuple[np.ndarray, np.ndarray]:
        ind = (
            sample_idxs.reshape(-1,1) + 
            np.arange(self._n_step*self._add_batch_size, step=self._add_batch_size).reshape(1,-1)
        ) % self._max_length

        rewards = self._rewards[ind]
        terminateds = self._terminateds[ind]
        truncateds = self._truncateds[ind]
        
        return (
            (rewards * (1-terminateds) * self._gammas).sum(1),
            1-(1-terminateds).prod(1),
            1-(1-truncateds).prod(1)
        )

    def add(self, timestep: Batch, horizon=1) -> None:
        # add samples to the buffer
        add_idxs = np.arange(self._add_batch_size) + self._current_idx
        add_idxs = add_idxs % self._max_length

        self._observations[add_idxs] = timestep["observation"]
        self._actions[add_idxs] = timestep["action"]
        self._rewards[add_idxs] = timestep["reward"]
        self._terminateds[add_idxs] = timestep["terminated"]
        self._truncateds[add_idxs] = timestep["truncated"]
        self._next_observations[add_idxs] = timestep["next_observation"]

        self._ep_timesteps += 1

        self._num_in_buffer = min(
            self._num_in_buffer + self._add_batch_size, self._max_length
        )
        self._current_idx = (
            self._current_idx + self._add_batch_size
        ) % self._max_length

        if max(horizon, self._n_step) > 1:
            mask_idxs = (
                add_idxs[self._ep_timesteps >= max(horizon, self._n_step)]
                +1
                -max(horizon, self._n_step)
            ) % self._max_length
            self._mask[mask_idxs] = 1
            self._mask[add_idxs] = 0
            self._ep_timesteps[(timestep["terminated"]+timestep["truncated"])>0] = 0
        else:
            self._mask[add_idxs] = 1

    def can_sample(self) -> bool:
        if self._num_in_buffer < self._min_length:
            return False
        else:
            return True
        
    def sample_ind(self):
        nz = np.nonzero(self._mask)[0].reshape(-1)
        sampled_ind = np.random.randint(nz.shape[0], size=self._sample_batch_size)
        sampled_ind = nz[sampled_ind]
        return sampled_ind

    def sample(self) -> Batch:
        sample_idxs = self.sample_ind()

        # copy the data for safeness
        batch = {}
        batch["observation"] = self._observations[sample_idxs]
        batch["action"] = self._actions[sample_idxs]
        batch["reward"], batch["terminated"], batch["truncated"] = self.multi_step_reward(sample_idxs)
        batch["next_observation"] = self._next_observations[sample_idxs]

        return batch
    
    def sample_horizon(self, horizon: int) -> Batch:
        sample_idxs = self.sample_ind()
        sample_idxs = (
            sample_idxs.reshape(-1,1) + 
            np.arange(horizon*self._add_batch_size, step=self._add_batch_size).reshape(1,-1)
        ) % self._max_length

        # copy the data for safeness
        batch = {}
        batch["observation"] = self._observations[sample_idxs]
        batch["action"] = self._actions[sample_idxs]
        batch["reward"] = self._rewards[sample_idxs]
        batch["terminated"] = self._terminateds[sample_idxs]
        batch["truncated"] = self._truncateds[sample_idxs]
        batch["next_observation"] = self._next_observations[sample_idxs]

        return batch

    def get_observations(self) -> np.ndarray:
        return self._observations[: self._num_in_buffer]


class NpyPrioritizedBuffer(NpyUniformBuffer):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        add_batch_size: int,
        sample_batch_size: int,
    ):
        super(NpyPrioritizedBuffer, self).__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            add_batch_size,
            sample_batch_size,
        )

    def reset(self) -> None:
        super().reset()
        self._priority_tree = SegmentTree(self._max_length)

    def add(self, timestep: Batch) -> None:
        super().add(timestep)

        # add samples to the priority tree
        # SegmentTree class is not vectorized so just added instance one-by-one.
        if len(self._n_step_transitions) == self._n_step:
            for _ in range(self._add_batch_size):
                self._priority_tree.add(value=self._priority_tree.max)

    def _sample_idx_from_priority_tree(self):
        p_total = self._priority_tree.total  # sum of the priorities
        segment_length = p_total / self._sample_batch_size
        segment_starts = np.arange(self._sample_batch_size) * segment_length
        valid = False

        while not valid:
            # Uniformly sample from within all segments
            samples = (
                np.random.uniform(0.0, segment_length, [self._sample_batch_size])
                + segment_starts
            )
            # Retrieve samples from tree with un-normalised probability
            buffer_idxs, tree_idxs, sample_probs = self._priority_tree.find(samples)
            if np.all(sample_probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        return buffer_idxs, tree_idxs, sample_probs

    def sample(self) -> Batch:
        sample_idxs, tree_idxs, sample_probs = self._sample_idx_from_priority_tree()

        batch = {}
        batch["observation"] = self._observations[sample_idxs]
        batch["action"] = self._actions[sample_idxs]
        batch["reward"] = self._rewards[sample_idxs]
        batch["terminated"] = self._terminateds[sample_idxs]
        batch["truncated"] = self._truncateds[sample_idxs]
        batch["next_observation"] = self._next_observations[sample_idxs]

        batch["tree_idxs"] = tree_idxs
        batch["sample_probs"] = sample_probs

        return batch
