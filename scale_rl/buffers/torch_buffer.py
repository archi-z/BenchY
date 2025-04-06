from collections import deque

import gymnasium as gym
import numpy as np
import torch

from scale_rl.buffers.base_buffer import BaseBuffer, Batch


class TorchUniformBuffer(BaseBuffer):
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
        device: torch.device
    ):
        super().__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            add_batch_size,
            sample_batch_size
        )

        self._current_idx = 0
        self._device = device

    def __len__(self):
        return self._num_in_buffer

    def reset(self):
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

        self._n_step_transitions = deque(maxlen=self._n_step)
        self._num_in_buffer = 0

    def _get_n_step_prev_timestep(self) -> Batch:
        """
        This method processes a n_step_transitions to compute and update the
            n-step return, the done status, and the next observation.
        """
        # pop n-step previous timestep
        n_step_prev_timestep = self._n_step_transitions[0]
        cur_timestep = self._n_step_transitions[-1]

        # copy (np.array(,) generates copy version of array) last timestep.
        n_step_reward = np.array(cur_timestep["reward"])
        n_step_terminated = np.array(cur_timestep["terminated"])
        n_step_truncated = np.array(cur_timestep["truncated"])
        n_step_next_observation = np.array(cur_timestep["next_observation"])

        for n_step_idx in reversed(range(self._n_step - 1)):
            transition = self._n_step_transitions[n_step_idx]
            reward = transition["reward"]  # (n, )
            terminated = transition["terminated"]  # (n, )
            truncated = transition["truncated"]  # (n, )
            next_observation = transition["next_observation"]  # (n, *obs_shape)

            # compute n-step return
            done = (terminated.astype(bool) | truncated.astype(bool)).astype(np.float32)
            n_step_reward = reward + self._gamma * n_step_reward * (1 - done)

            # assign next observation starting from done
            done_mask = done.astype(bool)
            n_step_terminated[done_mask] = terminated[done_mask]
            n_step_truncated[done_mask] = truncated[done_mask]
            n_step_next_observation[done_mask] = next_observation[done_mask]

        n_step_prev_timestep["reward"] = n_step_reward
        n_step_prev_timestep["terminated"] = n_step_terminated
        n_step_prev_timestep["truncated"] = n_step_truncated
        n_step_prev_timestep["next_observation"] = n_step_next_observation

        return n_step_prev_timestep

    def add(self, timestep: Batch) -> None:
        # temporarily hold current timestep to the buffer
        self._n_step_transitions.append(
            {key: np.array(value) for key, value in timestep.items()}
        )

        if len(self._n_step_transitions) >= self._n_step:
            n_step_prev_timestep = self._get_n_step_prev_timestep()

            # add samples to the buffer
            add_idxs = np.arange(self._add_batch_size) + self._current_idx
            add_idxs = add_idxs % self._max_length

            self._observations[add_idxs] = n_step_prev_timestep["observation"]
            self._actions[add_idxs] = n_step_prev_timestep["action"]
            self._rewards[add_idxs] = n_step_prev_timestep["reward"]
            self._terminateds[add_idxs] = n_step_prev_timestep["terminated"]
            self._truncateds[add_idxs] = n_step_prev_timestep["truncated"]
            self._next_observations[add_idxs] = n_step_prev_timestep["next_observation"]

            self._num_in_buffer = min(
                self._num_in_buffer + self._add_batch_size, self._max_length
            )
            self._current_idx = (
                self._current_idx + self._add_batch_size
            ) % self._max_length

    def can_sample(self) -> bool:
        return self._num_in_buffer >= self._min_length

    def sample(self) -> Batch:
        sample_idxs = np.random.randint(
            0, self._num_in_buffer, size=self._sample_batch_size
        )

        # copy the data for safeness
        batch = {}
        batch["observation"] = torch.as_tensor(self._observations[sample_idxs], device=self._device)
        batch["action"] = torch.as_tensor(self._actions[sample_idxs], device=self._device)
        batch["reward"] = torch.as_tensor(self._rewards[sample_idxs], device=self._device)
        batch["terminated"] = torch.as_tensor(self._terminateds[sample_idxs], device=self._device)
        batch["truncated"] = torch.as_tensor(self._truncateds[sample_idxs], device=self._device)
        batch["next_observation"] = torch.as_tensor(self._next_observations[sample_idxs], device=self._device)

        return batch
    
    def get_observations(self) -> np.ndarray:
        return self._observations[: self._num_in_buffer]
