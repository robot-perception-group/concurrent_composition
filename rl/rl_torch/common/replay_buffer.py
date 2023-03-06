from collections import deque
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler


class Memory:
    def __init__(self, capacity, state_shape, feature_shape, action_shape, device):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.feature_shape = feature_shape
        self.action_shape = action_shape
        self.device = device
        self.is_image = len(state_shape) == 3
        self.state_type = np.uint8 if self.is_image else np.float32

        self.reset()

    def append(
        self, state, feature, action, reward, next_state, done, episode_done=None
    ):
        state, feature, next_state = self.process_before_store(
            state, feature, next_state
        )
        self._append((state, feature, action, reward, next_state, done))

    def process_before_store(self, state, feature, next_state):
        state = np.array(state, dtype=np.float32)
        feature = np.array(feature, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        if self.is_image:
            state = (state * 255).astype(np.uint8)
            next_state = (next_state * 255).astype(np.uint8)
        return state, feature, next_state

    def _append(self, batch):
        n_sample = batch[0].shape[0]
        if self._p + n_sample <= self.capacity:
            self._insert(slice(self._p, self._p + n_sample), batch, slice(0, n_sample))
        else:
            mid_index = self.capacity - self._p
            end_index = n_sample - mid_index
            self._insert(slice(self._p, self.capacity), batch, slice(0, mid_index))
            self._insert(slice(0, end_index), batch, slice(mid_index, n_sample))

        self._n = min(self._n + n_sample, self.capacity)
        self._p = (self._p + n_sample) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return self._sample(indices)

    def _sample(self, indices):
        if self.is_image:
            states = self.states[indices].astype(np.float32) / 255.0
            next_states = self.next_states[indices].astype(np.float32) / 255.0
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]

        states = torch.FloatTensor(states).to(self.device)
        features = torch.FloatTensor(self.features[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, features, actions, rewards, next_states, dones

    def __len__(self):
        return self._n

    def reset(self):
        self._n = 0
        self._p = 0

        self.states = np.zeros(
            (self.capacity, *self.state_shape), dtype=self.state_type
        )
        self.features = np.zeros((self.capacity, *self.feature_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros(
            (self.capacity, *self.state_shape), dtype=self.state_type
        )
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.features[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
        )

    def load_memory(self, batch):
        num_data = len(batch[0])

        if self._p + num_data <= self.capacity:
            self._insert(slice(self._p, self._p + num_data), batch, slice(0, num_data))
        else:
            mid_index = self.capacity - self._p
            end_index = num_data - mid_index
            self._insert(slice(self._p, self.capacity), batch, slice(0, mid_index))
            self._insert(slice(0, end_index), batch, slice(mid_index, num_data))

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _insert(self, mem_indices, batch, batch_indices):
        states, features, actions, rewards, next_states, dones = batch
        self.states[mem_indices] = states[batch_indices]
        self.features[mem_indices] = features[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]


class MyMultiStepBuff:
    keys = ["state", "feature", "action", "reward"]

    def __init__(self, n_env, maxlen=3):
        super().__init__()
        self.n_env = int(n_env)
        self.maxlen = int(maxlen)
        self.memory = {
            key: [deque(maxlen=self.maxlen) for _ in range(n_env)] for key in self.keys
        }

    def append(self, state, feature, action, reward):
        [self.memory["state"][i].append(state[i]) for i in range(self.n_env)]
        [self.memory["feature"][i].append(feature[i]) for i in range(self.n_env)]
        [self.memory["action"][i].append(action[i]) for i in range(self.n_env)]
        [self.memory["reward"][i].append(reward[i]) for i in range(self.n_env)]

    def get(self, gamma=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma)
        feature = self._multi_step_feature(gamma)
        _ = self._pop("reward")
        _ = self._pop("feature")

        state = self._pop("state")
        action = self._pop("action")
        return state, feature, action, reward

    def _pop(self, name):
        return np.array([self.memory[name][i].popleft() for i in range(self.n_env)])

    def _multi_step_reward(self, gamma):
        return np.array(
            [
                np.sum(
                    [r * (gamma**i) for i, r in enumerate(self.memory["reward"][j])]
                )
                for j in range(self.n_env)
            ]
        )[:, None]

    def _multi_step_feature(self, gamma):
        return np.array(
            [
                np.sum(
                    [f * (gamma**i) for i, f in enumerate(self.memory["feature"][j])],
                    0,
                )
                for j in range(self.n_env)
            ]
        )

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f"There is no key {key} in MultiStepBuff.")
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            [self.memory[key][i].clear() for i in range(self.n_env)]

    def reset_by_indice(self, indices):
        for key in self.keys:
            [self.memory[key][i].clear() for i in indices]

    def __len__(self):
        return len(self.memory["state"][0])


class MyMultiStepMemory(Memory):
    def __init__(
        self,
        capacity,
        state_shape,
        feature_shape,
        action_shape,
        n_env,
        device,
        gamma=0.99,
        multi_step=3,
        **kwargs,
    ):
        super().__init__(capacity, state_shape, feature_shape, action_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MyMultiStepBuff(n_env=n_env, maxlen=self.multi_step)

    def append(
        self, state, feature, action, reward, next_state, done, episode_done=False
    ):
        state, feature, next_state = self.process_before_store(
            state, feature, next_state
        )

        if self.multi_step != 1:
            self.buff.append(state, feature, action, reward)

            if len(self.buff) == self.multi_step:
                state, feature, action, reward = self.buff.get(self.gamma)
                self._append((state, feature, action, reward, next_state, done))

            indices = np.where(np.squeeze((episode_done == True) | (done == True)))[0]
            if indices.size != 0:
                self.buff.reset_by_indice(indices)
        else:
            self._append((state, feature, action, reward, next_state, done))


class MyPrioritizedMemory(MyMultiStepMemory):
    def __init__(
        self,
        capacity,
        state_shape,
        feature_shape,
        action_shape,
        n_env,
        device,
        gamma=0.99,
        multi_step=3,
        alpha=0.6,
        beta=0.4,
        beta_annealing=0.001,
        epsilon=1e-4,
        **kwargs,
    ):

        super().__init__(
            capacity,
            state_shape,
            feature_shape,
            action_shape,
            n_env,
            device,
            gamma,
            multi_step,
        )
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

    def append(
        self,
        state,
        feature,
        action,
        reward,
        next_state,
        done,
        error,
        episode_done=False,
    ):
        state, feature, next_state = self.process_before_store(
            state, feature, next_state
        )

        if self.multi_step != 1:
            self.buff.append(state, feature, action, reward)

            if len(self.buff) == self.multi_step:
                state, feature, action, reward = self.buff.get(self.gamma)
                priorities = self.calc_priority(error)
                self._append(
                    (state, feature, action, reward, next_state, done, priorities)
                )

            indices = np.where(np.squeeze((episode_done == True) | (done == True)))[0]
            if indices.size != 0:
                self.buff.reset_by_indice(indices)
        else:
            priorities = self.calc_priority(error)
            self._append((state, feature, action, reward, next_state, done, priorities))

    def update_priority(self, indices, errors):
        self.priorities[indices] = np.reshape(self.calc_priority(errors), (-1, 1))

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size):
        self.beta = min(1.0 - self.epsilon, self.beta + self.beta_annealing)
        sampler = WeightedRandomSampler(self.priorities[: self._n, 0], batch_size)
        indices = list(sampler)
        batch = self._sample(indices)

        p = self.priorities[indices] / np.sum(self.priorities[: self._n])
        weights = (self._n * p) ** -self.beta
        weights /= np.max(weights)
        weights = torch.FloatTensor(weights).to(self.device)

        return batch, indices, weights

    def reset(self):
        super().reset()
        self.priorities = np.empty((self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.features[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
            self.priorities[valid],
        )

    def _insert(self, mem_indices, batch, batch_indices):
        states, features, actions, rewards, next_states, dones, priorities = batch
        self.states[mem_indices] = states[batch_indices]
        self.features[mem_indices] = features[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
        self.priorities[mem_indices] = priorities[batch_indices]


# ======================== RNN replay buffer ======================= #


class RNNMemory(Memory):
    def __init__(
        self,
        capacity,
        state_shape,
        feature_shape,
        action_shape,
        hidden_shape,
        device,
    ):
        self.hidden_shape = hidden_shape
        super().__init__(capacity, state_shape, feature_shape, action_shape, device)

        self.reset()

    def append(
        self,
        state,
        feature,
        action,
        reward,
        next_state,
        done,
        hin,
        hout,
        episode_done=None,
    ):
        state, feature, next_state = self.process_before_store(
            state, feature, next_state
        )

        self._append((state, feature, action, reward, next_state, done, hin, hout))

    def _sample(self, indices):
        states, features, actions, rewards, next_states, dones = super()._sample(
            indices
        )
        hin = torch.FloatTensor(self.hin[indices]).to(self.device).unsqueeze(0)
        hout = torch.FloatTensor(self.hout[indices]).to(self.device).unsqueeze(0)

        return states, features, actions, rewards, next_states, dones, hin, hout

    def __len__(self):
        return self._n

    def reset(self):
        super().reset()

        self.hin = np.zeros((self.capacity, *self.hidden_shape), dtype=np.float32)
        self.hout = np.zeros((self.capacity, *self.hidden_shape), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.features[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
            self.hin[valid],
            self.hout[valid],
        )

    def _insert(self, mem_indices, batch, batch_indices):
        states, features, actions, rewards, next_states, dones, hin, hout = batch
        self.states[mem_indices] = states[batch_indices]
        self.features[mem_indices] = features[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
        self.hin[mem_indices] = hin[batch_indices]
        self.hout[mem_indices] = hout[batch_indices]


class MyMultiStepRNNBuff(MyMultiStepBuff):
    keys = ["state", "feature", "action", "reward", "hin", "hout"]

    def __init__(self, n_env, maxlen=3):
        super().__init__(n_env, maxlen)

    def append(self, state, feature, action, reward, hin, hout):
        super().append(state, feature, action, reward)
        [self.memory["hin"][i].append(hin[i]) for i in range(self.n_env)]
        [self.memory["hout"][i].append(hout[i]) for i in range(self.n_env)]

    def get(self, gamma=0.99):
        state, feature, action, reward = super().get(gamma)
        hin = self._pop("hin")
        hout = self._pop("hout")
        return state, feature, action, reward, hin, hout


class MyMultiStepRNNMemory(RNNMemory):
    def __init__(
        self,
        capacity,
        state_shape,
        feature_shape,
        action_shape,
        hidden_shape,
        n_env,
        device,
        gamma=0.99,
        multi_step=3,
        **kwargs,
    ):
        super().__init__(
            capacity, state_shape, feature_shape, action_shape, hidden_shape, device
        )

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MyMultiStepRNNBuff(n_env=n_env, maxlen=self.multi_step)

    def append(
        self,
        state,
        feature,
        action,
        reward,
        next_state,
        done,
        hin,
        hout,
        episode_done=False,
    ):
        state, feature, next_state = self.process_before_store(
            state, feature, next_state
        )

        if self.multi_step != 1:
            self.buff.append(state, feature, action, reward, hin, hout)

            if len(self.buff) == self.multi_step:
                state, feature, action, reward, hin, hout = self.buff.get(self.gamma)
                self._append(
                    (state, feature, action, reward, next_state, done, hin, hout)
                )

            indices = np.where(np.squeeze((episode_done == True) | (done == True)))[0]
            if indices.size != 0:
                self.buff.reset_by_indice(indices)
        else:
            self._append((state, feature, action, reward, next_state, done, hin, hout))


class MyPrioritizedRNNMemory(MyMultiStepRNNMemory):
    def __init__(
        self,
        capacity,
        state_shape,
        feature_shape,
        action_shape,
        hidden_shape,
        n_env,
        device,
        gamma=0.99,
        multi_step=3,
        alpha=0.6,
        beta=0.4,
        beta_annealing=0.001,
        epsilon=1e-4,
        **kwargs,
    ):

        super().__init__(
            capacity,
            state_shape,
            feature_shape,
            action_shape,
            hidden_shape,
            n_env,
            device,
            gamma,
            multi_step,
        )
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

    def append(
        self,
        state,
        feature,
        action,
        reward,
        next_state,
        done,
        hin,
        hout,
        error,
        episode_done=False,
    ):
        state, feature, next_state = self.process_before_store(
            state, feature, next_state
        )

        if self.multi_step != 1:
            self.buff.append(state, feature, action, reward, hin, hout)

            if len(self.buff) == self.multi_step:
                state, feature, action, reward, hin, hout = self.buff.get(self.gamma)
                priorities = self.calc_priority(error)
                self._append(
                    state,
                    feature,
                    action,
                    reward,
                    next_state,
                    done,
                    hin,
                    hout,
                    priorities,
                )

            indices = np.where(np.squeeze((episode_done == True) | (done == True)))[0]
            if indices.size != 0:
                self.buff.reset_by_indice(indices)
        else:
            priorities = self.calc_priority(error)
            self._append(
                (
                    state,
                    feature,
                    action,
                    reward,
                    next_state,
                    done,
                    hin,
                    hout,
                    priorities,
                )
            )

    def update_priority(self, indices, errors):
        self.priorities[indices] = np.reshape(self.calc_priority(errors), (-1, 1))

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size):
        self.beta = min(1.0 - self.epsilon, self.beta + self.beta_annealing)
        sampler = WeightedRandomSampler(self.priorities[: self._n, 0], batch_size)
        indices = list(sampler)
        batch = self._sample(indices)

        p = self.priorities[indices] / np.sum(self.priorities[: self._n])
        weights = (self._n * p) ** -self.beta
        weights /= np.max(weights)
        weights = torch.FloatTensor(weights).to(self.device)

        return batch, indices, weights

    def reset(self):
        super().reset()
        self.priorities = np.empty((self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.features[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
            self.hin[valid],
            self.hout[valid],
            self.priorities[valid],
        )

    def _insert(self, mem_indices, batch, batch_indices):
        (
            states,
            features,
            actions,
            rewards,
            next_states,
            dones,
            hin,
            hout,
            priorities,
        ) = batch
        self.states[mem_indices] = states[batch_indices]
        self.features[mem_indices] = features[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
        self.hin[mem_indices] = hin[batch_indices]
        self.hout[mem_indices] = hout[batch_indices]
        self.priorities[mem_indices] = priorities[batch_indices]
