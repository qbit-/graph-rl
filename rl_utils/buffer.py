import queue
import random
from collections import deque

import numpy as np
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Batch


def _collate_fn_geom_weights(batch):
    """
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """
    states = list()
    actions = list()
    masks = list()
    returns = list()
    weights = list()
    advantages = list()
    idxes = list()
    # print(batch)
    labels = ["state", "action", "mask","return", "advantage", "weights", "idxes"] #"value", "advantage", "log_pi"]

    for i, b in enumerate(batch):
        # datas[i].append(b[labels[i]])
        states.append(b['state'])
        actions.append(b['action'])
        masks.append(b['mask'])
        returns.append(b['return'])
        # values.append(b['value'])
        idxes.append(b['idxes'])
        weights.append(b['weights'])
        advantages.append(b['advantage'])
        # log_pis.append(b['log_pi'])

    # images = torch.stack(images, dim=0)
    states = Batch.from_data_list(states)
    datas = [states, actions, masks, returns, advantages, weights, idxes]# values, advantages, log_pis]
    return dict(zip(labels, datas))

def _collate_fn_geom(batch):
    """
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """
    states = list()
    actions = list()
    masks = list()
    returns = list()
    advantages = list()
    labels = ["state", "action", "mask","return", "advantage"]

    for i, b in enumerate(batch):
        states.append(b['state'])
        actions.append(b['action'])
        masks.append(b['mask'])
        returns.append(b['return'])
        advantages.append(b['advantage'])

    states = Batch.from_data_list(states)
    datas = [states, actions, masks, returns, advantages]
    return dict(zip(labels, datas))


class BufferDataset(Dataset):
    def __init__(self, state_shape, max_size, is_graph_data=False):
        # super(BufferDataset, self).__init__(root='.', transform=None, pre_transform=None)
        self.max_size = max_size
        self.len = 0
        self.pointer = 0
        if is_graph_data:
            self.states = np.empty((self.max_size,), dtype=object)
        else:
            self.states = np.empty((self.max_size,) + state_shape, dtype=np.float32)
        self.actions = np.empty((self.max_size,), dtype=np.int32)
        self.masks = np.empty((self.max_size, state_shape[0]), dtype=np.int32)
        self.returns = np.empty((self.max_size,), dtype=np.float32)
        # self.values = np.empty((self.max_size,), dtype=np.float32)
        self.advantages = np.empty((self.max_size,), dtype=np.float32)
        # self.log_pis = np.empty((self.max_size,), dtype=np.float32)

    def push_episode(self, episode):
        states, actions, masks, ret, advantage = episode
        episode_len = len(actions)
        self.len = min(self.len + episode_len, self.max_size)
        indices = np.arange(
            self.pointer, self.pointer + episode_len) % self.max_size
        self.states[indices] = states
        self.actions[indices] = actions
        self.advantages[indices] = advantage[:episode_len]
        self.masks[indices] = masks
        self.returns[indices] = ret[:episode_len]
        self.pointer = (self.pointer + episode_len) % self.max_size

    def rescale_advantages(self):
        adv_centered = self.advantages[:self.len] - self.advantages[:self.len].mean()
        self.advantages[:self.len] = adv_centered / (self.advantages[:self.len].std() + 1e-6)

    def __getitem__(self, index):
        dct = {
            "state": self.states[index],
            "action": self.actions[index].astype(np.float32),
            "mask": self.masks[index].astype(np.float32),
            "return": self.returns[index],
            "advantage": self.advantages[index],
        }
        return dct

    def __len__(self):
        return self.len


class PriorityBufferDataset(Dataset):
    def __init__(self, state_shape, max_size, alpha):
        self.max_size = max_size
        self.len = 0
        self.pointer = 0
        self.states = np.empty((self.max_size,), dtype=object)
        self.actions = np.empty((self.max_size,), dtype=np.int32)
        self.masks = np.empty((self.max_size, state_shape[0]), dtype=np.int32)
        self.returns = np.empty((self.max_size,), dtype=np.float32)
        self.advantages = np.empty((self.max_size,), dtype=np.float32)
        self.priorities = deque(maxlen=self.max_size)
        self.alpha = alpha

        self.probabilities = None

    def push_episode(self, episode):
        states, actions, masks, ret, advantage = episode
        episode_len = len(actions)
        self.len = min(self.len + episode_len, self.max_size)
        indices = np.arange(self.pointer, self.pointer + episode_len) % self.max_size
        self.states[indices] = states
        self.actions[indices] = actions
        self.advantages[indices] = advantage
        self.masks[indices] = masks
        self.returns[indices] = ret
        self.pointer = (self.pointer + episode_len) % self.max_size

        max_priority = max(self.priorities) if len(self.priorities) > 0 else 10.0
        new_priorities = [max_priority] * len(episode[0])
        self.priorities.extend(new_priorities)
        self._update_probabilities()

    def rescale_advantages(self):
        adv_centered = self.advantages[:self.len] - self.advantages[:self.len].mean()
        self.advantages[:self.len] = adv_centered / (self.advantages[:self.len].std() + 1e-6)

    def __getitem__(self, index):
        dct = {
            "state": self.states[index],
            "action": self.actions[index].astype(np.float32),
            "mask": self.masks[index].astype(np.float32),
            "return": self.returns[index],
            "advantage": self.advantages[index],
            "weights": self.probabilities[index],
            "idxes": index
        }

        return dct

    def __len__(self):
        return self.len

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            self.priorities[i] = (priority + 1e-5) ** self.alpha

        self._update_probabilities()

    def _update_probabilities(self):
        self.probabilities = np.array(self.priorities)
        self.probabilities /= np.sum(self.probabilities)


class PrioritisedReplaySampler(Sampler):
    def __init__(self, buffer, num_mini_epochs=0):
        super().__init__(None)
        assert buffer.probabilities is not None
        self.buffer = buffer

    def __iter__(self):
        while len(self.buffer) > 100:
            yield np.random.choice(len(self.buffer.probabilities), p=self.buffer.probabilities)

    def __len__(self):
        return np.Inf
