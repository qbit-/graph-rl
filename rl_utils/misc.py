import torch
import numpy as np

from torch.utils.data import Dataset, Sampler, DataLoader
from torch_geometric.data import Batch


def _collate_fn(batch):
    """
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """
    states = list()
    actions = list()
    masks = list()
    returns = list()
    values = list()
    advantages = list()
    log_pis= list()
    # print(batch)
    labels = ["state", "action", "mask","return", "value", "advantage", "log_pi"]

    for i, b in enumerate(batch):
        # datas[i].append(b[labels[i]])
        states.append(b['state'])
        actions.append(b['action'])
        masks.append(b['mask'])
        returns.append(b['return'])
        values.append(b['value'])
        advantages.append(b['advantage'])
        log_pis.append(b['log_pi'])

    # images = torch.stack(images, dim=0)
    states = Batch.from_data_list(states)
    datas = [states, actions, masks, returns, values, advantages, log_pis]
    return dict(zip(labels, datas))
    # return states, actions, masks, values, advantages, log_pis

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
        self.values = np.empty((self.max_size,), dtype=np.float32)
        self.advantages = np.empty((self.max_size,), dtype=np.float32)
        self.log_pis = np.empty((self.max_size,), dtype=np.float32)

    def push_episode(self, episode):
        states, actions, masks, ret, val, adv, log_pi = episode
        episode_len = len(actions)
        self.len = min(self.len + episode_len, self.max_size)
        indices = np.arange(
            self.pointer, self.pointer + episode_len) % self.max_size
        self.states[indices] = states
        self.actions[indices] = actions
        self.masks[indices] = masks
        self.returns[indices] = ret
        self.values[indices] = val
        self.advantages[indices] = adv
        self.log_pis[indices] = log_pi
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
            "value": self.values[index],
            "advantage": self.advantages[index],
            "log_pi": self.log_pis[index]
        }
        return dct

    def __len__(self):
        return self.len


class PointerBuffer(Dataset):
    def __init__(self, state_shape, max_size):
        self.max_size = max_size
        self.len = 0
        self.pointer = 0

        # self.states = np.empty((self.max_size,) + state_shape, dtype=np.float32)
        self.actions = np.empty((self.max_size,), dtype=np.int32)
        # self.masks = np.empty((self.max_size, state_shape[0]), dtype=np.int32)
        # self.returns = np.empty((self.max_size,), dtype=np.float32)
        # self.values = np.empty((self.max_size,), dtype=np.float32)
        self.advantages = np.empty((self.max_size,), dtype=np.float32)
        self.log_pis = torch.empty((self.max_size,), dtype=torch.float32)

    def push_episode(self, episode):
        actions, adv, log_pi = episode
        print(actions.shape,adv.shape, log_pi.size())
        episode_len = actions.shape[1]
        self.len = min(self.len + episode_len, self.max_size)
        indices = np.arange(
            self.pointer, self.pointer + episode_len) % self.max_size
        print(indices)
        self.actions[indices] = actions

        # self.masks[indices] = masks
        # self.returns[indices] = ret
        # self.values[indices] = val
        # self.states[indices] = states

        self.advantages[indices] = adv
        self.log_pis[indices] = log_pi
        self.pointer = (self.pointer + episode_len) % self.max_size

    def rescale_advantages(self):
        adv_centered = self.advantages[:self.len] - self.advantages[:self.len].mean()
        self.advantages[:self.len] = adv_centered / (self.advantages[:self.len].std() + 1e-6)

    def __getitem__(self, index):
        dct = {
            "action": self.actions[index].astype(np.float32),
            "advantage": self.advantages[index],
            "log_pi": self.log_pis[index]
        }
        return dct

    def __len__(self):
        return self.len


class BufferSampler(Sampler):
    def __init__(self, buffer, num_mini_epochs):
        super().__init__(None)
        self.buffer = buffer
        self.num_mini_epochs = num_mini_epochs
        buffer_len = len(self.buffer)
        self.len = buffer_len * num_mini_epochs

        indices = []
        for i in range(num_mini_epochs):
            idx = np.arange(buffer_len)
            np.random.shuffle(idx)
            indices.append(idx)
        self.indices = np.concatenate(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.len


def create_gamma_matrix(tau, matrix_size):
    """
    Matrix of the following form
    --------------------
    1     y   y^2    y^3
    0     1     y    y^2
    0     0     1      y
    0     0     0      1
    --------------------
    for fast gae calculation
    """
    i = np.arange(matrix_size)
    j = np.arange(matrix_size)
    pow_ = i[None, :] - j[:, None]
    mat = np.power(tau, pow_) * (pow_ >= 0)
    return mat

def to_tensor(*args, **kwargs):
    return torch.Tensor(*args, **kwargs)