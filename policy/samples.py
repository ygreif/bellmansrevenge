import itertools
import collections

import numpy
import torch

def dict_to_tuple(d):
    lst = []
    for k in ['c', 'k', 'd', 'z']:
        if k in d:
            lst.append(d[k])
    return tuple(lst)


def sample_to_state_tuple(s):
    return dict_to_tuple({'k': s['k'], 'z': s['z']})


class Minibatch(object):

    def __init__(self):
        self.state = []
        self.next_state = []
        self.action = []
        self.reward = []
        self.max_prod = []
        self.original_action = []

    def to_torch(self, device):
        self.state = torch.tensor(self.state, dtype=torch.float32, device=device)
        self.next_state = torch.tensor(self.next_state, dtype=torch.float32, device=device)
        self.action = torch.tensor(self.action, dtype=torch.float32, device=device)
        self.reward = torch.tensor(self.reward, dtype=torch.float32, device=device)
        self.max_prod = torch.tensor(self.max_prod, dtype=torch.float32, device=device)
        self.original_action = torch.tensor(self.original_action, dtype=torch.float32, device=device)
        return self

    def append(self, state, next_state, action, reward, max_prod, original_action):
        self.state.append(dict_to_tuple(state))
        self.next_state.append(dict_to_tuple(next_state))
        self.action.append(dict_to_tuple(action))
        self.reward.append([reward])
        self.max_prod.append([max_prod])
        self.original_action.append(original_action)


class Samples(object):

    def __init__(self, size):
        self.samples = collections.deque(maxlen=size)

    def __len__(self):
        return len(self.samples)

    def append(self, state, next_state, action, reward, max_prod, original_action):
        self.samples.append(
            {'state': state, 'action': action, 'original_action': original_action,
                             'reward': reward, 'next_state': next_state,
                             'max_prod': max_prod})

    def batch(self, size):
        return self._batch_helper(numpy.random.choice(self.samples, size))

    def recent(self, size):
        if len(self.samples) < size:
            return Minibatch()
        return self._batch_helper(itertools.islice(self.samples, len(self.samples) - size, len(self.samples)))

    def _batch_helper(self, samples):
        batch = Minibatch()
        for sample in samples:
            batch.append(sample['state'], sample['next_state'], sample[
                         'action'], sample['reward'], sample['max_prod'], sample['original_action'])
        return batch
