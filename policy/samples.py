import collections
import numpy


def dict_to_tuple(d):
    return tuple(v for _, v in d.iteritems())


class Minibatch(object):

    def __init__(self):
        self.state = []
        self.next_state = []
        self.action = []
        self.reward = []
        self.max_prod = []

    def append(self, state, next_state, action, reward, max_prod):
        self.state.append(dict_to_tuple(state))
        self.next_state.append(dict_to_tuple(next_state))
        self.action.append(dict_to_tuple(action))
        self.reward.append([reward])
        self.max_prod.append([max_prod])


class Samples(object):

    def __init__(self, size):
        self.samples = collections.deque(maxlen=size)

    def append(self, state, next_state, action, reward, max_prod):
        self.samples.append({'state': state, 'action': action,
                             'reward': reward, 'next_state': next_state,
                             'max_prod': max_prod})

    def batch(self, size):
        batch = Minibatch()
        for sample in numpy.random.choice(self.samples, size):
            batch.append(sample['state'], sample['next_state'], sample[
                         'action'], sample['reward'], sample['max_prod'])
        return batch
