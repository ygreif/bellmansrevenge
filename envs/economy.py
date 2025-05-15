import math
import random

import numpy as np
import torch

class CobbDouglass(object):

    def __init__(self, theta, tao):
        self.theta = theta
        self.tao = tao

    def utility(self, c, l):
        return math.pow(math.pow(c, self.theta) * math.pow(1 - l, 1 - self.theta), 1 - self.tao) / (1 - tao)


class LogUtility(object):

    def utility(self, c):
        return math.log(c)

    def derivative(self, c):
        return 1.0 / c


class Production(object):

    def __init__(self, alpha, delta, technology):
        self.alpha = alpha
        self.delta = delta
        self.technology = technology

    def production(self, k, z, d=False, **kwargs):
        if not d:
            delta = self.delta
        else:
            delta = d
        return self.technology[z] * math.pow(k, self.alpha) + (1.0 - delta) * k

    def derivative(self, k, z, d=False, **kwargs):
        if not d:
            delta = self.delta
        else:
            delta = d
        return self.technology[z] * self.alpha * math.pow(k, self.alpha - 1.0) + (1.0 - delta)

    def __len__(self):
        return len(self.technology)


class Motion(object):

    def __init__(self, transition):
        self.transition = transition
        self.levels = range(len(transition[0, ]))

    def next(self, k, z, d=1.0):
        return np.random.choice(self.levels, p=self.transition[z, ])

    def distribution(self, k, z, d=1.0):
        return [p for p in self.transition[z, ]]


class GrowthEconomy(object):

    def __init__(self, utility, production, motion, uniform_sampling=True):
        self.utility = utility
        self.production = production
        self.motion = motion
        self.uniform_sampling = uniform_sampling

    def max_prod(self, state):
        return self.production.production(**state)

    def iterate(self, instate, action):
        p = self.production.production(**instate)
        action['c'] = min(action['c'], p)
        utility = self.utility.utility(**action)
        next_capital = max(p - action['c'], 0.0)
        return utility, {'k': next_capital, 'z': self.motion.next(**instate)}

    def iterate1d(self, instate, action):
        action = action[0]
        p = self.production.production(**instate)
        action = min(action, p)
        utility = self.utility.utility(action)
        next_capital = max(p - action, 0.0)
        return utility, {'k': next_capital, 'z': self.motion.next(**instate)}

    def distribution(self, instate, action):
        p = self.production.production(**instate)
        action['c'] = min(action['c'], p)
        utility = self.utility.utility(**action)
        next_capital = max(p - action['c'], 0.0)
        return utility, [{'k': next_capital, 'z': z, 'p': p} for z, p in enumerate(self.motion.distribution(**instate)) if p > 0]

    def shape(self):
        return (2, 1)

    def sample_state(self):
        if self.uniform_sampling:
            return {'k': random.random(), 'z': random.randrange(len(self.production))}
        else:
            return {'k': math.sqrt(random.random()), 'z': random.randrange(len(self.production))}

    def normalize_state_from_dict(self, state):
        return self.normalize_state((state['k'], state['z']))

    def normalize_state(self, state):
        return (state[0] - .5, state[1] / 4 - .5)

    def normalize_tensor(self, states):
        return torch.stack([
            states[:, 0] - 0.5,
            states[:, -1] / 4.0 - 0.5
        ], dim=1)

    def unnormalize_tensor(self, states):
        return torch.stack([
            states[:, 0] + 0.5,
            (states[:, 1] + 0.5) * 4
        ], dim=1)

    @property
    def alpha(self):
        return self.production.alpha

    @property
    def delta(self):
        return self.production.delta

# TODO unify RandomDeltagrowtheconomy and Growtheconomy
class RandomDeltaGrowthEconomy(GrowthEconomy):

    def __init__(self, utility, production, motion, uniform_sampling=True):
        super().__init__(utility, production, motion, uniform_sampling)

    def iterate(self, instate, action):
        p = self.production.production(**instate)
        action['c'] = min(action['c'], p)
        utility = self.utility.utility(**action)
        next_capital = max(p - action['c'], 0.0)
        d = instate['d']
        return utility, {'k': next_capital, 'd': d, 'z': self.motion.next(**instate)}

    def distribution(self, instate, action):
        p = self.production.production(**instate)
        action['c'] = min(action['c'], p)
        utility = self.utility.utility(**action)
        next_capital = max(p - action['c'], 0.0)
        d = instate['d']
        return utility, [{'k': next_capital, 'd': d, 'z': z, 'p': p} for z, p in enumerate(self.motion.distribution(**instate)) if p > 0]

    def shape(self):
        return (3, 1)

    def iterate1d(self, instate, action):
        action = action[0]
        p = self.production.production(**instate)
        action = min(action, p)
        utility = self.utility.utility(action)
        next_capital = max(p - action, 0.0)
        d = instate['d']
        return utility, {'k': next_capital, 'd': d, 'z': self.motion.next(**instate)}

    def sample_state(self):
        # don't use in parallel!
        delta = random.random() * .2 + .8
        return {'k':random.random(), 'd': delta, 'z': random.randrange(len(self.production))}

    def normalize_state_from_dict(self, state):
        return self.normalize_state((state['k'], state['d'], state['z']))

    def normalize_state(self, state):
        return (state[0] - .5, state[1] - .9, state[2] / 4 - .5)

    def normalize_tensor(self, states):
        return torch.stack([
            states[:, 0] - 0.5,
            states[:, 1] - 0.9,
            states[:, 2] / 4.0 - 0.5
        ], dim=1)

    def unnormalize_tensor(self, states):
        return torch.stack([
            states[:, 0] + 0.5,
            (states[:, 1] + .9),
            (states[:, 2] + 0.5) * 4
        ], dim=1)


vProductivity = np.array([0.9792, 0.9896, 1.0000, 1.0106, 1.0212], float)
mTransition = np.array([[0.9727, 0.0273, 0.0000, 0.0000, 0.0000],
                        [0.0041, 0.9806, 0.0153, 0.0000, 0.0000],
                        [0.0000, 0.00815, 0.9837, 0.00815, 0.0000],
                        [0.0000, 0.0000, 0.0153, 0.9806, 0.0041],
                        [0.0000, 0.0000, 0.0000, 0.0273, 0.9727]], float)

alpha = 1.0 / 3.0
delta = 1.0 # capital decay
prod = Production(alpha, delta, vProductivity)
motion = Motion(mTransition)
jesusfv = GrowthEconomy(LogUtility(), prod, motion)
extended_jesusfv = RandomDeltaGrowthEconomy(LogUtility(), prod, motion)

if __name__ == '__main__':
    state = jesusfv.sample_state()
    action = {'c': state['k'] / 2.0}
    print(state, action)
    print(jesusfv.distribution(state, action))
