import math
import random

import numpy as np


class CobbDouglass(object):

    def __init__(self, theta, tao):
        self.theta = theta
        self.tao = tao

    def utility(self, c, l):
        return math.pow(math.pow(c, self.theta) * math.pow(1 - l, 1 - self.theta), 1 - self.tao) / (1 - tao)


class LogUtility(object):

    def utility(self, c):
        return math.log(c)


class Production(object):

    def __init__(self, alpha, delta, technology):
        self.alpha = alpha
        self.delta = delta
        self.technology = technology

    def production(self, k, z):
        return self.technology[z] * math.pow(k, self.alpha) + (1.0 - self.delta) * k


class Motion(object):

    def __init__(self, transition):
        self.transition = transition
        self.levels = range(len(transition[0, ]))

    def next(self, k, z):
        return np.random.choice(self.levels, p=self.transition[z, ])


class GrowthEconomy(object):

    def __init__(self, utility, production, motion):
        self.utility = utility
        self.production = production
        self.motion = motion

    def iterate(self, instate, action):
        p = self.production.production(**instate)
        action['c'] = min(action['c'], p)
        utility = self.utility.utility(**action)
        next_capital = max(p - action['c'], 0.0)
        return utility, {'k': next_capital, 'z': self.motion.next(**instate)}

    def shape(self):
        return (2, 1)

    def sample_state(self):
        state = {'k': random.random(), 'z': random.randrange(1)}
        return state

    def action_loss(self, action):
        import tensorflow as tf
        tf.cond(tf.greater(0, action), tf.abs(action), 0)
    '''
    def dist_to_valid(self, tf, action, state):
        return tf.cond(action < 0.0, tf.abs(action), 0.0) + tf.cond(action > state[0], action, 0.0)
'''

if __name__ == '__main__':
    import numpy as np
    vProductivity = np.array([0.9792, 0.9896, 1.0000, 1.0106, 1.0212], float)
    mTransition = np.array([[0.9727, 0.0273, 0.0000, 0.0000, 0.0000],
                            [0.0041, 0.9806, 0.0153, 0.0000, 0.0000],
                            [0.0000, 0.0082, 0.9837, 0.0082, 0.0000],
                            [0.0000, 0.0000, 0.0153, 0.9806, 0.0041],
                            [0.0000, 0.0000, 0.0000, 0.0273, 0.9727]], float)
    alpha = 1.0 / 3.0
    delta = 0
    prod = Production(alpha, delta, vProductivity)
    motion = Motion(mTransition)
    print prod.production(.5, 1)
    print motion.next(1, 1)
    economy = GrowthEconomy(LogUtility(), prod, motion)
    print economy.iterate({'k': .5, 'z': 1}, {'c': .5})
