import random

import numpy as np

import samples


def constructStrategy(episode, n_episodes, **kwargs):
    weight = 1.0 - episode / (n_episodes + 1.0)
    if kwargs.get('delta'):
        return FixedDeltaStrat(kwargs['delta'] * weight, max_iters=kwargs.get('max_iters'), minibatch_size=kwargs.get('minibatch_size', 500))
    elif kwargs.get('lo'):
        return RelativeRangeStrat(kwargs.get('lo'), kwargs.get('hi'), kwargs.get('max_iters'), kwargs.get('minibatch_size', 500))
    else:
        return NoExploration()


class FixedDeltaStrat(object):

    def __init__(self, delta, max_iters, minibatch_size=500):
        self.delta = delta
        self.max_iters = max_iters
        self.minibatch_size = minibatch_size

    def explore(self, action, max_action):
        delta = random.uniform(-self.delta, self.delta)
        # print action, delta, max_action
        if action + delta < .1 * max_action:
            return action
        elif action + delta > .9 * max_action:
            return action
        else:
            return action + delta


class RelativeRangeStrat(object):

    def __init__(self, lo, hi, max_iters, minibatch_size=500):
        if lo > hi:
            self.lo = 1.0
            self.hi = 1.0
        self.lo = lo
        self.hi = hi
        self.max_iters = max_iters
        self.minibatch_size = minibatch_size

    def explore(self, action, max_action):
        action = action * random.uniform(self.lo, self.hi)
        if action > .9 * max_action:
            return random.uniform(.5, .9) * max_action
        elif action < .1 * max_action:
            return random.uniform(.1, .4) * max_action
        elif action == np.nan:
            return random.uniform(.1, .9) * max_action
        else:
            return action


class NoExploration(object):

    def __init__(self, max_iters, minibatch_size=500):
        self.max_iters = max_iters
        self.minibatch_size = minibatch_size

    def explore(self, action, max_action):
        return action


class Agent(object):

    def __init__(self, qtable, memory):
        self.qtable = qtable
        self.memory = memory

    def episode(self, model, strat, state=False, train=True, debug=False):
        reward = 0
        if not state:
            state = model.sample_state()
        q = self.qtable
        discount = 1.0
        for _ in range(strat.max_iters):
            max_prod = model.production.production(**state)
            original_action = q.action(
                [samples.dict_to_tuple(state)], [(max_prod,)])[0]
            action = {'c': strat.explore(original_action, max_prod)}

            if debug:
                print action, original_action, state, max_prod, len(self.memory.samples)
            # print action, original_action
            if action['c'] <= 0 or max_prod <= 0:
                return -99999

            utility, next_state = model.iterate(state, action)
            if utility == np.nan:
                import pdb
                pdb.set_trace()
            if train and len(self.memory.samples) > strat.minibatch_size:
                batch = self.memory.batch(strat.minibatch_size)
                q.trainstep(batch.state, batch.action, batch.max_prod,
                            batch.reward, batch.next_state)

            self.memory.append(
                state, next_state, action, utility, max_prod, original_action)
            reward += utility * discount
            discount *= q.beta
            state = next_state
        return reward
