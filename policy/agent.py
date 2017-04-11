import random

import samples


class RelativeRangeStrat(object):

    def __init__(self, lo, hi, max_iters):
        self.lo = lo
        self.hi = hi
        self.max_iters = max_iters

    def explore(self, action, max_action):
        action = action * random.uniform(self.lo, self.hi)
        if action > .9 * max_action:
            return random.uniform(.5, .9) * max_action
        elif action < .1 * max_action:
            return random.uniform(.1, .4) * max_action
        else:
            return action


class NoExploration(object):

    def __init__(self, max_iters):
        self.max_iters = max_iters

    def explore(self, action, max_action):
        return action


class Agent(object):

    def __init__(self, qtable, memory):
        self.qtable = qtable
        self.memory = memory

    def episode(self, model, strat, state=False, train=True):
        reward = 0
        if not state:
            state = model.sample_state()
        q = self.qtable
        for _ in range(strat.max_iters):
            max_prod = model.production.production(**state)
# print "state", [samples.dict_to_tuple(state)], "action", [(max_prod,)]
            action = q.action([samples.dict_to_tuple(state)], [(max_prod,)])[0]
            action = {'c': strat.explore(action, max_prod)}

            if action['c'] <= 0 or max_prod <= 0:
                return -999999

            utility, next_state = model.iterate(state, action)

            if train and len(self.memory.samples) > 100:
                batch = self.memory.batch(100)
                q.trainstep(batch.state, batch.action, batch.max_prod,
                            batch.reward, batch.next_state)

            self.memory.append(state, next_state, action, utility, max_prod)
#            print "state", state, "next_state", next_state, "utility", utility
            reward += utility
            state = next_state
        return reward
