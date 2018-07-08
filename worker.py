from concurrent.futures import ProcessPoolExecutor

from qtable import naf
from policy import samples, agent

from params import random_parameters, model


def worker(num, params, env, train_episodes, num_attempts, test_state):
    results = []
    print "parameters", num, "train_episodes", train_episodes, "attempts", num_attempts
    for attempt in range(num_attempts):
        print "parameters", num, "attempt", attempt
        q = naf.SetupNAF.setup(env, **params['naf'])
        memory = samples.Samples(10000)
        a = agent.Agent(q, memory)

        strat = agent.RelativeRangeStrat(**params['strat'])
        for _ in range(train_episodes):
            a.episode(env, strat)

        strat = agent.NoExploration(max_iters=20)
        utilities = []
        for _ in range(50):
            utilities.append(a.episode(env, strat, train=False))
        results.append(sum(utilities) / len(utilities))
    print "Worker", num, "done"
    return (params, results)


def helper(args):
    return worker(*args)

import tensorflow as tf
default_params = {'naf': {'learningParameters': {'compress': True,
                                                 'discount': 0.99,
                                                 'learning_rate': 0.001},
                  'nnpParameters': {'hidden_layers': [5],
                                    'nonlinearity': tf.nn.tanh},
                  'nnqParameters': {'hidden_layers': [10],
                                    'nonlinearity': tf.nn.tanh},
                  'nnvParameters': {'hidden_layers': [200],
                                    'nonlinearity': tf.nn.tanh}},
                  'strat': {'hi': 1.1, 'lo': 0.5, 'max_iters': 40}}


def runworkers(num_params, max_iters, num_runs, max_workers, test_state, default_params=False):
    print "num_params", num_params, "max_workers", max_workers
    args = []
    for i in range(num_params):
        if not default_params:
            params = random_parameters()
        else:
            params = default_params
        args.append((i, params, model, max_iters, num_runs, test_state))
    print "args are", args
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        r = [r for r in executor.map(helper, args)]
    return r

from envs.economy import jesusfv
if __name__ == '__main__':
    print "Starting workers from worker"
    print runworkers(1, 400, 5, 3, {'k': .2, 'z': 1}, default_params=default_params)
