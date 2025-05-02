from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import torch.nn as nn

from qtable import naf
from policy import samples, agent

from params import random_parameters, model


def worker(num, params, env, train_episodes, num_attempts, test_state):
    results = []
    print("parameters", num, "train_episodes", train_episodes, "attempts", num_attempts)
    log_interval = 10
    for attempt in range(num_attempts):
        print("parameters", num, "attempt", attempt)
        q = naf.SetupNAF.setup(env, **params['naf'])
        memory = samples.Samples(10000)
        a = agent.Agent(q, memory)

        strat = agent.RelativeRangeStrat(**params['strat'])
        for i in range(train_episodes):
            utility = a.episode(env, strat)
            if i % log_interval == 0:
                print(f"[Worker {num}] Eval rollout {i + 1}/{train_episodes}, Utility: {utility:.3f}")

        strat = agent.NoExploration(max_iters=20)
        utilities = []
        for _ in range(50):
            utilities.append(a.episode(env, strat, train=False))
        avg_utility = sum(utilities) / len(utilities)
        results.append(avg_utility)
        print(f"[Worker {num}] Avg Utility: {avg_utility:.3f}")
    print("Worker", num, "done")
    return (params, results)


def helper(args):
    return worker(*args)


default_params = {'naf': {'learningParameters': {'compress': True,
                                                 'discount': 0.99,
                                                 'learning_rate': 0.001},
                  'nnpParameters': {'hidden_layers': [5],
                                    'nonlinearity_cls': nn.LeakyReLU},
                  'nnqParameters': {'hidden_layers': [10],
                                    'nonlinearity_cls': nn.LeakyReLU},
                  'nnvParameters': {'hidden_layers': [200],
                                    'nonlinearity_cls': nn.LeakyReLU}},
                  'strat': {'hi': 1.1, 'lo': 0.5, 'max_iters': 40}}


def runworkers(num_params, max_iters, num_runs, max_workers, test_state, default_params=False):
    print("num_params", num_params, "max_workers", max_workers)
    args = []
    for i in range(num_params):
        if not default_params:
            params = random_parameters()
        else:
            params = default_params
        args.append((i, params, model, max_iters, num_runs, test_state))
    print("args are", args)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        r = [r for r in executor.map(helper, args)]
    return r

from envs.economy import jesusfv
if __name__ == '__main__':
    mp.set_start_method('spawn')
    print("Starting workers from worker")
    result = runworkers(5, 400, 5, 5, {'k': .2, 'z': 1}, False)
    import pdb;pdb.set_trace()
