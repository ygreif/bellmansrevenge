from concurrent.futures import ProcessPoolExecutor

from qtable import naf
from policy import samples, agent

from params import random_parameters, model


def worker(num, params, env, train_episodes, num_attempts):
    results = []
    print "parameters", num, "train_episodes", train_episodes, "attempts", num_attempts
    for attempt in range(num_attempts):
        print "parameters", num, "attempt", attempt
        q = naf.SetupNAF.setup(env, **params['naf'])
        memory = samples.Samples(10000)
        a = agent.Agent(q, memory)

        strat = agent.RelativeRangeStrat(**params['strat'])
        for _ in range(train_episodes):
            a.episode(model, strat)

        strat = agent.NoExploration(max_iters=20)
        utilities = []
        for _ in range(50):
            utilities.append(a.episode(model, strat, train=False))
        results.append(sum(utilities) / len(utilities))
    return (params, results)


def helper(args):
    return worker(*args)


def runworkers(num_params, max_iters, num_runs, max_workers):
    args = []
    for i in range(num_params):
        params = random_parameters()
        args.append((i, params, model, max_iters, num_runs))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        r = [r for r in executor.map(helper, args)]
    return r


if __name__ == '__main__':
    print runworkers(20, 400, 5, 5)
