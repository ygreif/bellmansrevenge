import samples


def convergence(q, memory, size=100, delta=.01):
    if len(memory) < size:
        return False
    batch = memory.recent(size)
    actions = q.actions(batch.state, batch.max_prod)

    error = 0
    for i in range(size):
        error += abs(
            (batch.original_action[i] - actions[i][0]) / batch.original_action[i])
    return error / size


def value_error(q, model, iters, state=False):
    if not state:
        state = model.sample_state()
    error = 0
    for _ in range(iters):
        max_prod = model.production.production(**state)
        action = {
            'c': q.action([samples.dict_to_tuple(state)], [(max_prod,)])[0]}
        utility, next_states = model.distribution(state, action)
        states = [state] + [{'k': next_state['k'], 'z': next_state['z']}
                            for next_state in next_states]
        states = [samples.dict_to_tuple(s) for s in states]

        values = q.value(states)
        expected_value = utility
        for next_state, value in zip(next_states, values[1:]):
            expected_value += q.beta * value[0] * next_state['p']
        error += abs((values[0][0] - expected_value) / expected_value)
        _, next_state = model.iterate(state, action)
    return error / iters


def euler_error(q, model, iters, state=False):
    if not state:
        state = model.sample_state()
    error = 0
    for _ in range(iters):
        max_prod = model.production.production(**state)
        c = q.action([samples.dict_to_tuple(state)], [(max_prod,)])[0]
        if c <= 0 or c > max_prod:
            return -9999
        action = {'c': c}
        utility, next_states = model.distribution(state, action)
        states = [samples.sample_to_state_tuple(s) for s in next_states]
        max_prods = [(model.production.production(**s),)
                     for s in next_states]
        actions = q.actions(states, max_prods)
        eps = -1.0
        dUtility = model.utility.derivative(action['c'])
        for idx in range(len(next_states)):
            cPrime = max(min(max_prods[idx][0], actions[idx][0]), 0)
            kPrime = max_prod - action['c']
            dUtilityPrime = model.utility.derivative(cPrime)
            dProdPrime = model.production.derivative(
                kPrime, next_states[idx]['z'])
            eps += next_states[idx]['p'] * q.beta * dUtilityPrime / dUtility * \
                (1 - model.production.delta + dProdPrime)
        error += abs(eps)
        if error < 0:
            import pdb
            pdb.set_trace()
        _, state = model.iterate(state, action)
    return error / iters
