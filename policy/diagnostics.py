import numpy as np
import torch

import math
import matplotlib.pyplot as plt
from policy import samples


def convergence(q, memory, size=100, delta=.01):
    if len(memory) < size:
        return False
    batch = memory.recent(size)
    with torch.no_grad():
        actions = q.actions(batch.state, batch.max_prod, normalize=True)

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
        states = [state] + next_states
        states = [samples.dict_to_tuple(s) for s in states]
        states = torch.tensor(states, dtype=torch.float32, device=q.device)

        with torch.no_grad():
            values = q.value(states)
        expected_value = utility
        for next_state, value in zip(next_states, values[1:]):
            expected_value += q.beta * value[0] * next_state['p']
        error += abs((values[0][0] - expected_value) / expected_value)
        _, next_state = model.iterate(state, action)
    return error / iters

def euler_error(q, model, iters, state=None, debug=False, clip=False, print_every=50):
    if state is None:
        state = model.sample_state()

    error = 0.0
    for i in range(iters):
        max_prod = model.production.production(**state)
        c = q.action([samples.dict_to_tuple(state)], [(max_prod,)], clamp=True)[0]
        if clip:
            c = max(c, .05 * max_prod)
            c = min(c, .95 * max_prod)

        action = {'c': c}
        utility, next_states = model.distribution(state, action)
        #import pdb;pdb.set_trace()
        #states = [samples.sample_to_state_tuple(s) for s in next_states]
        states = [samples.dict_to_tuple(s) for s in next_states]
        max_prods = [(model.production.production(**s),) for s in next_states]

        state_tensor = torch.tensor(states, dtype=torch.float32, device=q.device)
        max_prod_tensor = torch.tensor(max_prods, dtype=torch.float32, device=q.device)

        with torch.no_grad():
            actions = q.actions(state_tensor, max_prod_tensor, normalize=True).cpu().numpy()

        eps = -1.0
        dUtility = model.utility.derivative(c)
        for idx in range(len(next_states)):
            cPrime = max(min(max_prods[idx][0], actions[idx][0]), 0)
            dUtilityPrime = model.utility.derivative(cPrime)
            dProdPrime = model.production.derivative(next_states[idx]['k'], next_states[idx]['z'])
            eps += (
                next_states[idx]['p']
                * q.beta
                * dUtilityPrime
                / dUtility
                * (1 - model.production.delta + dProdPrime)
            )
        if debug and i % print_every == 0:
            print('iter', i, 'eps', eps, 'k', state['k'], 'z', state['z'])
        error += abs(eps)
        _, state = model.iterate(state, action)

    return error / iters

def evaluate_euler_error_grid(q, model, k_vals=None, z_vals=None, delta_vals=None, debug=False, clip=True):
    if k_vals is None:
        # focus on low values of k
        k_vals = np.logspace(-2, 0, 20, base=10)
        k_vals = np.clip(k_vals, 0.01, 0.99)
    if z_vals is None:
        z_vals = list(range(len(model.production.technology)))
    if delta_vals is None:
        delta_vals = np.linspace(0.8, 1.0, 10)

    for z in z_vals:
        error_grid = np.zeros((len(k_vals), len(delta_vals)))

        for i, k in enumerate(k_vals):
            for j, delta in enumerate(delta_vals):
                state = {'k': k, 'z': z, 'd': delta}
                err = euler_error(q, model, iters=1, state=state, debug=False, clip=clip)
                error_grid[i, j] = math.log(err)
                if debug and err > 0.1:
                    print(f"[z={z}] k={k:.2f}, delta={delta:.2f}, error={err:.4f}")

        plt.figure(figsize=(8, 6))
        plt.imshow(error_grid, extent=[delta_vals[0], delta_vals[-1], k_vals[0], k_vals[-1]],
                   aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Euler Error')
        plt.title(f'Euler Error Heatmap (z={z})')
        plt.xlabel('Delta')
        plt.ylabel('Capital (k)')
        plt.tight_layout()
        plt.show()


def naf_debug_diagnostics(naf, env, n_samples=50):
    """
    Collects debugging diagnostics for NAFApproximation.
    Use outside training to inspect P, mu, advantage, and qout.

    Args:
        naf: the trained NAFApproximation instance
        env: the environment used for state and max_prod sampling
        n_samples: number of samples to analyze
    """

    # Sample states and compute tensors
    states_raw = [env.sample_state() for _ in range(n_samples)]
    state_tuples = [(s['k'], s['z']) for s in states_raw]
    max_prod = torch.tensor([[env.production.production(**s)] for s in states_raw],
                            dtype=torch.float32, device=naf.device)
    state_tensor = naf._state_to_torch(state_tuples)
    dummy_action = torch.zeros_like(max_prod)

    # Forward pass
    with torch.no_grad():
        qout = naf.nnq(state_tensor)
        _, _, mu, A, P = naf.forward(state_tuples, dummy_action, max_prod)

    # Diagnostics
    print("======== NAF DEBUG DIAGNOSTICS ========")
    print(f"qout range: min={qout.min().item():.4f}, max={qout.max().item():.4f}, mean={qout.mean().item():.4f}")
    print(f"mu range: min={mu.min().item():.4f}, max={mu.max().item():.4f}, mean={mu.mean().item():.4f}")
    print(f"max_prod range: min={max_prod.min().item():.4f}, max={max_prod.max().item():.4f}")
    print(f"A (advantage): mean={A.mean().item():.4f}, max={A.max().item():.4f}, std={A.std().item():.4f}")

    # Check P matrix strength
    P_norms = P.norm(dim=(1, 2))
    print(f"P matrix norm: mean={P_norms.mean().item():.4f}, max={P_norms.max().item():.4f}, std={P_norms.std().item():.4f}")
    print("P scale:", torch.exp(naf.log_scale).item())

    P_svals = torch.linalg.svdvals(P).detach().cpu()
    print(f"P SVD min: {P_svals.min():.4e}, max: {P_svals.max():.4e}")

    sigmoid_q = torch.sigmoid(qout)
    hist_bins = torch.histc(sigmoid_q, bins=10, min=0.0, max=1.0)
    hist_str = ', '.join(f"{v.item():.2f}" for v in hist_bins)
    print(f"sigmoid(qout) hist: [{hist_str}]")

    mu_vals = mu.detach().cpu().numpy()
    hist_mu, _ = np.histogram(mu_vals, bins=10, range=(0.0, 1.0))
    print("mu hist:", hist_mu.tolist())
    print(f"mu: min={mu.min().item():.4f}, max={mu.max().item():.4f}, mean={mu.mean().item():.4f}")

    # Optional: check sigmoid saturation if using sigmoid
    if naf.compress:
        sig_range = torch.sigmoid(qout)
        print(f"sigmoid(qout): min={sig_range.min().item():.4f}, max={sig_range.max().item():.4f}")

    print("=======================================")

def log_nnp_gradients(self):
    print("---- NNP Gradient Diagnostics ----")
    for name, param in self.nnp.named_parameters():
        if param.grad is None:
            print(f"{name}: no gradient")
            continue
        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.mean().item()
        grad_max = param.grad.abs().max().item()
        print(f"{name:<40} | norm: {grad_norm:.2e} | mean: {grad_mean:.2e} | max: {grad_max:.2e}")
    print("----------------------------------")
