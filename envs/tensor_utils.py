import torch
import numpy as np

from .economy import jesusfv

def compute_transition_outcomes(state, action, max_prod, transition_matrix, env):
    """
    Args:
        state: (B, D) tensor, normalized, assume last
        action: (B, 1) tensor, consumption
        max_prod: (B, 1) tensor, upper bounds on production
        transition_matrix: (Z, Z) numpy array (row: z, col: z')

    Returns:
        next_states: (B, Z, 2) tensor where next_states[b, z'] = (k', z')
        probs: (B, Z) tensor where probs[b, z'] = P(z' | z)
    """
    device = state.device
    B, D = state.shape
    Z = transition_matrix.shape[1]

    state = env.unnormalize_tensor(state)

    k = state[:, 0]
    z = state[:, -1].long()

    c = torch.minimum(action.view(-1), max_prod.view(-1))  # shape (B,)
    k_prime = torch.clamp(max_prod.view(-1) - c, min=0.0)    # shape (B,)

    # Build (B, Z) transition probabilities
    T = torch.tensor(transition_matrix, dtype=torch.float32, device=device)  # shape (Z, Z)
    probs = T[z]  # shape (B, Z) — row-wise lookup of transition probabilities

    static_middle = state[:, 1:-1]

    # Create all (k', z') combinations
    z_prime_vals = torch.arange(Z, dtype=torch.float32, device=device).view(1, Z)  # shape (1, Z)
    z_prime_expanded = z_prime_vals.expand(B, -1)                                  # shape (B, Z)

    k_prime_expanded = k_prime.view(B, 1).expand(-1, Z)                            # shape (B, Z)
    static_expanded = static_middle.unsqueeze(1).expand(-1, Z, -1)

    # Stack into (B, Z, 2): each next state = (k', z')
    next_states = torch.cat([
        k_prime_expanded.unsqueeze(2),     # (B, Z, 1)
        static_expanded,                   # (B, Z, D - 2)
        z_prime_expanded.unsqueeze(2)      # (B, Z, 1)
    ], dim=2)  # → (B, Z, D)
    #next_states = torch.stack([k_prime_expanded, z_prime_expanded], dim=2)         # shape (B, Z, 2)

    return next_states, probs

def test_compute_transition_outcomes():
    from economy import jesusfv, mTransition
    from economy import extended_jesusfv as jesusfv

    B = 3
    states = [jesusfv.sample_state() for _ in range(B)]
    actions = [{'c': s['k'] * 0.5} for s in states]
    max_prods = [jesusfv.max_prod(s) for s in states]

    state_tensor = torch.tensor([jesusfv.normalize_state_from_dict(s) for s in states], dtype=torch.float32)
    action_tensor = torch.tensor([[a['c']] for a in actions], dtype=torch.float32)
    max_prod_tensor = torch.tensor([[m] for m in max_prods], dtype=torch.float32)

    # vectorized version
    next_states, probs = compute_transition_outcomes(state_tensor, action_tensor, max_prod_tensor, mTransition, jesusfv)

    # check for loop version
    for i in range(B):
        utility, dist = jesusfv.distribution(states[i], actions[i])

        for entry in dist:
            expected_k = entry['k']
            expected_z = entry['z']
            expected_p = entry['p']

            if len(next_states[i, expected_z]) > 2:
                actual_k, _, actual_z = next_states[i, expected_z]
            else:
                actual_k, actual_z = next_states[i, expected_z]
            actual_p = probs[i, expected_z]
            assert abs(actual_k.item() - expected_k) < 1e-6, f"k mismatch at {i},{expected_z}"
            assert abs(actual_z.item() - expected_z) < 1e-6, f"z mismatch at {i},{expected_z}"
            assert abs(actual_p.item() - expected_p) < 1e-6, f"p mismatch at {i},{expected_z},{actual_p.item()}, {expected_p}"

    print("passed")

if __name__ == "__main__":
    test_compute_transition_outcomes()
