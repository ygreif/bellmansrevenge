import torch

from envs.economy import jesusfv
from qtable.neural_grid import NeuralGrid
from qtable.neuralnetwork import NeuralNetwork

# test if the vectorized q computation matches the non-vectorized version
def test_q_matches_reference(neuralgrid, batch_size=10):
    # generate inputs
    states, max_prods = neuralgrid._sample(batch_size)
    actions = [s['k'] * 0.5 for s in states]
    state_tensor = torch.tensor(
        [jesusfv.normalize_state_from_dict(s) for s in states],
        dtype=torch.float32
    )
    action_tensor = torch.tensor(actions, dtype=torch.float32).unsqueeze(1)
    max_prod_tensor = torch.tensor(max_prods, dtype=torch.float32).unsqueeze(1)

    q_tensor_vectorized = neuralgrid.q(state_tensor, action_tensor, max_prod_tensor)  # shape [B, 1]
    q_array_loop = _q_tensor_for_loop(states, actions, neuralgrid)
    q_tensor_loop = torch.tensor(q_array_loop, dtype=torch.float32).unsqueeze(1)

    assert torch.allclose(q_tensor_vectorized, q_tensor_loop, atol=1e-4)

    print("Vectorized and for loop computations match")

def _q_tensor_for_loop(states, actions, neuralgrid):
    value_func = neuralgrid.critic_target
    beta = neuralgrid.beta
    q = []

    for s, a in zip(states, actions):
        utility, dist = jesusfv.distribution(s, {'c': a})
        value = 0.0
        for entry in dist:
            next_state = {'k': entry['k'], 'z': entry['z']}
            next_state_tensor = torch.tensor(
                jesusfv.normalize_state_from_dict(next_state),
                dtype=torch.float32).unsqueeze(0)
            v = value_func(next_state_tensor).item()
            value += entry['p'] * v
        q_val = utility + beta * value
        q.append(q_val)
    return q


if __name__ == "__main__":
    input_dim, action_dim = jesusfv.shape()

    # set output biases to make actor/critic more interesting
    actor = NeuralNetwork(indim=input_dim, enddim=action_dim, hidden_layers=[4], output_bias=-2)
    critic = NeuralNetwork(indim=input_dim, enddim=1, hidden_layers=[4], output_bias=1)

    model = NeuralGrid(env=jesusfv, actor=actor, critic=critic)
    test_q_matches_reference(model)
