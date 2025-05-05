import torch
import numpy as np

from envs.economy import jesusfv
from policy import diagnostics
from qtable.neuralnetwork import NeuralNetwork
from qtable.neural_grid import NeuralGrid
from qtable.q_utils import render_v_vs_target, renderBestA, renderV

TOLERANCE = 0.0000001

def main(num_iterations=2000,
         batch_size=512,
         print_every=250,
         hidden_sizes=[256, 256],
         lr=1e-3,
         tau=0.01,
         beta=0.95):
    torch.manual_seed(42)
    np.random.seed(42)

    input_dim, action_dim = jesusfv.shape()

    actor = NeuralNetwork(indim=input_dim, enddim=action_dim, hidden_layers=hidden_sizes, output_bias=1)
    critic = NeuralNetwork(indim=input_dim, enddim=1, hidden_layers=hidden_sizes)

    model = NeuralGrid(env=jesusfv, actor=actor, critic=critic, learning_rate=lr, tau=tau, beta=beta)
    model.train_actions_cold_start()

    for step in range(1, num_iterations + 1):
        value_loss = model.value_step(batch_size=batch_size)
        action_loss = model.actor_step(batch_size=batch_size)

        if step % print_every == 0:
            value_error = diagnostics.value_error(model, jesusfv, 500).item()
            euler_error = diagnostics.euler_error(model, jesusfv, 500)
            print(f"Step {step} completed, value_loss {value_loss} action_loss {action_loss} value error {value_error} euler error {euler_error}")



    print("Training is complete")
    renderV(model)
    renderBestA(model).show()
    render_v_vs_target(model).show()
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
