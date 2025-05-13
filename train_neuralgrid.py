import argparse

import torch
import numpy as np

from envs.economy import jesusfv, extended_jesusfv
from policy import diagnostics
from qtable.neuralnetwork import NeuralNetwork
from qtable.neural_grid import NeuralGrid
from qtable.q_utils import coldStart, render_v_vs_target, renderBestA, renderV

TOLERANCE = 0.0000001

def main(num_iterations=500,
         batch_size=512,
         print_every=250,
         hidden_sizes=[256, 256],
         lr=1e-3,
         tau=0.01,
         beta=0.95,
         cold_start=False,
         uniform_sampling=True,
         random_delta=True):
    torch.manual_seed(42)
    np.random.seed(42)

    if random_delta:
        env = extended_jesusfv
    else:
        env = jesusfv

    input_dim, action_dim = env.shape()
    env.uniform_sampling = uniform_sampling
    state_0 = env.sample_state()

    actor = NeuralNetwork(indim=input_dim, enddim=action_dim, hidden_layers=hidden_sizes, output_bias=1)
    critic = NeuralNetwork(indim=input_dim, enddim=1, hidden_layers=hidden_sizes)

    model = NeuralGrid(env=env, actor=actor, critic=critic, learning_rate=lr, tau=tau, beta=beta)
    if cold_start:
        model.train_actions_cold_start()
        euler_error = diagnostics.euler_error(model, env, 500, state=state_0)
        print(f"Euler error {euler_error}")
        #diagnostics.evaluate_euler_error_grid(model, model.env, clip=False)

    for step in range(1, num_iterations + 1):
        value_loss = model.value_step(batch_size=batch_size)
        action_loss = model.actor_step(batch_size=batch_size)

        if step % print_every == 0:
            value_error = diagnostics.value_error(model, env, 500).item()
            euler_error = diagnostics.euler_error(model, env, 500, state=state_0)
            euler_error_clip = diagnostics.euler_error(model, env, 500, clip=True, state=state_0)
            print(f"Step {step} completed, value_loss {value_loss} action_loss {action_loss} value error {value_error} euler error {euler_error} euler error clipped {euler_error_clip}")



    print("Training is complete")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', default=500, type=int)
    parser.add_argument('--lr', default=.001, type=float)
    parser.add_argument('--tau', default=.01, type=float)
    parser.add_argument('--cold_start', action='store_true')
    args = parser.parse_args()

    model = main(num_iterations=args.num_iterations, lr=args.lr, tau=args.tau, cold_start=args.cold_start)
    diagnostics.evaluate_euler_error_grid(model, model.env, clip=False)
    renderBestA(model).show()
    render_v_vs_target(model).show()
    renderV(model)
    import pdb
    pdb.set_trace()
