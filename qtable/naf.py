import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from . import neuralnetwork


class SetupNAF(object):

    @classmethod
    def setup(cls, env, nnvParameters, nnpParameters, nnqParameters, learningParameters):
        indim, actiondim = env.shape()
        nnv = neuralnetwork.NeuralNetwork(indim, 1, **nnvParameters)
        nnq = neuralnetwork.NeuralNetwork(
            indim, actiondim, **nnqParameters)
        if actiondim == 1:
            pdim = 1
        else:
            pdim = (actiondim) * (actiondim + 1) // 2
        nnp = neuralnetwork.NeuralNetwork(indim, pdim, **nnpParameters)
        naf = NAFApproximation(nnv, nnp, nnq, actiondim, **learningParameters)
        return naf


def coldStart(naf, env, coldstart_len, n_samples=100, learning_params={'compress': True}):
    # initialize NAF so actions are in range
    success = False
    states = [env.sample_state() for _ in range(n_samples)]
    max_prod = [(env.production.production(**state),) for state in states]
    if learning_params['compress']:
        targets = [(0, ) for state in states]
    else:
        targets = [(state['k'] / 2, ) for state in states]
    states = [(state['k'], state['z']) for state in states]
    for i in range(coldstart_len):
        actions = naf.actions(states, max_prod)
        naf.train_actions_coldstart(states, max_prod, targets)
        if i % 10000 == 0:
            print("action", [a[0] for a in actions[0:10]])
            print("target", [t[0] for t in targets[0:10]])
        if np.allclose(actions, targets, atol=.2):
            success = True
            break
    if not success:
        print("WARNING actions did not converge")
        print(naf.actions(states, max_prod)[0:5])
    success = False
    for i in range(coldstart_len):
        states = [env.sample_state() for _ in range(n_samples)]
        targets = [(-10.0 + state['k'] * .01, ) for state in states]
        states = [(state['k'], state['z']) for state in states]
        naf.train_values_coldstart(states, targets)
        values = naf.value(states)
        if i % 10000 == 0:
            print("action", values[0:10])
            print("target", targets[0:10])
        if np.allclose(values, targets, atol=.2):
            success = True
            break
    if not success:
        print("WARNING values did not converge")
    return naf


class NAFApproximation(nn.Module):
    def __init__(self, nnv, nnp, nnq, actiondim, learning_rate=1e-3, discount=0.99, compress=False):
        super(NAFApproximation, self).__init__()
        self.nnv = nnv
        self.nnp = nnp
        self.nnq = nnq
        self.actiondim = actiondim
        self.compress = compress

        self.beta = discount
        self.gamma = torch.tensor(discount, dtype=torch.float32)

        # Create optimizer (params from all subnets)
        self.optimizer = torch.optim.Adam(
            list(nnv.parameters()) + list(nnp.parameters()) + list(nnq.parameters()),
            lr=learning_rate
        )

        # Precompute lower-triangular mask for L
        mask = torch.ones(actiondim, actiondim)
        mask[torch.triu_indices(actiondim, actiondim)] = 0
        self.register_buffer("mask", mask)

        # Create loss functions
        self.mse_loss = nn.MSELoss()

    def _build_L_matrix(self, tril_elements):
        # builds from minimal representation (a(a+1)/2)
        batch_size = tril_elements.size(0)
        L = torch.zeros((batch_size, self.actiondim, self.actiondim), device=tril_elements.device)
        tril_indices = torch.tril_indices(row=self.actiondim, col=self.actiondim, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = tril_elements

        # Ensure positive diagonals using exp
        diag_idx = torch.arange(self.actiondim)
        L[:, diag_idx, diag_idx] = torch.exp(L[:, diag_idx, diag_idx])

        return L

    def _to_semi_definite_pytorch(self, M):
        # M: (batch, actiondim, actiondim)
        diag = torch.sqrt(torch.exp(torch.diagonal(M, dim1=1, dim2=2)))
        L = M * self.mask
        L = L.clone()
        diag_idx = torch.arange(self.actiondim)
        L[:, diag_idx, diag_idx] = diag
        return torch.bmm(L, L.transpose(1, 2))

    def coldstart_action_loss(self, state, max_prod, target_action):
        _, _, mu, _, _ = self.forward(state, target_action, max_prod)
        return self.mse_loss(mu, target_action)

    def coldstart_value_loss(self, env, state, target_value):
        v = self.nnv(state)
        return self.mse_loss(v, target_value)

    def train_step(self, state, action, max_prod, reward, next_state):
        with torch.no_grad():
            v_next = self.nnv(next_state)
            target = reward + self.gamma * v_next

        Q, _, _, _, _ = self.forward(state, action, max_prod)
        loss = self.mse_loss(Q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _calcP(self, state):
        tril_elements = self.nnp(state)  # recall a(a+1)/2
        L = self._build_L_matrix(tril_elements)
        return torch.bmm(L, L.transpose(1, 2))

    def forward(self, state, action, max_prod):
        v = self.nnv(state)

        P = self._calcP(state)

        qout = self.nnq(state)
        if self.compress:
            mu = ((torch.tanh(qout) + 1.0) / 2.0) * max_prod
        else:
            mu = qout

        diff = (action - mu).unsqueeze(1)
        A = torch.bmm(torch.bmm(diff, P), diff.transpose(1, 2)).squeeze(2)  # (batch, 1)

        Q = v + .5 * A

        return Q, v, mu, A, P

    def action(self, x, max_prod):
        self.eval()
        state_tensor = torch.tensor(x, dtype=torch.float32)
        max_prod_tensor = torch.tensor(max_prod, dtype=torch.float32)

        with torch.no_grad():
            mu = self.actions(state_tensor, max_prod_tensor)

        return mu[0].numpy()


    def actions(self, state, max_prod):
        self.eval()
        with torch.no_grad():
            _, _, mu, _, _ = self.forward(state, mu := torch.zeros_like(max_prod), max_prod)
        return mu

    def calcq(self, rewards, next_state):
        self.nnv.eval()
        with torch.no_grad():
            v_next = self.nnv(next_state)
            return rewards + self.gamma * v_next

    def calcA(self, state, action, max_prod):
        self.eval()
        with torch.no_grad():
            _, _, _, A, _ = self.forward(state, action, max_prod)
            return A

    def value(self, state):
        self.nnv.eval()
        with torch.no_grad():
            return self.nnv(state)

    def trainstep(self, state, action, max_prod, rewards, next_state):
        self.train()

        target = self.calcq(rewards, next_state)

        Q, _, _, _, _ = self.forward(state, action, max_prod)
        loss = self.mse_loss(Q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_actions_coldstart(self, state, max_prod, target_action):
        self.nnq.train()
        self.optimizer.zero_grad()

        qout = self.nnq(state)
        if self.compress:
            mu = ((torch.tanh(qout) + 1.0) / 2.0) * max_prod
        else:
            mu = qout

        loss = self.mse_loss(mu, target_action)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_values_coldstart(self, state, target_value):
        self.nnv.train()
        self.optimizer.zero_grad()

        v = self.nnv(state)
        loss = self.mse_loss(v, target_value)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def restore(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

    def renderBestA(self, include_best=True):
        x = [(x, 0) for x in np.arange(0, 1.0, .01)]
        max_prod = [(math.pow(s[0], 1.0 / 3.0) * .9792,) for s in x]

        actions = self.actions(x, max_prod)
        best = [p[0] * (1 - 1.0 / 3.0 * .95) for p in max_prod]

        plt.plot([s[0] for s in x], actions, label="action")
        plt.plot([s[0] for s in x], max_prod, label="max")
        if include_best:
            plt.plot([s[0] for s in x], best, label="best")
        plt.title("Best Actions")
        return plt

    def renderA(self):
        states = [(x, 0) for x in np.arange(.1, 1.1, .25)]
        for state in states:
            mprod = math.pow(state[0], 1.0 / 3.0) * .9792
            actions = np.expand_dims(np.arange(0, mprod, mprod / 10.0), axis=1)
            max_prod = ([(mprod,) for _ in actions])
            value = self.calcA([state for a in actions], actions, max_prod)

            plt.plot(actions, value, label=str(state))
        plt.legend()
        plt.waitforbuttonpress(0)
        plt.close()

    def renderQ(self):
        self.eval()  # Set to evaluation mode

        # Define fixed state
        state = (0.4, 0)
        mprod = math.pow(state[0], 1.0 / 3.0) * 0.9792

        # Generate candidate actions
        actions = np.arange(0.001, mprod, mprod / 10.0).reshape(-1, 1)
        xx = actions[:, 0].tolist()
        max_prod = np.full((len(actions), 1), mprod)
        utility = np.log(actions)

        # Torch tensors
        state_batch = torch.tensor([state] * len(actions), dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.float32)
        max_prod_tensor = torch.tensor(max_prod, dtype=torch.float32)
        reward_tensor = torch.tensor(utility, dtype=torch.float32)
        next_states = torch.tensor([(mprod - a[0], 0) for a in actions], dtype=torch.float32)

        # Compute values
        stored_q = self.forward(state_batch, action_tensor, max_prod_tensor)[0]
        advantage = self.calcA(state_batch, action_tensor, max_prod_tensor)
        future_value = self.value(next_states)
        future_reward = reward_tensor + self.gamma * future_value

        # Plot
        plt.plot(xx, stored_q.numpy(), label="value + action reward")
        plt.plot(xx, (-0.5 * advantage).numpy(), label="a reward")
        plt.plot(xx, future_reward.numpy(), label="utility + next_value")
        plt.plot(xx, future_value.numpy(), label="next value")

        plt.legend()
        plt.waitforbuttonpress(0)
        plt.close()

    def renderV(self):
        self.eval()

        x_vals = np.arange(0, 1.0, 0.01)
        states = torch.tensor([[x, 0] for x in x_vals], dtype=torch.float32)
        v_vals = self.value(states).squeeze(1).numpy()

        plt.plot(x_vals, v_vals)
        plt.title("v over range")
        plt.waitforbuttonpress(0)
        plt.close()
