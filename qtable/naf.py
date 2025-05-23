import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import neuralnetwork

class SetupNAF(object):

    @classmethod
    def setup(cls, env, nnvParameters, nnpParameters, nnqParameters, learningParameters):
        indim, actiondim = env.shape()
        nnv = neuralnetwork.NeuralNetwork(indim, 1, **nnvParameters, output_bias=-10)
        nnq = neuralnetwork.NeuralNetwork(indim, actiondim, **nnqParameters)
        if actiondim == 1:
            pdim = 1
        else:
            pdim = (actiondim) * (actiondim + 1) // 2
        nnp = neuralnetwork.NeuralNetwork(indim, pdim, **nnpParameters)
        with torch.no_grad():
            for p in nnp.parameters():
                p.zero_()

        naf = NAFApproximation(nnv, nnp, nnq, actiondim, **learningParameters)
        return naf

def sample_states_and_max_prod(env, n_samples, target_strat='zeros'):
    # generate samples for the cold start
    states_raw = [env.sample_state() for _ in range(n_samples)]
    max_prod = torch.tensor(
        [[env.production.production(**s)] for s in states_raw],
        dtype=torch.float32, device=device
    )
    state_tuples = [(s['k'], s['z']) for s in states_raw]
    state_tensor = torch.tensor(state_tuples, dtype=torch.float32, device=device)
    if target_strat == 'zeros':
        targets = torch.zeros((n_samples, 1), dtype=torch.float32, device=device)
    elif target_strat == 'half':
        targets = torch.tensor([[s['k'] / 2] for s in states_raw],
                               dtype=torch.float32, device=device)
    else:
        targets = torch.tensor(
            [[-10.0 + s['k'] * 0.01] for s in states_raw],
            dtype=torch.float32, device=device
        )

    return state_tensor, max_prod, targets


def coldStart(naf, env, coldstart_len, n_samples=100, learning_params={'compress': True}):
    naf.train()  # Ensure we're in training mode
    success = False
    target_strat = 'half' #'zeros' if learning_params['compress'] else 'half'
    print(target_strat)
    for i in range(coldstart_len):
        state_tensor, max_prod, targets = sample_states_and_max_prod(env, n_samples, target_strat)
        actions = naf.actions(state_tensor, max_prod)
        loss = naf.train_actions_coldstart(state_tensor, max_prod, targets)
        print(i, "Action Loss", loss)
        if i % 10000 == 0 and i > 0:
            print("action", actions[:10].squeeze(1).cpu().numpy())
            print("target", targets[:10].squeeze(1).cpu().numpy())
        # TODO: tune atol
        if torch.allclose(actions, targets, atol=0.005):
            success = True
            break
    if not success:
        state_tensor, max_prod, targets = sample_states_and_max_prod(env, n_samples, target_strat)
        print("WARNING actions did not converge")
        print("action", naf.actions(state_tensor, max_prod)[:5])
        print("target", targets[:5].squeeze(1).cpu().numpy())

    success = False
    for i in range(coldstart_len):
        state_tensor, max_prod, targets = sample_states_and_max_prod(env, n_samples, target_strat='linear')

        loss = naf.train_values_coldstart(state_tensor, targets)
        print(i, "Value Loss", loss)
        values = naf.value(state_tensor)
        if i % 10000 == 0 and i > 0:
            print("values", values[:10].squeeze(1).cpu().numpy())
            print("target", targets[:10].squeeze(1).cpu().numpy())
        # TODO: tune atol
        mse = torch.mean((values - targets) ** 2).item()
        if mse < .01:
            success = True
            break
    if not success:
        values = naf.value(state_tensor)
        print("WARNING values did not converge")
        print("values", values[:10].squeeze(1).cpu().numpy())
        print("target", targets[:10].squeeze(1).cpu().numpy())

    return naf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NAFApproximation(nn.Module):
    def __init__(self, nnv, nnp, nnq, actiondim, learning_rate=1e-3, discount=0.99, compress=False, q_learning_rate=1e-2, v_learning_rate=1e-1):
        super(NAFApproximation, self).__init__()
        self.nnv = nnv.to(device)
        self.nnp = nnp.to(device)
        self.nnq = nnq.to(device)
        self.actiondim = actiondim
        self.compress = compress

        self.beta = discount
        self.gamma = torch.tensor(discount, dtype=torch.float32)

        self.optimizer = torch.optim.Adam(
            list(nnv.parameters()) + list(nnp.parameters()) + list(nnq.parameters()),
            lr=learning_rate
        )

        print("Q learning rate", q_learning_rate, "V learning rate", v_learning_rate)
        self.q_optimizer = torch.optim.Adam(self.nnq.parameters(), lr=q_learning_rate)
        self.v_optimizer = torch.optim.Adam(self.nnv.parameters(), lr=v_learning_rate)


        # TODO: clean up, only useful if we use _to_semi_definite instead of _build_L_matrix
        mask = torch.ones(actiondim, actiondim)
        mask[torch.triu_indices(actiondim, actiondim)] = 0
        self.register_buffer("mask", mask)

        # Create loss functions
        self.mse_loss = nn.MSELoss()

        self.device = device

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

    def action_penalty(self, mu, max_prod):
        mask_high = (mu > 0.8 * max_prod).float()
        mask_low = (mu < 0.2 * max_prod).float()
        penalty = (mask_high * mu + mask_low * (0.2 * max_prod - mu)).sum()
        return penalty * 1e8

    def _calcP(self, state):
        # calculate P using the _to_semi_definite
        tril_elements = self.nnp(state)  # recall a(a+1)/2
        L = self._build_L_matrix(tril_elements)
        return torch.bmm(L, L.transpose(1, 2))

    def check_parameters(self, network_name, network):
        for name, param in network.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm < 1e-6:
                    print(f"[{network_name}, {name}] Gradient too small: {grad_norm:.2e}")
                elif grad_norm > 1e2:
                    print(f"[{network_name}, {name}] Gradient too large: {grad_norm:.2e}")

    def forward(self, state, action, max_prod):
        v = self.nnv(state)

        P = self._calcP(state)

        qout = self.nnq(state)
        # bound the capital allocation to 0 to max_prod
        # TODO: handle multidimenstional mu
        if self.compress:
            import pdb;pdb.set_trace()
            mu = torch.sigmoid(qout) * max_prod
            #mu = ((torch.tanh(qout) + 1.0) / 2.0) * max_prod
        else:
            assert False
            mu = qout

        diff = (action - mu).unsqueeze(1)
        A = torch.bmm(torch.bmm(diff, P), diff.transpose(1, 2)).squeeze(2)
        Q = v - .5 * A

        return Q, v, mu, A, P

    def action(self, x, max_prod):
        self.eval()
        # TODO: inconsistent about when to convert to device, I think I prefer converting in naf
        state_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        max_prod_tensor = torch.tensor(max_prod, dtype=torch.float32, device=device)

        with torch.no_grad():
            mu = self.actions(state_tensor, max_prod_tensor)

        return mu[0].cpu().numpy()


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

        Q, _, mu, _, _ = self.forward(state, action, max_prod)
        loss = self.mse_loss(Q, target) + self.action_penalty(mu, max_prod)

        self.optimizer.zero_grad()
        loss.backward()

        # check parameters
        self.check_parameters("nnv", self.nnv)
        self.check_parameters("nnp", self.nnp)
        self.check_parameters("nnq", self.nnq)

        assert not torch.isnan(mu).any(), "mu exploded"
        #assert not torch.isnan(P).any(), "P exploded"

        self.optimizer.step()

        return loss.item()

    def train_actions_coldstart(self, state, max_prod, target_action):
        self.nnq.train()
        self.q_optimizer.zero_grad()

        qout = self.nnq(state)
        if self.compress:
            mu = ((torch.tanh(qout) + 1.0) / 2.0) * max_prod
        else:
            mu = qout

        loss = self.mse_loss(mu, target_action)
        loss.backward()
        self.q_optimizer.step()

        return loss.item()

    def train_values_coldstart(self, state, target_value):
        self.nnv.train()
        self.v_optimizer.zero_grad()

        v = self.nnv(state)
        loss = self.mse_loss(v, target_value)

        loss.backward()
        self.v_optimizer.step()

        return loss.item()

    def checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def restore(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

    # Diagnostic functions
    def renderBestA(self, include_best=True):
        state = [(x, 0) for x in np.arange(0, 1.0, .01)]
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        max_prod = [(math.pow(s[0], 1.0 / 3.0) * .9792,) for s in state]
        max_prod_tensor = torch.tensor(max_prod, dtype=torch.float32, device=self.device)

        actions = self.actions(state_tensor, max_prod_tensor)
        # when delta=1 there is a closed for solution here (pg 4 https://www.sas.upenn.edu/~jesusfv//comparison_languages.pdf)
        best = [p[0] * (1 - 1.0 / 3.0 * .95) for p in max_prod]

        plt.plot([s[0] for s in state], actions.cpu().numpy(), label="action")
        plt.plot([s[0] for s in state], max_prod, label="max")
        if include_best:
            plt.plot([s[0] for s in state], best, label="best")
        plt.legend()
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
        self.eval()

        # Render Q for capital level .4, productivity 0
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
