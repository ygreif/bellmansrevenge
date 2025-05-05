import math
import copy

import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        targets = torch.tensor([[s['k'] * .75] for s in states_raw],
                               dtype=torch.float32, device=device)
    else:
        targets = torch.tensor(
            [[-10.0 + (s['k'] - 0.5) * 2] for s in states_raw],
            dtype=torch.float32, device=device
)
    return state_tuples, max_prod, targets

def coldstart_value_with_policy(qtable, env, gamma=0.95, n_steps=40, n_samples=100):
    """
    Cold-start the value network by estimating true discounted returns under
    the initial (greedy) policy and training nnv to match them.

    Args:
        qtable: the QTABLEApproximation instance
        env: the environment (must support iterate() and sample_state())
        gamma: discount factor (same as qtable.beta)
        n_steps: number of rollout steps to simulate per state
        n_samples: number of start states to simulate from
    """
    print("Coldstarting V(s) using fixed policy simulation...")
    qtable.eval()  # Don't train μ or P during this phase

    states = [env.sample_state() for _ in range(n_samples)]
    discounted_returns = []

    for s in states:
        total, discount = 0.0, 1.0
        state = s
        for _ in range(n_steps):
            max_prod = env.production.production(**state)
            c = qtable.action([tuple(state.values())], [(max_prod,)])[0]
            if c <= .01:
                c = .01
            elif c >= max_prod:
                c = max_prod - .01
            action = {'c': c}
            try:
                utility, next_state = env.iterate(state, action)
            except Exception as e:
                print(e, c, max_prod)
                continue
            total += discount * utility
            discount *= gamma
            state = next_state
        discounted_returns.append(total)

    # Prepare tensors
    state_tensor = torch.tensor(
        [tuple((s['k'], s['z'])) for s in states],
        dtype=torch.float32, device=qtable.device
    )
    returns_tensor = torch.tensor(
        discounted_returns,
        dtype=torch.float32, device=qtable.device
    ).unsqueeze(1)

    # Train only nnv to match these returns
    for i in range(1000):
        qtable.v_optimizer.zero_grad()
        v_pred = qtable.nnv(state_tensor)
        loss = F.mse_loss(v_pred, returns_tensor)
        loss.backward()
        qtable.v_optimizer.step()
        if i % 100 == 0:
            print(f"[coldstart V] iter={i}, loss={loss.item():.4f}")

    # Sync target V
    qtable.target_nnv = copy.deepcopy(qtable.nnv)
    print("Finished coldstarting V. Sample V[0]:", v_pred[0].item(), "Target:", returns_tensor[0].item())

def coldStart(qtable, env, coldstart_len, n_samples=100, learning_params={'compress': False}):
    qtable.train()  # Ensure we're in training mode
    success = False
    target_strat = 'half' #'zeros' if learning_params['compress'] else 'half'
    print(target_strat)
    for i in range(coldstart_len):
        state_tuples, max_prod, targets = sample_states_and_max_prod(env, n_samples, target_strat)
        actions = qtable.actions(state_tuples, max_prod)
        mse = torch.mean((actions - targets) ** 2).item()
        if mse < .01 and i > 50:
            success = True
            break
        loss = qtable.train_actions_coldstart(state_tuples, max_prod, targets)
        #print(i, "Action Loss", loss)
        if i % 10000 == 0 and i > 0:
            print("action", actions[:10].squeeze(1).cpu().numpy())
            print("target", targets[:10].squeeze(1).cpu().numpy())
    if not success:
        state_tensor, max_prod, targets = sample_states_and_max_prod(env, n_samples, target_strat)
        print("WARNING actions did not converge")
        actions = qtable.actions(state_tensor, max_prod)[:5]
        targets = targets[:5].squeeze(1)
        print("action",actions)
        print("target", targets.cpu().numpy())
        print("mse", torch.mean((actions - targets) ** 2).item())

    if qtable.shared:
        for param in qtable.shared.parameters():
            param.requires_grad = False

    success = False
    for i in range(coldstart_len):
        state_tuples, max_prod, targets = sample_states_and_max_prod(env, n_samples, target_strat='linear')

        loss = qtable.train_values_coldstart(state_tuples, targets)
        #print(i, "Value Loss", loss)
        values = qtable.value(state_tuples)
        if i % 10000 == 0 and i > 0:
            print("values", values[:10].squeeze(1).cpu().numpy())
            print("target", targets[:10].squeeze(1).cpu().numpy())
        # TODO: tune atol
        mse = torch.mean((values - targets) ** 2).item()
        if mse < .05:
            success = True
            break
    if not success:
        state_tuples, max_prod, targets = sample_states_and_max_prod(env, n_samples, target_strat='linear')
        values = qtable.value(state_tuples)
        print("WARNING values did not converge")
        print("values", values[:10].squeeze(1).cpu().detach().numpy())
        print("target", targets[:10].squeeze(1).cpu().detach().numpy())

    #coldstart_value_with_policy(qtable, env, gamma=0.95, n_steps=40, n_samples=100)
    qtable.target_nnv = copy.deepcopy(qtable.nnv)
    if qtable.shared:
        print("Turnign shared back on")
        for param in qtable.shared.parameters():
            param.requires_grad = True
    return qtable

def renderBestA(q, include_best=True):
    state = [(x, 0) for x in np.arange(0, 1.0, .01)]
    state_tensor = torch.tensor(state, dtype=torch.float32, device=q.device)
    max_prod = [(math.pow(s[0], 1.0 / 3.0) * .9792,) for s in state]
    max_prod_tensor = torch.tensor(max_prod, dtype=torch.float32, device=q.device)

    actions = q.actions(state_tensor, max_prod_tensor)
    # when delta=1 there is a closed for solution here (pg 4 https://www.sas.upenn.edu/~jesusfv//comparison_languages.pdf)
    best = [p[0] * (1 - 1.0 / 3.0 * .95) for p in max_prod]

    plt.plot([s[0] for s in state], actions.cpu().numpy(), label="action")
    plt.plot([s[0] for s in state], max_prod, label="max")
    if include_best:
        plt.plot([s[0] for s in state], best, label="best")
    plt.legend()
    plt.title("Best Actions")
    return plt

def renderA(q):
    q.eval()

    states = [(x, 0) for x in np.arange(.1, 1.1, .25)]

    for state in states:
        mprod = math.pow(state[0], 1.0 / 3.0) * 0.9792
        actions = np.arange(0, mprod, mprod / 10.0).reshape(-1, 1)
        state_list = [state for _ in actions]
        max_prod_list = [(mprod,) for _ in actions]

        state_tensor = torch.tensor(state_list, dtype=torch.float32, device=q.device)
        action_tensor = torch.tensor(actions, dtype=torch.float32, device=q.device)
        max_prod_tensor = torch.tensor(max_prod_list, dtype=torch.float32, device=q.device)

        value = q.calcA(state_tensor, action_tensor, max_prod_tensor).cpu()
        mu = q.actions([state], [(mprod,)])[0].item()
        plt.axvline(mu, color='black', linestyle='--', label=f"μ={mu:.2f}")
        plt.plot(actions[:, 0], value.numpy(), label=str(state))

    plt.legend()
    plt.title("Advantage Function A(s,a)")
    return plt

def renderQ(q):
    q.eval()

    # Render Q for capital level .4, productivity 0
    state = (0.4, 0)
    mprod = math.pow(state[0], 1.0 / 3.0) * 0.9792

    # Generate candidate actions
    actions = np.arange(0.001, mprod, mprod / 10.0).reshape(-1, 1)
    xx = actions[:, 0].tolist()
    max_prod = np.full((len(actions), 1), mprod)
    utility = np.log(actions)

    # Torch tensors
    state_batch = torch.tensor([state] * len(actions), dtype=torch.float32, device=device)
    action_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
    max_prod_tensor = torch.tensor(max_prod, dtype=torch.float32, device=device)
    reward_tensor = torch.tensor(utility, dtype=torch.float32, device=device)
    next_states = torch.tensor([(mprod - a[0], 0) for a in actions], dtype=torch.float32, device=device)

    # Compute values
    stored_q = q.forward(state_batch, action_tensor, max_prod_tensor)[0].cpu()
    advantage = q.calcA(state_batch, action_tensor, max_prod_tensor).cpu()
    future_value = q.value(next_states).cpu()
    future_reward = reward_tensor.cpu() + q.gamma.cpu() * future_value

    # Plot
    plt.plot(xx, stored_q.detach().numpy(), label="value + action reward")
    plt.plot(xx, (-0.5 * advantage).numpy(), label="a reward")
    plt.plot(xx, future_reward.numpy(), label="utility + next_value")
    plt.plot(xx, future_value.numpy(), label="next value")
    plt.legend()
    return plt

def renderV(q):
    x_vals = np.arange(0, 1.0, 0.01)
    states = torch.tensor([[x, 0] for x in x_vals], dtype=torch.float32)

    v_vals = q.values_unnormalized(states).squeeze(1).detach().numpy()

    plt.plot(x_vals, v_vals)
    plt.title("v over range")
    plt.waitforbuttonpress(0)
    plt.close()

def render_v_vs_target(q):
    x_vals = np.arange(0.0, 1.0, 0.01)
    state_tuples = [(x, 0.0) for x in x_vals]
    states = torch.as_tensor(state_tuples, device=q.device, dtype=torch.float32)

    v_online = q.values_unnormalized(states, False).detach().cpu().numpy().squeeze()
    v_target = q.values_unnormalized(states, True).detach().cpu().numpy().squeeze()

    plt.plot(x_vals, v_online, label='V (main)')
    plt.plot(x_vals, v_target, label='V (target)', linestyle='--')
    plt.legend()
    plt.title("Main vs Target Value Network")
    return plt

def renderBestA(q, include_best=True):
    state = [(x, 0) for x in np.arange(0, 1.0, .01)]
    state_tensor = torch.tensor(state, dtype=torch.float32, device=q.device)
    max_prod = [(math.pow(s[0], 1.0 / 3.0) * .9792,) for s in state]
    max_prod_tensor = torch.tensor(max_prod, dtype=torch.float32, device=q.device)

    actions = q.actions_unnormalized(state_tensor, max_prod_tensor, clamp=False)
    # when delta=1 there is a closed for solution here (pg 4 https://www.sas.upenn.edu/~jesusfv//comparison_languages.pdf)
    best = [p[0] * (1 - 1.0 / 3.0 * .95) for p in max_prod]

    plt.plot([s[0] for s in state], actions.cpu().detach().numpy(), label="action")
    plt.plot([s[0] for s in state], max_prod, label="max")
    if include_best:
        plt.plot([s[0] for s in state], best, label="best")
    plt.legend()
    plt.title("Best Actions")
    return plt
