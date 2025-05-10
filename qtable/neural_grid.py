import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.tensor_utils import compute_transition_outcomes
from . import neuralnetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eps = .0001

class NeuralGrid:
    def __init__(self, env, actor, critic, learning_rate=1e-2, tau=.005, beta=.95):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(critic)

        self.env = env
        self.beta = beta

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.mse_loss = nn.MSELoss()
        self.tau = tau

        self._action_in_range_scale = 10
        self._action_in_range_scale_decay = .99

        self.device = device

    def _as_tensor(self, lst):
        return torch.as_tensor(lst, device=device, dtype=torch.float32)

    def action_penalty(self, mu, max_prod):
        lower_violation = torch.clamp(0.05 * max_prod - mu, min=0.0)
        upper_violation = torch.clamp(mu - 0.95 * max_prod, min=0.0)
        penalty = lower_violation.pow(2) + upper_violation.pow(2)
        return self._action_in_range_scale * penalty.mean()

    def values_unnormalized(self, states, target_critic=False):
        critic = self.critic_target if target_critic else self.critic
        return critic(self.env.normalize_tensor(states))

    def actions_unnormalized(self, states, max_prod, clamp=False):

        return self.actions(self.env.normalize_tensor(states), max_prod, clamp)

    def action(self, state, max_prod, clamp=False):
        with torch.no_grad():
            state = self._states_to_tensor(state, from_dict=False)
            max_prod = self._as_tensor(max_prod)
            return self.actions(state, max_prod, clamp)[0].cpu().numpy()

    def actions(self, state, max_prod, clamp=False, normalize=False):
        if normalize:
            state = self._states_to_tensor(state, from_dict=False)
            max_prod = self._as_tensor(max_prod)
        mu = self.actor(state)
        if clamp:
            mu = torch.clamp(mu, min=eps)
            if max_prod.ndim == 1:
                max_prod = max_prod.unsqueeze(1)
            mu = torch.minimum(mu, max_prod - eps)

        return mu

    def _sample(self, batch_size):
        states = [self.env.sample_state() for _ in range(batch_size)]
        max_prods = [self.env.max_prod(s) for s in states]
        return states, max_prods


    def _states_to_tensor(self, lst, from_dict=True):
        if from_dict:
            return self._as_tensor( [self.env.normalize_state_from_dict(s) for s in lst])
        else:
            return self._as_tensor( [self.env.normalize_state(s) for s in lst])

    def q(self, state, action, max_prod):
        c = torch.minimum(action.view(-1), max_prod.view(-1))
        rewards = torch.log(c).unsqueeze(1)

        next_states, probs = compute_transition_outcomes(state, action, max_prod, self.env.motion.transition, self.env)

        B, Z, _ = next_states.shape
        # TODO use env to unnormalize, constants for k and z
        k_next_norm = next_states[..., 0] - 0.5
        z_next_norm = next_states[..., -1] / 4.0 - 0.5
        next_states_norm = torch.stack([k_next_norm, z_next_norm], dim=2)
        next_states_flat = next_states_norm.view(B * Z, 2)
        next_values_flat = self.critic_target(next_states_flat)           # shape [B*Z, 1]
        next_values = next_values_flat.view(B, Z, 1)                      # shape [B, Z, 1]

        # === Step 6: Compute expected value ===
        expected_value = (probs.unsqueeze(2) * next_values).sum(dim=1)   # shape [B, 1]

        # === Step 7: Combine ===
        return rewards + self.beta * expected_value  # shape [B, 1]

    def q_no_tensor(self, initial_states_lst, actions_lst):
        next_states_and_utilities = [self.env.iterate1d(s, a) for s, a in zip(initial_states_lst, actions_lst)]
        next_states_tensor = self._states_to_tensor([x[1] for x in next_states_and_utilities]  )
        next_values_tensor = self.critic_target(next_states_tensor)
        rewards_tensor = self._as_tensor([x[0] for x in next_states_and_utilities]).unsqueeze(1)
        return rewards_tensor + self.beta * next_values_tensor

    def value(self, state_tensor):
        with torch.no_grad():
            return self.critic(state_tensor)

    def value_step(self, batch_size=1000):
        states = [self.env.sample_state() for _ in range(batch_size)]
        max_prods_tensor = self._as_tensor([self.env.max_prod(s) for s in states])
        normalized_states_tensor = self._states_to_tensor(states)

        with torch.no_grad():
            actions = self.actions(normalized_states_tensor, max_prods_tensor)
            q_tensor = self.q(normalized_states_tensor, actions, max_prods_tensor)
        value_tensor = self.critic(normalized_states_tensor)

        self.critic_optimizer.zero_grad()
        loss = self.mse_loss(q_tensor, value_tensor)
        loss.backward()
        self.critic_optimizer.step()

        neuralnetwork.soft_update(self.critic_target, self.critic, self.tau)
        return loss.item()

    def actor_step(self, batch_size=1000):
        self._action_in_range_scale *= self._action_in_range_scale_decay
        states = [self.env.sample_state() for _ in range(batch_size)]
        max_prods_tensor = self._as_tensor([self.env.max_prod(s) for s in states])
        normalized_states_tensor = self._states_to_tensor(states)
        actions = self.actions(normalized_states_tensor, max_prods_tensor, clamp=False)
        values = self.q(normalized_states_tensor, actions, max_prods_tensor)

        self.actor_optimizer.zero_grad()
        loss = -values.mean() + self.action_penalty(actions, max_prods_tensor)
        loss.backward()
        self.actor_optimizer.step()
        return loss.item()

    def train_actions_cold_start(self, batch_size=1000, n_iters=100):
        print("Training actions coldstart")
        for i in range(n_iters):
            states = [self.env.sample_state() for _ in range(batch_size)]
            max_prods_tensor = self._as_tensor([self.env.max_prod(s) for s in states])
            normalized_states_tensor = self._states_to_tensor(states)
            actions = self.actions(normalized_states_tensor, max_prods_tensor, clamp=False)
            target_actions = max_prods_tensor / 2

            self.actor_optimizer.zero_grad()
            loss = self.mse_loss(actions.view(-1), target_actions.view(-1))
            loss.backward()
            if i % 25 == 0:
                print(i, loss.item())
                print(f"[{i}] predicted mean: {actions.mean().item():.4f}, target mean: {target_actions.mean().item():.4f}, loss: {loss.item():.4f}")
            self.actor_optimizer.step()

    def train_values_coldstart(self, n_iters=250, batch_size=1000, print_every=250):
        tau_backup = self.tau
        self.tau = 1.0
        for step in range(0, n_iters):
            value_loss = self.value_step(batch_size=batch_size)

            if step % print_every == 0 or step == 1:
                print(f"[Critic Pretraining Step {step}] value loss = {value_loss:.6f}")
        self.critic_target = copy.deepcopy(self.critic)
        self.tau = tau_backup

    # trainstep for RL with an agent
    def trainstep(self, state, action, max_prod, reward, next_state):
        state_tensor = self._states_to_tensor(state, from_dict=False)
        action_tensor = self._as_tensor(action)
        max_prod_tensor = self._as_tensor(max_prod)
        reward_tensor = self._as_tensor(reward)
        next_state_tensor = self._states_to_tensor(next_state, from_dict=False)

        # critic update
        with torch.no_grad():
            next_actions = self.actions(next_state_tensor, max_prod_tensor, clamp=True)
            target_q = reward_tensor + self.beta * self.q(next_state_tensor, next_actions, max_prod_tensor)
        current_q = self.q(state_tensor, action_tensor, max_prod_tensor)
        critic_loss = self.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor update
        actions = self.actions(state_tensor, max_prod_tensor, clamp=False)
        actor_loss = -self.q(state_tensor, actions, max_prod_tensor).mean()
        actor_loss += self.action_penalty(actions, max_prod_tensor)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target network
        neuralnetwork.soft_update(self.critic_target, self.critic, self.tau)
