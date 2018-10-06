import numpy as np
import torch as t
import ptan
import torch.nn as nn

HID_SIZE = 128


class ModelA2C(nn.Module):
    def __init__(self, input_shape, act_size):

        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(nn.Linear(input_shape, HID_SIZE), nn.ReLU(),)

        self.mu = nn.Sequential(nn.Linear(HID_SIZE, act_size), nn.Tanh(),)

        self.var = nn.Sequential(nn.Linear(HID_SIZE, act_size), nn.Softplus(),)

        self.value = nn.Sequential(nn.Linear(HID_SIZE, 1))

    def forward(self, x):
        base = self.base(x)
        return self.mu(base), self.var(base), self.value(base)


class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__(self, states, agent_states):

        states_v = ptan.agent.float32_preprocessor(states)
        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = t.sqrt(var_v).data.cpu().numpy()
        actions = np.random.uniform(mu, sigma)
        actions = np.clip(actions, -1, 1)

        return actions, agent_states

def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = ptan.agent.float32_preprocessor(states).to(device)
    actions_v = t.FloatTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = ptan.agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = t.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v
