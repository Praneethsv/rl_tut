import torch.nn as nn
import torch as t
import torch.nn.functional as F
import numpy as np
import ptan


class DDPGActor(nn.Module):
    def __init__(self, input_shape, act_size):
        super(DDPGActor, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_shape, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 300),
                                 nn.ReLU(),
                                 nn.Linear(300, act_size),
                                 nn.Tanh()
                                 )

    def forward(self, x):

        return self.net(x)

class DDPGCritic(nn.Module):
    def __init__(self, input_shape, act_size):
        super(DDPGCritic, self).__init__()
        self.obs_net = nn.Sequential(nn.Linear(input_shape, 400),
                                 nn.ReLU(),
                                 )
        self.out_net = nn.Sequential(nn.Linear(400 + act_size, 300),
                                     nn.ReLU(),
                                     nn.Linear(300, 1))

    def forward(self, x, actions):
        obs = self.obs_net(x)
        return self.out_net(t.cat([obs, actions], dim=1))

class AgentDDPG(ptan.agent.BaseAgent):
    def __init__(self, net, device="cpu", ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2,
                 ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = 0.2
        self.ou_epsilon = ou_epsilon

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, actions in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape=actions.shape)
                a_state += self.ou_teta * (self.ou_mu - actions)
                a_state += self.ou_sigma * np.random.normal(size=actions.shape)

                actions += self.ou_epsilon * a_state
                new_a_states.append(a_state)

        else:
            new_a_states = agent_states

        actions = np.clip(actions, -1, 1)
        return actions, new_a_states
