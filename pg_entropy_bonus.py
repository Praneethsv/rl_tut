import gym
import torch as t
import collections
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from ptan.experience import ExperienceSource
import torch.optim as optim
import ptan

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 8

REWARD_STEPS = 10

class PGN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PGN, self).__init__()
        self.input_shape = input_shape
        self.n_hidden = 128
        self.n_actions = n_actions

        self.net = nn.Sequential(nn.Linear(self.input_shape, self.n_hidden),
                                 nn.ReLU(),
                                 nn.Linear(self.n_hidden, self.n_actions)
                                 )

    def forward(self, x):
        return self.net(x)

def calc_qvals(rewards):

    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    steps_rewards = []
    step_idx = 0
    done_episodes = 0


    batch_states, batch_actions, batch_scales = [], [], []
    cur_rewards = []
    reward_sum = 0.0
    for step_idx, exp in enumerate(exp_source):
        reward_sum += exp.reward
        base_line = reward_sum / (step_idx +1)
        writer.add_scalar("baseline", base_line, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - base_line)

        # handle new rewards
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
                step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        states_v = t.FloatTensor(batch_states)
        batch_actions_t = t.LongTensor(batch_actions)
        #batch_qvals_v = t.FloatTensor(batch_qvals)
        batch_scales_v = t.FloatTensor(batch_scales)

        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scales_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_bonus = - ENTROPY_BETA * entropy_v
        loss_v = loss_policy_v + entropy_bonus

        loss_v.backward()
        optimizer.step()


        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()
