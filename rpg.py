import torch as t
import numpy as np
from tensorboardX import SummaryWriter
import gym
import torch.nn.functional as F
import ptan
import torch.optim as optim

lr = 0.01
gamma = 0.99
episodes_to_train = 4

class PGN(t.nn.Module):
    def __init__(self, input_size, n_act):
        super(PGN, self).__init__()

        self.net = t.nn.Sequential(t.nn.Linear(input_size, 128),
                                   t.nn.ReLU(),
                                   t.nn.Linear(128, n_act)
            )

    def forward(self, x):
        return self.net(x)


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= gamma
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="cp-rein-revision")
    net = PGN(env.observation_space.shape[0], env.action_space.n)
    #print(net)
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
    
    #print("Returned from agent is here :")
    #print(agent.__call__())
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma)
    #print(exp_source)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    total_rewards = []
    step_idx = 0
    done_episodes = 0

    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []

    for step_idx, exp in enumerate(exp_source):
        #print(step_idx, exp)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))

        cur_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

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

        if batch_episodes < episodes_to_train:
            continue

        optimizer.zero_grad()
        states_v = t.FloatTensor(batch_states)
        batch_actions_t = t.LongTensor(batch_actions)
        batch_qvals_v = t.FloatTensor(batch_qvals)

        logits_v = net(states_v)
        # print(logits_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        #print(log_prob_v)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()
