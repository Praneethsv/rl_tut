import gym
import torch as t
import numpy as np
from tensorboardX import SummaryWriter
from collections import namedtuple


class NNet(t.nn.Module):

    def __init__(self, obs_size, hidden_size, n_actions):

        super(NNet, self).__init__()
        self.net = t.nn.Sequential(t.nn.Linear(obs_size, hidden_size),
                                   t.nn.ReLU(), t.nn.Linear(hidden_size, n_actions))

    def forward(self, x):

        return self.net(x)

Episode = namedtuple('Episode', field_names= ['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names= ['observation', 'action'])


def iterate_batches(env, net, batch_size):

    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = t.nn.Softmax(dim=1)

    while True:

        obs_v = t.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)

        episode_reward += reward
        episode_steps.append(EpisodeStep(observation = obs, action = action))

        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward=0.0
            episode_steps=[]
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch=[]

        obs = next_obs

def filter_batch(batch, percentile):   ### core to cross entropy  ###

    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []   ## observations that we are gonna use to train the network
    train_act = []   ## actions that are the result of train obs.
    for example in batch:
        if example.reward < reward_bound:
            continue

        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

        train_obs_v = t.FloatTensor(train_obs)
        train_act_v = t.LongTensor(train_act)

        return train_obs_v, train_act_v, reward_bound, reward_mean



if __name__ == "__main__":

    env = gym.make('CartPole-v0')

    #action = action_distribution()

    #observation, reward, is_done, _ = env.step(action=action)

    observation_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden_nodes = 128

    BATCH_SIZE = 32

    PERCENTILE = 70

    net = NNet(obs_size=observation_size, hidden_size=hidden_nodes, n_actions=n_actions)
    objective = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter()

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v,acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_boundary=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)

        if reward_m > 199:
            print("Solved")
            break

    writer.close()





