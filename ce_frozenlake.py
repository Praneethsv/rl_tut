import gym

import torch as t

import numpy as np

from collections import namedtuple

from tensorboardX import SummaryWriter
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


GAMMA = 0.9


class DiscreteOneHotWrapper(gym.ObservationWrapper):

    """Main reason for this class is that previous network which solved cartpole using cross entropy accepts a tensor for an input observation space. so
    we are using this wrapper class to create a tensor of rational numbers"""

    def __init__(self, env):

        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)

        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)
        ## the above line makes an n-dimensional tensor of rational numbers with intervals [low = 0.0, high = 1.0]


    def observation(self, observation):

        res = np.copy(self.observation_space.low)  # np.copy returns an array copy of low in Box tensor
        res[observation] = 1.0

        return res

#### Now, with the above class our previous network is 100% compatible. However, if we carefully look at the reward distribution of both enviroments we will
### get to know that it is not solving the forzen lake. Look at the reward distribution in the book. The previous percentile selection of elite episodes
#  is totally wrong because failed episodes will dominate


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

    disc_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []   ## observations that we are gonna use to train the network
    train_act = []   ## actions that are the result of train obs.
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

        return elite_batch, train_obs, train_act, reward_bound





if __name__ == "__main__":

    #env = gym.make('CartPole-v0')

    env = FrozenLakeEnv(is_slippery=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    env = DiscreteOneHotWrapper(env)

    #action = action_distribution()

    #observation, reward, is_done, _ = env.step(action=action)

    observation_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden_nodes = 128

    BATCH_SIZE = 100

    PERCENTILE = 30

    net = NNet(obs_size=observation_size, hidden_size=hidden_nodes, n_actions=n_actions)
    objective = t.nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(params=net.parameters(), lr=0.001)
    writer = SummaryWriter()
    full_batch = []

    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_bound = filter_batch(full_batch+ batch, PERCENTILE)
        if not full_batch:
            continue
        obs_v = t.FloatTensor(obs)
        acts_v = t.LongTensor(acts)
        full_batch = full_batch[-500:]
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v,acts_v)
        loss_v.backward()
        optimizer.step()

        print("%d: loss=%.3f, reward_mean=%.1f, reward_boundary=%.1f" % (iter_no, loss_v.item(), reward_mean, reward_bound))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)

        if reward_mean > 0.8:
            print("Solved")
            break

    writer.close()

