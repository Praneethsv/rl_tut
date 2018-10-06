import  gym
import tensorboardX
import torch as t
import numpy as np
from collections import namedtuple

class Net(t.nn.Module):

    def __init__(self, obs_size, hidden_size, n_actions):

        super(Net, self).__init__()
        self.net = t.nn.Sequential(t.nn.Linear(obs_size, hidden_size),
                                   t.nn.ReLU(), t.nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.net(x)

Episode  = namedtuple("Episode", field_names= ['reward', 'steps'])

# class Episode(tuple):
#     'Episode(reward, steps)'
#
#     __slots__ = ()
#
#     _fields = ('reward', 'steps')
#
#     def __new__(_cls, reward, steps):
#         'Create new instance of Episode(reward, steps)'
#         return _tuple.__new__(_cls, (reward, steps))
#
#     @classmethod
#     def _make(cls, iterable, new=tuple.__new__, len=len):
#         'Make a new Episode object from a sequence or iterable'
#         result = new(cls, iterable)
#         if len(result) != 2:
#             raise TypeError('Expected 2 arguments, got %d' % len(result))
#         return result
#
#     def _replace(_self, **kwds):
#         'Return a new Episode object replacing specified fields with new values'
#         result = _self._make(map(kwds.pop, ('reward', 'steps'), _self))
#         if kwds:
#             raise ValueError('Got unexpected field names: %r' % list(kwds))
#         return result
#
#     def __repr__(self):
#         'Return a nicely formatted representation string'
#         return self.__class__.__name__ + '(reward=%r, steps=%r)' % self
#
#     def _asdict(self):
#         'Return a new OrderedDict which maps field names to their values.'
#         return OrderedDict(zip(self._fields, self))
#
#     def __getnewargs__(self):
#         'Return self as a plain tuple.  Used by copy and pickle.'
#         return tuple(self)
#
#     reward = _property(_itemgetter(0), doc='Alias for field number 0')
#
#     steps = _property(_itemgetter(1), doc='Alias for field number 1')



EpisodeStep = namedtuple("EpisodeStep", field_names= ['observation', 'action'])


# class EpisodeStep(tuple):
#     'EpisodeStep(observation, action)'
#
#     __slots__ = ()
#
#     _fields = ('observation', 'action')
#
#     def __new__(_cls, observation, action):
#         'Create new instance of EpisodeStep(observation, action)'
#         return _tuple.__new__(_cls, (observation, action))
#
#     @classmethod
#     def _make(cls, iterable, new=tuple.__new__, len=len):
#         'Make a new EpisodeStep object from a sequence or iterable'
#         result = new(cls, iterable)
#         if len(result) != 2:
#             raise TypeError('Expected 2 arguments, got %d' % len(result))
#         return result
#
#     def _replace(_self, **kwds):
#         'Return a new EpisodeStep object replacing specified fields with new values'
#         result = _self._make(map(kwds.pop, ('observation', 'action'), _self))
#         if kwds:
#             raise ValueError('Got unexpected field names: %r' % list(kwds))
#         return result
#
#     def __repr__(self):
#         'Return a nicely formatted representation string'
#         return self.__class__.__name__ + '(observation=%r, action=%r)' % self
#
#     def _asdict(self):
#         'Return a new OrderedDict which maps field names to their values.'
#         return OrderedDict(zip(self._fields, self))
#
#     def __getnewargs__(self):
#         'Return self as a plain tuple.  Used by copy and pickle.'
#         return tuple(self)
#
#     observation = _property(_itemgetter(0), doc='Alias for field number 0')
#
#     action = _property(_itemgetter(1), doc='Alias for field number 1')



def iterate_batches(env, net, batch_size):

    batch = []
    episode_steps = []
    episode_reward = 0.0
    obs = env.reset()
    sm = t.nn.Softmax(dim=1)

    while True:

        obs_v = t.FloatTensor([obs])
        act_probs_v = t.FloatTensor(sm(net(obs_v)))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p= act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if is_done:
            batch.append(Episode(reward= episode_reward, steps = episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()

            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs



def crossentropy_core(batch, percentile):

    train_obs = []
    train_actions = []







if __name__ == "__main__":

    env = gym.make("CartPole-v0")

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
