import gym
from tensorboardX import SummaryWriter
import numpy as np

import collections

GAMMA = 0.95


class Agent:

    def __init__(self):
        self.env = gym.make("FrozenLake-v0")
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)  ## declaring rewards as a dictionary
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):  ### for exploration during training or approxiamation of value

        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action=action)

            self.rewards[(self.state, action, new_state)] = reward    #### reward dictionary for first iter ex: (0, 0, 0) : 0
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_values(self, state, action):

        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count/total) * (reward + GAMMA * self.values[(tgt_state)])

        return action_value


    def select_action(self, state):

        best_value, best_action = None, None

        for action in range(self.env.action_space.n):

            action_value = self.calc_action_values(state, action)  ### here you are approximating the best value and then corresponding action is treated as best action
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action

        return best_action

    def play_episode(self, env):  ### this is to play for one episode for testing

        total_reward = 0.0
        state = env.reset()

        while True:

            action = self.select_action(state)

            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state

        return total_reward


    def value_iteration(self):

        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_values(state, action) for action in range(self.env.action_space.n)]  #### calculating Vs for all possible actions from current state
            self.values[state] = max(state_values)



if __name__ == "__main__":

    test_env = gym.make("FrozenLake-v0")

    agent = Agent()

    writer = SummaryWriter()
    print(type(writer))
    TEST_EPISODES = 20
    iter_no = 0.0
    best_reward = 0.0

    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()
        reward = 0.0

        for i in range(TEST_EPISODES):

            reward += agent.play_episode(test_env)

        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, iter_no)

        if reward > best_reward:

            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))

            best_reward = reward

        if reward > 0.80:

            print("Solved in %d iterations!" % iter_no)

            break

    writer.close()



