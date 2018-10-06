import numpy as np

from rl.random import OrnsteinUhlenbeckProcess


def calculate_naive_returns(rewards):

    """calculates a list of naive returns given a list of rewards"""
    #total_returns = []
    print(len(rewards))
    print(rewards)
    total_returns=np.zeros(len(rewards))
    total_return = 0.0
    for t in range(len(rewards)):

        total_return = total_return + rewards[t]
        total_returns[t] = total_return

    return total_returns

def discount_rewards(rewards, gamma=0.99):
    discounted_returns = [0 for _ in rewards]
    discounted_returns[-1]=rewards[-1]
    for t in range(len(rewards)-2, -1, -1):
        print(t)
        discounted_returns[t]=rewards[t] + discounted_returns[t+1]*gamma
    return discounted_returns


######## exploration and exploitation strategies ###################################################################

def epsilon_greedy_action(action_distribution, epsilon=1e-1):
    if np.random.random() < epsilon:
        return np.argmax(np.random.random(action_distribution.shape))
    else:
        return np.argmax(action_distribution)

def epsilon_greedy_annealed(action_distribution, percentage, epsilon_start=1.0, epsilon_end=1e-2):
    annealed_epsilon = epsilon_start*(1.0-percentage) + epsilon_end*percentage
    if np.random.random() < annealed_epsilon:
        return np.argmax(np.random.random(action_distribution.shape))
    else:
        return np.argmax(action_distribution)



###################################################################################################################


if __name__ == "__main__":

    ftrt = calculate_naive_returns([2., 3., 5.])
    print(ftrt)
    dsr = discount_rewards([2., 3., 5.])
    print(dsr)