import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import epsilon_greedy_action, epsilon_greedy_annealed, discount_rewards
import gym
import tqdm

class PGradient(object):

    def __init__(self, session, state_size, num_actions,
                 hidden_size, learning_rate, discount_factor,
                 exploration_exploitation_setting):

        self.session = session
        self.state_size = state_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_exploitaion_setting = exploration_exploitation_setting

        self.build_model()
        self.build_train()


    def build_model(self):

        with tf.variable_scope('pg-model'):
            self.state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)

            self.h0 = slim.fully_connected(self.state, self.hidden_size)
            self.h1 = slim.fully_connected(self.h0, self.hidden_size)
            self.output = slim.fully_connected(self.h1, self.num_actions, activation_fn=tf.nn.softmax)

    def build_train(self):

        self.action_input = tf.placeholder(tf.int32, shape=[None])
        self.reward_input = tf.placeholder(tf.float32, shape=[None])
        self.output_index_for_actions = (tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1]) + self.action_input
        self.logits_for_actions = tf.gather(tf.reshape(self.output, [-1]), self.output_index_for_actions)

        self.loss = -tf.reduce_mean(tf.log(self.logits_for_actions) * self.reward_input)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_step = self.optimizer.minimize(self.loss)

    def sample_action_from_distribution(self, action_distribution, epsilon_percentage ):
        if self.exploration_exploitaion_setting == 'greedy':
            action = epsilon_greedy_action(action_distribution)

        elif self.exploration_exploitaion_setting == 'epsilon_greedy_0.05':
            action = epsilon_greedy_action(action_distribution, epsilon=0.05)
        elif self.exploration_exploitaion_setting == 'epsilon_greedy_0.25':
            action = epsilon_greedy_action(action_distribution, epsilon=0.25)

        elif self.exploration_exploitaion_setting == 'epsilon_greedy_0.50':
            action = epsilon_greedy_action(action_distribution, epsilon=0.50)
        elif self.exploration_exploitaion_setting == 'epsilon_greedy_0.90':
            action = epsilon_greedy_action(action_distribution, epsilon=0.90)
        elif self.exploration_exploitaion_setting == 'epsilon_greedy_annealed_1.0->0.001':
            action = epsilon_greedy_annealed(action_distribution, epsilon_percentage, 1.0, 0.001)
        elif self.exploration_exploitaion_setting == 'epsilon_greedy_annealed_0.5->0.001':
            action = epsilon_greedy_annealed(action_distribution, epsilon_percentage, 0.5, 0.001)
        elif self.exploration_exploitaion_setting == 'epsilon_greedy_annealed_0.25->0.001':
            action = epsilon_greedy_annealed(action_distribution, epsilon_percentage, 0.25, 0.001)
        return  action
    def predict_action(self, state, epsilon_percentage):

        action_distribution = self.session.run(self.output, feed_dict={self.state: [state]})[0]
        action = self.sample_action_from_distribution(action_distribution,epsilon_percentage)

        return action


class EpisodeHistory(object):

    def __init__(self):
        self.states=[]
        self.actions=[]
        self.rewards = []
        self.discounted_returns=[]
        self.state_primes = []

    def add_to_history(self, state, action, reward, state_prime):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.state_primes.append(state_prime)

class Memory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []

    def reset_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []

    def add_episode(self, episode):
        self.states += episode.states
        self.actions += episode.actions
        self.rewards += episode.rewards
        self.discounted_returns += episode.discounted_returns


def main():

    total_episodes = 5000
    total_steps_max = 10000
    epsilon_stop = 3000
    train_frequency = 8
    max_episode_length = 500
    render_start = 1
    should_render = True

    explore_exploitation_setting = 'epsilon_greedy_annealed_1.0->0.001'
    env= gym.make('CartPole-v0')
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n
    solved = False

    with tf.Session() as session:
        agent = PGradient(session=session,state_size=state_size,num_actions=num_actions,exploration_exploitation_setting=explore_exploitation_setting,
                          hidden_size=16,discount_factor=0.99, learning_rate=1e-3)

        session.run(tf.global_variables_initializer())
        episode_rewards = []

        batch_losses = []

        global_memory = Memory()
        steps = 0

        for i in tqdm.tqdm(range(total_episodes)):
            state = env.reset()  # state is zero, starting case will be s = s`
            episode_reward = 0.0
            episode_history = EpisodeHistory()
            epsilon_percentage = float(min(i/(epsilon_stop), 1.0))
            for j in range(max_episode_length):
                action = agent.predict_action(state, epsilon_percentage)

                state_prime, reward, terminal, _ = env.step(action=action)  # state_prime is next state which will happen after you execute env.step

                if (render_start > 0 and i > render_start and should_render) or (solved and should_render):

                    env.render()

                episode_history.add_to_history(state, action, reward, state_prime)

                state = state_prime
                episode_reward += reward
                steps+=1
                if terminal:
                    episode_history.discounted_returns = discount_rewards(rewards=episode_history.rewards)
                    global_memory.add_episode(episode_history)

                    if np.mod(i, train_frequency) == 0:
                        feed_dict = {
                            agent.reward_input: np.array(global_memory.discounted_returns),
                            agent.action_input: np.array(global_memory.actions),
                            agent.state: np.array(global_memory.states)
                        }
                        _, batch_loss = session.run([agent.train_step, agent.loss], feed_dict=feed_dict)

                        batch_losses.append(batch_loss)
                        global_memory.reset_memory()

                    episode_rewards.append(episode_reward)
                    break

            if i % 10:
                if np.mean(episode_rewards[:-100])> 100.0:
                    solved = True

                else:
                    solved = False

        print('Solved:', solved, 'Mean Reward', np.mean(episode_rewards[:-100]))

main()
