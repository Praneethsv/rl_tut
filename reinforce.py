import numpy as np
import tensorflow as tf
import gym
import tensorflow.contrib.slim as slim



class ReinforcePG(object):

    def __init__(self, session, hidden_size, state_size,
                 num_actions, learning_rate, exploration_exploitation_setting ):

        self.session = session
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.exploration_exploitation_setting = exploration_exploitation_setting


    def build_model(self):

       with tf.variable_scope('reinforce-model'):
           self.state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)

           self.h0 = slim.fully_connected(self.state, self.hidden_size)
           self.h1 = slim.fully_connected(self.h0, self.hidden_size)
           self.output = slim.fully_connected(self.h1, self.num_actions, activation_fn=tf.nn.softmax)


#    def train_model(self):


