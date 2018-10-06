import numpy as np
import tensorforce.environments as te
import gym
import _pickle as pickle


#Hyperparameters

H = 200
decay_rate = 0.99  # it is for rmsprop
gamma = 0.99
learning_rate = 1e-4
batch_size = 10  # For every 10 episodes parameters gets update
resume = False
render = False  # resume from previous checkpoint?
D = 80 * 80

if resume:
    pickle.load(open('save.p', 'rb'))  # seems  like rb mode is deprecated try with r
else:
    model={}
    model['W1'] = np.random.randn(H,D)/np.sqrt(D)  ## xavier intitialization
    model['W2'] = np.random.randn(H)/np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items()}


def sigmoid(x):
    return  1.0/(1.0 + np.exp(-x))

def prepro(I):
    I=I[35:195]
    I=I[::2,::2,0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0] = 1
    return I.astype(np.float).ravel()


def forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 # ReLu
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h

def backward(eph, epdlogp):
    """ backward pass. eph is an array of intermediate hidden states """
    dW2 = np.dot(eph.T, epdlogp).ravel()  # which means
    dh = np.outer(epdlogp, model['W2'])
    dh[eph<0] = 0
    dW1 = np.dot(dh.T, epx)

    return {'W1':dW1, 'W2':dW2}


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None
xs, hs, dlogps, drs = [],[],[],[]
running_reward = None
reward_sum=0
episode_number = 0

while True:

    if render:env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    aprob, h = forward(x)

    action = 2 if np.random.uniform() < aprob else 3

    xs.append(x) # observations

    hs.append(h) # hidden states






