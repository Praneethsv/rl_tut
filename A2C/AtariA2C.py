import gym
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import ptan
import numpy as np
from tensorboardX import SummaryWriter
import argparse
import time
import sys

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50
REWARD_STEPS = 4
CLIP_GRAD = 0.1

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

class AtariA2C(nn.Module):

    def __init__(self, input_size, num_actions):
        super(AtariA2C, self).__init__()
        self.input_size = input_size
        self.num_actions = num_actions

        self.conv = nn.Sequential(nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1))

        conv_out_size = self._get_conv_out(input_size)

        self.policy = nn.Sequential(nn.Linear(conv_out_size, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, num_actions))

        self.value = nn.Sequential(nn.Linear(conv_out_size, 512), \
                                   nn.ReLU(),
                                   nn.Linear(512, 1))

    def _get_conv_out(self, shape): ## _ meaning it's use is only internal
        o = self.conv(t.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)   ##  flattened output here
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, device="cpu"):
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []

    for stp_idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(stp_idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = t.FloatTensor(states).to(device)
    actions_t = t.LongTensor(actions).to(device)
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = t.FloatTensor(last_states).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np
    ref_vals_v = t.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = t.device("cuda" if args.cuda else "cpu")
    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
    envs = [make_env() for _ in range(NUM_ENVS)] # make_env doesn't work because you are not calling it
    writer = SummaryWriter(comment="-AtariA2C" + args.name)

    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device) # here envs[0].observation_space won't work function object should be called
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    """In the above step lambda is used as a function object to get policy
    from AtariA2C class because PolicyAgent needs policy that is coming from neural nets"""

    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    """epsilon is  a small constant added to denominator in Adam algorithm to prevent division situations typical value
    is 1e-8 or 1e-10 but in this case default values are too small
    giving hard time to algorithm to converge."""

    batch = []
    with RewardTracker(writer, stop_reward=18) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            """ TBMeanTracker is from ptan library and is responsible for writing into TensorBoard the mean of the measured parameters
            for the last 10 steps. This is helpful when training can take millions of steps. so we don't want to write millions of points
            into TensorBoard, but rather write smoothed values every 10 steps."""

            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)
                # handles new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                """ try this one also net(states_v) and check whether it is running  """
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                """so value_v could be a column vector or shape (None, 1) and then we can use value_v.squeeze(-1)"""
                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                """1) good to use detach when calculating gradients for back prop instead of data.
                   both data and detach() will return the values of the tensor.
                   2) We calculate the policy loss to obtain PG. the first two steps are to obtain
                   a log of our policy and calculate advantage of actions, which is A(s,a) = Q(s,a) - V(s).
                   The call to value_v.detach() is important, as we don't want to propagate the PG into our
                   value approximation head"""
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = - log_prob_actions_v.mean()
                """Note: retain_graph is useful when we need to back propagate loss multiple times before the call to
                the optimizer and optimizer.step() tells the optimizer to update the network"""
                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean() # this already gives you negative sign?
                """The above piece of our loss function is entropy loss, which equals to the
                scaled entropy of our policy, taken with opposite sign"""
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten() for p in net.parameters() if p.grad is not None])
                """ The above step is to calculate and accumulate the gradients"""
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm(net.parameters(), CLIP_GRAD)
                optimizer.step()
                loss_v += loss_policy_v

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)






