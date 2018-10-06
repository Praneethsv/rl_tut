import numpy as np
import torch as t
import torch.nn.functional as F
import torch.optim as optim
import os
import time
import math
import ptan
import gym
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter

from Ch14 import model


ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4
TEST_ITERS = 1000


def test_net( net, env, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v, var_v, _ = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count




def calc_logprob(mu_v, var_v, actions_v):

    p1 = - ((mu_v - actions_v) ** 2 / (2 * var_v.clamp(min=1e-3)))
    p2 = - (t.log(t.sqrt(2 * var_v * math.pi * var_v)))

    return p1+p2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = t.device("cuda" if args.cuda else "cpu")
    save_path = os.path.join("save", "a2c-" + args.name)

    env = gym.make(ENV_ID)
    test_env = gym.make(ENV_ID)
    test_env.render()
    net = model.ModelA2C(input_shape=env.observation_space.shape[0], act_size=env.action_space.shape[0]).to(device)
    print(net)
    writer = SummaryWriter(comment="-a2c_" + args.name)
    Agent = model.AgentA2C(net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, Agent, GAMMA, steps_count=REWARD_STEPS)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                reward_steps = exp_source.pop_rewards_steps()
                if reward_steps:
                    rewards, steps = zip(*reward_steps)
                    tb_tracker.track("episode_steps", steps[0], step_idx)
                    tracker.reward(rewards[0], step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(net, test_env, device=device)
                    print("Test done is %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    writer.add_scalar("test_reward", rewards, step_idx)
                    writer.add_scalar("test_steps", steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
                            name = "best_%+.3f_%d.dat" % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            t.save(net.state_dict(), fname)
                        best_reward = rewards
                batch.append(exp)
                if len(batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, ref_vals_v = model.unpack_batch_a2c(batch=batch, net=net, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()
                optimizer.zero_grad()
                mu_v, var_v, values_v = net(states_v)
                loss_value_v = F.mse_loss(values_v.squeeze(-1), ref_vals_v)
                adv_v = ref_vals_v.unsqueeze(dim=-1) - values_v.detach()
                log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = ENTROPY_BETA * (-(t.log(2 * math.pi * var_v) + 1) / 2).mean()

                loss_v = loss_policy_v + entropy_loss_v + loss_value_v
                loss_v.backward()
                optimizer.step()

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", values_v, step_idx)
                tb_tracker.track("batch_rewards", ref_vals_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)








