import gym
import torch as t
import numpy as np
import cv2
import argparse
from tensorboardX import SummaryWriter
from torchvision import utils

log = gym.logger

log.set_level(gym.logger.INFO)

IMAGE_SIZE = 64

BATCH_SIZE = 16

class InputWrapper(gym.ObservationWrapper):

    def __init__(self, *args):

        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gym.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gym.spaces.Box(self.observation_space(old_space.low), self.observation_space(old_space.high), dtype=np.float32)


    def observation(self, observation):

        new_obs = cv2.resize(observation, (IMAGE_SIZE,IMAGE_SIZE))

        new_obs = np.moveaxis(new_obs, 2, 0)

        return new_obs.astype(np.float32) / 255.0


class Generator(t.nn.Module):

    def __init__(self, input_shape, hidden_size):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        self.map1 = t.nn.Linear(input_shape, hidden_size)
        self.map2  = t.nn.Linear(hidden_size, hidden_size)
        self.map3 = t.nn.Linear(hidden_size, output_size)


    def forward(self, x):











class Discriminator(t.nn.Module):

    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape












def iterate_batches(envs, batch_size=BATCH_SIZE):

    batch = [e.reset() for e in envs]
    env_gen = iter(lambda: np.random.choice(envs), None )

    while True:

        e = next(env_gen)

        obs, reward, done, _ = e.step(e.action_space.sample())

        if np.mean(obs) > 0.01:

            batch.append(obs)

        if len(batch) == batch_size:
            yield t.FloatTensor(batch)
            batch.clear()

        if done:
            e.reset()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true')

    args = parser.parse_args()
    device = t.device("cuda" if args.cuda else "cpu")

    env_names = ('Breakout-v0', 'AirRaid-v0', 'Pong-v0')

    envs = [InputWrapper(gym.make(name)) for name in env_names]

    input_shape = envs[0].observation_space.shape

    hidden_size = 20

    Writer = SummaryWriter()

    net_discr = Discriminator(input_shape=input_shape).to(device)

    net_gener = Generator(input_shape=input_shape, hidden_size=hidden_size).to(device)

    objective = t.nn.BCELoss()

    LEARNING_RATE = 1e-3

    REPORT_EVERY_ITER = 10

    BATCH_SIZE = 32

    SAVE_IMAGE_EVERY_ITER = 10

    LATENT_VECTOR_SIZE = 100

    gen_optimizer = t.optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE)
    dis_optimizer = t.optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE)

    gen_losses = []

    dis_losses = []

    iter_no = 0


    true_labels_v = t.ones(BATCH_SIZE, dtype=t.float32, device=device)
    fake_labels_v = t.zeros(BATCH_SIZE, dtype=t.float32, device=device)

    for batch_v in iterate_batches(envs):

        gen_input_v = t.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1).normal_(0, 1).to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)

        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_gener(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no +=1

        if iter_no % REPORT_EVERY_ITER == 0:

            log.info("Iter %d: gen_loss = %.3e, dis_loss = %.3e", iter_no, np.mean(gen_losses), np.mean(dis_losses))
            Writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            Writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)

            gen_losses = []

            dis_losses = []

        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:

            Writer.add_image("fake", utils.make_grid(gen_output_v.data[:64]), iter_no)
            Writer.add_image("real", utils.make_grid(batch_v.data[:64]), iter_no)
