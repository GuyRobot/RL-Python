import gym
import gym.envs.toy_text
import gym.wrappers
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from RL.CrossEntropy.CrossEntropyCartPole.CrossEntropyCartPole import iterate_batches, Net


HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)

        self.observation_space = gym.spaces.Box(0.0, 1.0,
                                                (env.observation_space.n, ),
                                                dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


def filter_batch(batch, percentile):
    """
    Calculate reward and return good episodes
    :param batch:
    :param percentile:
    :return: elite batch, obs, action, reward_bound
    """
    disc_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)
    return elite_batch, train_obs, train_act, reward_bound


if __name__ == '__main__':
    random.seed(12345)
    env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(is_slippery=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    env = DiscreteOneHotWrapper(env)
    # env = gym.wrappers.Monitor(env, directory="mon", force=True)
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)
    writer = SummaryWriter(comment="-frozenlake-nonslippery")

    full_batch = []
    for epoch, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_bound = filter_batch(batch, PERCENTILE)
        if not full_batch:
            continue
        obs_v = torch.FloatTensor(obs)
        acts_v = torch.LongTensor(acts)
        full_batch = full_batch[-500:]  # take the last 500 elite batch

        optimizer.zero_grad()
        action_scores_v = net(obs_v)  # reward
        loss_v = objective(action_scores_v, acts_v)  # loss
        loss_v.backward()  # back propagation
        optimizer.step()  # update paramaters
        print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (
            epoch, loss_v.item(), reward_mean, reward_bound, len(full_batch)))
        writer.add_scalar("loss", loss_v.item(), epoch)
        writer.add_scalar("reward_mean", reward_mean, epoch)
        writer.add_scalar("reward_bound", reward_bound, epoch)
        if reward_mean > 0.8:
            print("Solve!")
            break
    writer.close()

