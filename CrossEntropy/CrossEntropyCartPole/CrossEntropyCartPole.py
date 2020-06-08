from collections import namedtuple

import torch.nn as nn
import torch
import torch.optim as optim
import gym
import numpy as np
from tensorboardX import SummaryWriter

HIDDEN_SIZE = 128  # randomly
BATCH_SIZE = 16
PERCENTILE = 70  # bound for reward


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# Use to represent one step that agent made and store the observation from environment
# and what action the agent completed
Episode = namedtuple('Episode', field_names=['reward', 'steps'])

# Store episode as total reward
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    """

    :param env: Environment
    :param net: Network
    :param batch_size: int, batch size
    :return:
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()  # observation
    softmax = nn.Softmax(dim=1)  # softmax for probability distribution

    while True:
        obs_v = torch.FloatTensor([obs])  # convert to float tensor
        act_probs_v = softmax(net(obs_v))  # action prob
        # Unpack track tensor using .data and convert to numpy
        # Get the first batch ([0]) to obtain one-dimensional vector
        act_probs = act_probs_v.data.numpy()[0]

        # Take action random
        action = np.random.choice(len(act_probs), p=act_probs)
        # Pass to env to obtain observation
        next_obs, reward, is_done, _ = env.step(action)

        episode_reward += reward
        # Note that we save
        # the observation (obs) that was used to choose the action,
        # but not the observation returned by the environment
        # as a result of the action (obs_v). These are the tiny but
        # important details that you need to keep in mind
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        # When episode end we append the final episode to the batch
        # Save total reward and step have taken
        # Then reset to init state
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs  # clear up batch


def filter_batches(batch, percentile):
    """
    This function is at the core of the cross-entropy method: from the given batch
    of episodes and percentile value, it calculates a boundary reward, which is
    used to filter elite episodes to train on. To obtain the boundary reward, we're
    using NumPy's percentile function, which from the list of values and the
    desired percentile, calculates the percentile's value.
    :param batch:
    :param percentile: boundary
    :return: tuple of observation, action, reward bound filtered and reward mean
    """
    rewards = list(map(lambda s: s.reward, batch))
    # Boundary reward to filter episodes
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        # If reward below bound the filter it
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = torch.tensor(train_obs, dtype=torch.float32)
    train_act_v = torch.tensor(train_act, dtype=torch.long)

    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    obs_size = env.observation_space.shape[0]

    n_actions = env.action_space.n
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)
    writer = SummaryWriter(comment="_CartPole-v0")

    # Loop for each episode, take filtered batches
    for epoch, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        # Get observation, actions, reward bound and reward mean from filter batch using cross entropy
        obs_v, acts_v, reward_b, reward_m = filter_batches(batch, PERCENTILE)
        optimizer.zero_grad()
        action_scores_v = net(obs_v)  # take score from observation
        loss_v = loss(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound = %.1f" % (epoch, loss_v.item(), reward_m, reward_b))
        writer.add_scalar("loss", loss_v.item(), epoch)
        writer.add_scalar("reward_bound", reward_b, epoch)
        writer.add_scalar("reward_mean", reward_m, epoch)
        if reward_m > 199:
            print("Solved!")
            break
    writer.close()











