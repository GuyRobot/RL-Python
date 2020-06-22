from RL.DeepQLearning import Wrapper
from RL.Library import DeepQNetModel

import os
import argparse
import time
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

"""
    Hyper parameter
    Our gamma value used for Bellman approximation
    The batch size sampled from the replay buffer (BATCH_SIZE)
    The maximum capacity of the buffer (REPLAY_SIZE)
    The count of frames we wait for before starting training to populate the
    replay buffer (REPLAY_START_SIZE)
    The learning rate used in the Adam optimizer, which is used in this
    example
    How frequently we sync model weights from the training model to the
    target model, which is used to get the value of the next state in the
    Bellman approximation.
"""
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
REPLAY_START_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

"""
    The next chunk of the code defines our experience replay buffer, the purpose
    of which is to keep the last transitions obtained from the environment (tuples
    of the observation, action, reward, done flag, and the next state). Each time
    we do a step in the environment, we push the transition into the buffer,
    keeping only a fixed number of steps, in our case 10k transitions. For
    training, we randomly sample the batch of transitions from the replay buffer,
    which allows us to break the correlation between subsequent steps in the
    environment.

"""
Experience = collections.namedtuple('Experience', field_names=
['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        In the sample() method, we create a list of
        random indices and then repack the sampled entries into NumPy arrays for
        more convenient loss calculation.

        :param batch_size:
        :return:
        """
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward
        new_state = new_state

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def calc_loss(batch, net, tgt_net, device="cpu"):
    """
    In arguments, we pass our batch as a tuple of arrays (repacked by the
    sample() method in the experience buffer), our network that we're training
    and the target network, which is periodically synced with the trained one. The
    first model (passed as the argument net) is used to calculate gradients, while
    the second model in the tgt_net argument is used to calculate values for the
    next states and this calculation shouldn't affect gradients. To achieve this,
    we're using the detach() function of the PyTorch tensor to prevent gradients
    from flowing into the target network's graph. This function was described in

    :param batch:
    :param net:
    :param tgt_net:
    :param device:
    :return:
    """
    states, actions, rewards, dones, new_states = batch
    state_v = torch.tensor(states).to(device)
    new_states_v = torch.tensor(new_states).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    # In the line above, we pass observations to the first model and extract the
    # specific Q-values for the taken actions using the gather() tensor operation.
    # The first argument to the gather() call is a dimension index that we want to
    # perform gathering on (in our case it is equal to 1, which corresponds to
    # actions). The second argument is a tensor of indices of elements to be chosen.
    # Extra unsqueeze() and squeeze() calls are required to fulfill the
    # requirements of the gather functions to the index argument and to get rid of
    # extra dimensions that we created (the index should have the same number of
    # dimensions as the data we're processing)
    state_action_values = net(state_v).gather(
        1, actions_v.unsqueeze(-1)).squeeze(-1)

    # Max Q-value from target network
    next_state_values = tgt_net(new_states_v).max(1)[0]
    # if transition in the batch
    # is from the last step in the episode, then our value of the action doesn't have a
    # discounted reward of the next state, as there is no next state to gather reward
    # from. This may look minor, but this is very important in practice: without
    # this, training will not converge.
    next_state_values[done_mask] = 0.0

    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME,
                        help="Name of the environment, default="
                             + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float,
                        default=MEAN_REWARD_BOUND,
                        help="Mean reward boundary for stop of training, default = % .2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = Wrapper.make_env(args.env)
    net = DeepQNetModel.DQN(env.observation_space.shape, env.action_space.n).to(device)
    if os.path.exists("PongNoFrameskip-v4-best.dat"):
        print("Load Best Model Pretrained")
        net.load_state_dict(torch.load("PongNoFrameskip-v4-best.dat"))
    tgt_net = DeepQNetModel.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment="--" + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, mean reward %.3f, eps %.2f, "
                  "speed %.2f f/s"
                  % (frame_idx, len(total_rewards), mean_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("best mean reward updated %.3f -> %.3f, model save"
                          % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solve in %d frame!" % frame_idx)
                break
        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()


# import argparse
# import time
# import numpy as np
# import collections
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# from tensorboardX import SummaryWriter
#
#
# DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
# MEAN_REWARD_BOUND = 19.5
#
# GAMMA = 0.99
# BATCH_SIZE = 32
# REPLAY_SIZE = 10000
# LEARNING_RATE = 1e-4
# SYNC_TARGET_FRAMES = 1000
# REPLAY_START_SIZE = 10000
#
# EPSILON_DECAY_LAST_FRAME = 10**5
# EPSILON_START = 1.0
# EPSILON_FINAL = 0.02
#
#
# Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
#
#
# class ExperienceBuffer:
#     def __init__(self, capacity):
#         self.buffer = collections.deque(maxlen=capacity)
#
#     def __len__(self):
#         return len(self.buffer)
#
#     def append(self, experience):
#         self.buffer.append(experience)
#
#     def sample(self, batch_size):
#         indices = np.random.choice(len(self.buffer), batch_size, replace=False)
#         states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
#         return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
#                np.array(dones, dtype=np.uint8), np.array(next_states)
#
#
# class Agent:
#     def __init__(self, env, exp_buffer):
#         self.env = env
#         self.exp_buffer = exp_buffer
#         self._reset()
#
#     def _reset(self):
#         self.state = env.reset()
#         self.total_reward = 0.0
#
#     def play_step(self, net, epsilon=0.0, device="cuda"):
#         done_reward = None
#
#         if np.random.random() < epsilon:
#             action = env.action_space.sample()
#         else:
#             state_a = np.array([self.state], copy=False)
#             state_v = torch.tensor(state_a).to(device)
#             q_vals_v = net(state_v)
#             _, act_v = torch.max(q_vals_v, dim=1)
#             action = int(act_v.item())
#
#         # do step in the environment
#         new_state, reward, is_done, _ = self.env.step(action)
#         self.total_reward += reward
#
#         exp = Experience(self.state, action, reward, is_done, new_state)
#         self.exp_buffer.append(exp)
#         self.state = new_state
#         if is_done:
#             done_reward = self.total_reward
#             self._reset()
#         return done_reward
#
#
# def calc_loss(batch, net, tgt_net, device="cuda"):
#     states, actions, rewards, dones, next_states = batch
#
#     states_v = torch.tensor(states).to(device)
#     next_states_v = torch.tensor(next_states).to(device)
#     actions_v = torch.tensor(actions, dtype=torch.long).to(device)
#     rewards_v = torch.tensor(rewards).to(device)
#     done_mask = torch.tensor(dones, dtype=torch.bool).to(device)
#
#     state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
#     next_state_values = tgt_net(next_states_v).max(1)[0]
#     next_state_values[done_mask] = 0.0
#     next_state_values = next_state_values.detach()
#
#     expected_state_action_values = next_state_values * GAMMA + rewards_v
#     return nn.MSELoss()(state_action_values, expected_state_action_values)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
#     parser.add_argument("--env", default=DEFAULT_ENV_NAME,
#                         help="Name of the environment, default=" + DEFAULT_ENV_NAME)
#     parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND,
#                         help="Mean reward boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
#     args = parser.parse_args()
#     device = torch.device("cuda" if args.cuda else "cpu")
#
#     env = wrappers.make_env(args.env)
#
#     net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
#     net.load_state_dict(torch.load("PongNoFrameskip-v4-best.dat"))
#     tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
#     writer = SummaryWriter(comment="-" + args.env)
#     print(net)
#
#     buffer = ExperienceBuffer(REPLAY_SIZE)
#     agent = Agent(env, buffer)
#     epsilon = EPSILON_START
#
#     optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
#     total_rewards = []
#     frame_idx = 0
#     ts_frame = 0
#     ts = time.time()
#     best_mean_reward = None
#
#     while True:
#         frame_idx += 1
#         epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
#
#         reward = agent.play_step(net, epsilon, device=device)
#         if reward is not None:
#             total_rewards.append(reward)
#             speed = (frame_idx - ts_frame) / (time.time() - ts)
#             ts_frame = frame_idx
#             ts = time.time()
#             mean_reward = np.mean(total_rewards[-100:])
#             print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
#                 frame_idx, len(total_rewards), mean_reward, epsilon,
#                 speed
#             ))
#             writer.add_scalar("epsilon", epsilon, frame_idx)
#             writer.add_scalar("speed", speed, frame_idx)
#             writer.add_scalar("reward_100", mean_reward, frame_idx)
#             writer.add_scalar("reward", reward, frame_idx)
#             if best_mean_reward is None or best_mean_reward < mean_reward:
#                 torch.save(net.state_dict(), args.env + "-best.dat")
#                 if best_mean_reward is not None:
#                     print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
#                 best_mean_reward = mean_reward
#             if mean_reward > args.reward:
#                 print("Solved in %d frames!" % frame_idx)
#                 break
#
#         if len(buffer) < REPLAY_START_SIZE:
#             continue
#
#         if frame_idx % SYNC_TARGET_FRAMES == 0:
#             tgt_net.load_state_dict(net.state_dict())
#
#         optimizer.zero_grad()
#         batch = buffer.sample(BATCH_SIZE)
#         loss_t = calc_loss(batch, net, tgt_net, device=device)
#         loss_t.backward()
#         optimizer.step()
#     writer.close()
