import os

import gym
import ptan
import torch
import torch.optim as optim
import numpy as np
from RL.Library import Common, DeepQNetModel
from tensorboardX import SummaryWriter
import argparse


PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000


class PrioReplayBuffer:

    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        """
        The class for the priority replay buffer stores samples in a circular buffer (it
        allows us to keep a fixed amount of entries without reallocating the list) and
        NumPy array to keep priorities. We also store the iterator to the experience
        source object, to pull the samples from the environment.
        :param exp_source:
        :param buf_size:
        :param prob_alpha:
        """
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        """
        The populate() method needs to pull the given amount of transitions from
        the ExperienceSource object and store them in the buffer. As our storage for
        the transitions is implemented as a circular buffer, we have two different
        situations with this buffer:
            When our buffer hasn't reached the maximum capacity, we just need to
                append a new transition to the buffer.
            If the buffer is already full, we need to overwrite the oldest transition,
                which is tracked by the pos class field, and adjust this position module's
                buffer size
        :param count:
        :return:
        """
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """
        Convert priorities to probabilities using our
        hyperparameter prob_alpha.
        Then, using those probabilities, we sample our buffer to obtain a batch of
        samples.
        Finally calculate weights for samples and return batch, indices and weights
        :param batch_size:
        :param beta:
        :return:
        """

        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """
        Update new
        priorities for the processed batch
        :param batch_indices:
        :param batch_priorities:
        :return:
        """

        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    """
    We need to
    calculate the MSE and explicitly multiply the result on weight

    Take into account weights of samples and keep individual loss values for
    every sample. Those values will be passed to the priority replay buffer to
    update priorities. Small values are added to every loss to handle the situation
    of zero loss value, which will lead to zero priority of entry
    :param batch:
    :param batch_weights:
    :param net:
    :param tgt_net:
    :param gamma:
    :param device:
    :return:
    """
    states, actions, rewards, dones, next_states = Common.unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_states_values = tgt_net(next_states_v).max(1)[0]
    next_states_values[done_mask] = 0.0

    expected_states_action_values = next_states_values.detach() * gamma + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_states_action_values) ** 2
    return losses_v.mean(), losses_v + 1e-5


if __name__ == '__main__':
    params = Common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True,
                        action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    path = params['env_name'] + "-best.dat"

    writer = SummaryWriter(comment="-" + params['run_name'] + "-prio-replay")
    net = DeepQNetModel.DQN(env.observation_space.shape, env.action_space.n).to(device)
    if os.path.exists(path):
        print("Load Best Model Pretrained")
        net.load_state_dict(torch.load(path))
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = Common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, action_selector=selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent,
                                                           gamma=params['gamma'])
    buffer = PrioReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    beta = BETA_START
    best_mean_reward = None

    if os.path.exists("best_reward.txt"):
        f = open("best_reward.txt")
        best_mean_reward = float(f.readline())
        print("Best Mean Reward Previous", best_mean_reward)

    with Common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            # Pull one transition from the experience
            # source and update the epsilon according to the schedule. We use the similar
            # schedule to linearly increase the beta hyperparameter for priority replay
            # buffer weights' adjustment
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            mean_reward = reward_tracker.get_mean_reward()
            if new_rewards:
                writer.add_scalar("beta", beta, frame_idx)
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break
            if best_mean_reward is None or (mean_reward is not None and best_mean_reward < mean_reward):
                torch.save(net.state_dict(), path)
                if best_mean_reward is not None:
                    print("best mean reward updated %.3f -> %.3f, model save"
                          % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
                f = open("best_reward.txt", "w")
                f.write(str(best_mean_reward))
                f.close()

            if len(buffer) < params['replay_initial']:
                continue

            # We pass both batch and weights
            # to the loss function, the result of which is two things: the first is the
            # accumulated loss value that we need to backpropagate, and the second is a
            # tensor with individual loss values for every sample in the batch. We
            # backpropagate the accumulated loss and ask our priority replay buffer to
            # update the samples' priorities.
            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            loss_v, sample_prios_v = calc_loss(batch, batch_weights,
                                               net, tgt_net.target_model,
                                               gamma=params['gamma'],
                                               device=device)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, batch_priorities=sample_prios_v.data.cpu().numpy())














