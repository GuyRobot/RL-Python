"""
    Combine to hybrid network
        Categorical DQN: Our network will predict the value probability
            distribution of actions.
        Dueling DQN: Our network will have two separate paths for value of
            state distribution and advantage distribution. On the output, both paths
            will be summed together, providing the final value probability
            distributions for actions. To force advantage distribution to have a zero
            mean, we'll subtract distribution with mean advantage in every atom.
        NoisyNet: Our linear layers in the value and advantage paths will be
            noisy variants of nn.Linear.
        Use prioritized replay buffer
        to keep environment transitions and sample them proportionally to KL-divergence.
        Finally, we'll unroll the Bellman equation to n-steps and use the
        double DQN action selection process to prevent the overestimation of values
        of states.
"""
import numpy as np
import torch.nn as nn
import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
import ptan
import argparse
from RL.Library import Common
from RL.Library import Common as common
from RL.Library import NoisyNet

REWARD_STEPS = 2
# priority replay
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000
# C51
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # A value network path predicts the distribution of values for
        # the input state, thus giving us a single vector of N_ATOMS for every batch
        # sample. The advantage path produces the distribution for every action that we
        # have in the game
        self.fc_val = nn.Sequential(
            NoisyNet.NoisyLinear(conv_out_size, 256),
            nn.ReLU(),
            NoisyNet.NoisyLinear(256, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            NoisyNet.NoisyLinear(conv_out_size, 256),
            NoisyNet.NoisyLinear(256, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin,
                                                      Vmax + DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        the value path will be reshaped into (batch_size,
        1, N_ATOMS), so the second dimension will be broadcasted to all actions in the
        advantage path. The baseline advantage that we need to subtract is obtained
        by calculating the mean advantage for every atom over all actions. The
        keepdim=True argument asks the mean() call to keep the second dimension,
        which produces the tensor of (batch_size, 1, N_ATOMS)
        :param x:
        :return:
        """
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, N_ATOMS)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean

    def both(self, x):
        """
        combine probability
        distributions into Q-values, without calling the network several times
        :param x:
        :return:
        """
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    """
    Here we use a small trick to speed up our calculations a bit. As the double
    DQN method requires us to use our main network to select actions but use
    the target network to obtain values (in our case, value distributions) for those
    actions, we need to pass to our main network both the current states and the
    next states. Earlier, we calculated the network output in two calls, which is
    not very efficient on GPU. Now, we concatenate both current states and next
    states into one tensor and obtain the result in one network pass, splitting the
    result later. We need to calculate both Q-values and raw values' distributions,
    as our action selection policy is still greedy: we choose the action with the
    largest Q-value.
    :param batch:
    :param batch_weights:
    :param net:
    :param tgt_net:
    :param gamma:
    :param  Union[str, torch.device] device:
    :return:
    """

    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]

    # we decide on actions to take in the next state and
    # obtain the distribution of those actions using our target network. So, the
    # above net/tgt_net shuffling implements the double DQN method. Then we
    # apply softmax to distribution for those best actions and copy the data into
    # CPU to perform the Bellman projection.
    next_actions_v = next_qvals_v.max(1)[1]
    next_distr_v = tgt_net(next_states_v)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = tgt_net.apply_softmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()

    dones = dones.astype(np.bool)
    proj_distr = common.distribute_projection(next_best_distr, rewards,
                                              dones, v_min=Vmin, v_max=Vmax,
                                              n_atoms=N_ATOMS, gamma=gamma)
    # Here we obtain the distributions for taken actions and apply log_softmax to
    # calculate the loss
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)

    # calculate the KL-divergence loss, multiply
    # it by weights and return two quantities: combined loss to be used in the
    # optimizer step and individual loss values for batch, which will be used as
    # priorities in the replay buffer
    proj_distr_v = torch.tensor(proj_distr)
    loss_v = -state_log_sm_v * proj_distr_v
    loss_v = batch_weights_v * loss_v.sum(dim=1)
    return loss_v.mean(), loss_v + 1e-5


if __name__ == '__main__':
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env=env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-rainbow")

    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)

    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x),
                                ptan.actions.ArgmaxActionSelector(),
                                device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent,
                                                           gamma=params['gamma'],
                                                           steps_count=REWARD_STEPS)
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, params['replay_size'],
                                                     PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    beta = BETA_START

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            # Call to buffer.populate(1) will start the following
            # chain of actions
            # ExperienceReplayBuffer will ask the experience source to get the next
            #   transition.
            # The experience source will feed the current observation to the agent to
            #   obtain the action.
            # The agent will apply the NN to the observation to calculate Q-values,
            #   then ask the action selector to choose the action to take.
            # The action selector (which is an epsilon-greedy selector) will generate
            #   the random number to check how to act: greedily or randomly. In both
            #   cases, it will decide which action to take.
            # The action will be returned to the experience source, which will feed it
            #   into the environment to obtain the reward and the next observation. All
            #   this data (the current observation, action, reward, and next observation)
            #   will be returned to the buffer.
            # The buffer will store the transition, pushing out old observations to keep
            #   its length constant.
            buffer.populate(1)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx):
                    break
            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                               params['gamma'] ** REWARD_STEPS, device=device)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()










