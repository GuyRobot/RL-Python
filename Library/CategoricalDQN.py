"""
    Use probability instead of number average
    Bellman equation can be generalized for a distribution case and it will
    have a form:
        Z(x,a) = R(x,a) +y*Z(x',a')
    , which is very similar to the familiar
    Bellman equation, but now Z(x, a), R(x, a) are the probability distributions
    and not numbers.

    The resulting distribution can be used to train our network to give better
    predictions of value distribution for every action of the given state, exactly
    the same way as with Q-learning. The only difference will be in the loss
    function, which now has to be replaced to something suitable for
    distributions' comparison.

    The central part of the method is probability distribution, which we're
    approximating
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


def distribute_projection(next_distribute, rewards, dones, v_min, v_max,
                          n_atoms, gamma):
    """
    For every atom (we have 51 of them), our network predicts the probability
    that future discounted value will fall into this atom's range. The central part
    of the method is the code, which performs the contraction of distribution of
    the next state's best action using gamma, adds local reward to the distribution
    and projects the results back into our original atoms. The following is the
    function that does exactly this:

    :param next_distribute:
    :param rewards:
    :param dones:
    :param v_min:
    :param v_max:
    :param n_atoms:
    :param gamma:
    :return:
    """
    # In the beginning, we allocate the array that will keep the result of the
    # projection. This function expects the batch of distributions with a shape
    # (batch_size, n_atoms), array of rewards
    batch_size = len(rewards)
    proj_dist = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (v_max - v_min) / (n_atoms - 1)
    # iterate over every atom in the original distribution
    # that we have and calculate the place that this atom will be projected by the
    # Bellman operator, taking into account our value bounds
    # For example, the
    # very first atom, with index 0, corresponds with value V_min=-10, but for the
    # sample with reward +1 will be projected into value -10 * 0.99 + 1 = -8.9. In
    # other words, it will be shifted to the right (assume our gamma=0.99)
    for atom in range(n_atoms):
        # clip it when out of range v_min max (use min and max)
        tz_j = np.minimum(v_max, np.maximum(v_min, rewards +
                                            (v_min + atom * delta_z)))
        # calculate the atom numbers that our samples have
        # projected.
        b_j = (tz_j - v_min) / delta_z
        # our target atom can land exactly at some atom's
        # position. In that case, we just need to add the source distribution value to the
        # target atom.

        # l and u (which correspond to the indices of atoms below and above the
        # projected point)
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_dist[eq_mask, l[eq_mask]] += next_distribute[eq_mask, atom]

        # When the projected point lands between atoms, we need to spread the
        # probability of the source atom between atoms below and above.
        ne_mask = u != l
        proj_dist[ne_mask, l[ne_mask]] += next_distribute[ne_mask,
                                                          atom] * (u - b_j)[ne_mask]
        proj_dist[ne_mask, u[ne_mask]] += next_distribute[ne_mask,
                                                          atom] * (b_j - l)[ne_mask]

        # our projection shouldn't
        # take into account the next distribution and will just have a 1 probability
        # corresponding to the reward obtained. However, we need, again, to take into
        # account our atoms and properly distribute this probability if the reward value
        # falls between the atoms. This case is handled by the code branch below,
        # which zeroes resulting distribution for samples with the done flag set and
        # then calculates the resulting projection.

        if dones.any():
            proj_dist[dones] = 0.0
            tz_j = np.minimum(v_max, np.maximum(v_min, rewards[dones]))
            b_j = (tz_j - v_min) / delta_z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            eq_dones = dones.copy()
            eq_dones[dones] = eq_mask
            if eq_dones.any():
                proj_dist[eq_dones, l] = 1.0
            ne_mask = u != l
            ne_dones = dones.copy()
            ne_dones[dones] = ne_mask
            if ne_dones.any():
                proj_dist[ne_dones, l] = (u - b_j)[ne_mask]
                proj_dist[ne_dones, u] = (b_j - l)[ne_mask]
    return proj_dist


SAVE_STATES_IMG = False
SAVE_TRANSITIONS_IMG = False
if SAVE_STATES_IMG or SAVE_TRANSITIONS_IMG:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pylab as plt

v_max = 10
v_min = -10
N_ATOMS = 51
DELTA_Z = (v_max - v_min) / (N_ATOMS - 1)

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100
SAVE_STATES_IMG = False
SAVE_TRANSITIONS_IMG = False


class DistributionalDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DistributionalDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64,
                      kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64,
                      kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(v_min, v_max + DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 255
        conv_out = self.conv(fx).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        return fc_out.view(batch_size, -1, N_ATOMS)

    def both(self, x):
        """
        returns both raw
        distribution and Q-values. Q-values will be used to make decisions on
        actions
        :param x:
        :return:
        """
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        """
        Get Q values
        :param x: input
        :return: Q values
        """

        return self.both(x)[1]

    def apply_softmax(self, t):
        """
        Apply softmax to output tensor
        :param t: output
        :return: softmax of t
        """
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


def calc_loss(batch, net, tgt_net, gamma, device="cpu", save_prefix=None):
    """

    :param batch:
    :param net:
    :param tgt_net:
    :param gamma:
    :param device:
    :type device Union[str, torch.device]
    :param save_prefix:
    :return:
    """
    states, actions, rewards, dones, next_states = Common.unpack_batch(batch)
    batch_size = len(batch)

    state_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions, dtype=torch.long).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    # need both probability distributions and Q-values for the next
    # states, so we use the both() call to the network, obtain the best actions to
    # take in the next state, apply softmax to the distribution, and convert it to the
    # array.
    next_distribute_v, next_qvals_v = tgt_net.both(next_states_v)
    next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distribute = tgt_net.apply_softmax(next_distribute_v).data.cpu().numpy()

    # extract distributions of the best actions and perform their projection
    # using the Bellman operator. The result of the projection will be target
    # distribution about what we want our network output
    next_best_distribute = next_distribute[range(batch_size), next_actions]
    dones = dones.astype(torch.bool)
    proj_distribute = Common.distribute_projection(next_best_distribute,
                                                   rewards,
                                                   dones,
                                                   v_min,
                                                   v_max,
                                                   N_ATOMS,
                                                   gamma)

    # compute the output of the network and
    # calculate KL-divergence between projected distribution and the network's
    # output for the taken actions. KL-divergence shows how much two
    # distributions differ
    distribute_v = net(state_v)
    state_action_values = distribute_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distribute_v = torch.tensor(proj_distribute).to(device)
    loss_v = -state_log_sm_v * proj_distribute_v
    return loss_v.sum(dim=1).mean()


def calc_values_of_states(states, net, device="cpu"):
    """

    :param states:
    :param net:
    :param device:
    :type device Union[str, torch.device]
    :return:
    """
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net.qvals(states_v)
        best_actions_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_actions_values_v.mean().item())
    return np.mean(mean_vals)


def save_state_images(frame_idx, states, net, device="cpu", max_states=200):
    """

    :param frame_idx:
    :param states:
    :param net:
    :param Union[str, torch,device] device:
    :param max_states:
    :return:
    """
    ofs = 0
    p = np.arange(v_min, v_max + DELTA_Z, DELTA_Z)
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_prob = net.apply_softmax(net(states_v)).data.cpu().numpy()
        batch_size, num_actions, _ = action_prob.shape
        for batch_idx in range(batch_size):
            plt.clf()
            for action_idx in range(num_actions):
                plt.subplot(num_actions, 1, action_idx + 1)
                plt.bar(p, action_prob[batch_idx, action_idx], width=.5)
            plt.savefig("sates/%05d_%08d.png" % (ofs + batch_idx, frame_idx))
        ofs += batch_size
        if ofs >= max_states:
            break


def save_transition_images(batch_size, predicted, projected, next_distribute,
                           domes, rewards, save_prefix):
    for batch_idx in range(batch_size):
        is_done = domes[batch_idx]
        reward = rewards[batch_idx]
        plt.clf()
        p = np.arange(v_min, v_max + DELTA_Z, DELTA_Z)
        plt.subplot(3, 1, 1)
        plt.bar(p, predicted[batch_idx], width=.5)
        plt.title('Predicted')
        plt.subplot(3, 1, 2)
        plt.bar(p, projected[batch_idx], width=.5)
        plt.title("Projected")
        plt.subplot(3, 1, 3)
        plt.bar(p, next_distribute[batch_idx], width=.5)
        plt.title("Next State")
        suffix = ""
        if reward != 0.0:
            suffix = suffix + "_%.0f" % reward
        if is_done:
            suffix = suffix + '_done'
        plt.savefig("%s_%02d%s.png" % (save_prefix, batch_idx, suffix))


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
    #    params['epsilon_frames'] *= 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-distrib")
    net = DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    eval_states = None
    prev_save = 0
    save_prefix = None

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])

            save_prefix = None
            if SAVE_TRANSITIONS_IMG:
                interesting = any(map(lambda s: s.last_state is None or s.reward != 0.0, batch))
                if interesting and frame_idx // 30000 > prev_save:
                    save_prefix = "images/img_%08d" % frame_idx
                    prev_save = frame_idx // 30000

            loss_v = calc_loss(batch, net, tgt_net.target_model, gamma=params['gamma'],
                               device=device, save_prefix=save_prefix)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()

            if frame_idx % EVAL_EVERY_FRAME == 0:
                mean_val = calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, frame_idx)

            if SAVE_STATES_IMG and frame_idx % 10000 == 0:
                save_state_images(frame_idx, eval_states, net, device=device)
