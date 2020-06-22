import numpy as np
import torch
import torch.nn as nn
import time

HYPERPARAMS = {
    'pong': {
        'env_name': "PongNoFrameskip-v4",
        'stop_reward': 18.0,
        'run_name': 'pong',
        'replay_size': 100000,
        'replay_initial': 10000,
        'target_net_sync': 1000,
        'epsilon_frames': 10 ** 5,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 32
    },
    'breakout-small': {
        'env_name': "BreakoutNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout-small',
        'replay_size': 3 * 10 ** 5,
        'replay_initial': 20000,
        'target_net_sync': 1000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 64
    },
    'breakout': {
        'env_name': "BreakoutNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
    'invaders': {
        'env_name': "SpaceInvadersNoFrameskip-v4",
        'stop_reward': 500.0,
        'run_name': 'breakout',
        'replay_size': 10 ** 6,
        'replay_initial': 50000,
        'target_net_sync': 10000,
        'epsilon_frames': 10 ** 6,
        'epsilon_start': 1.0,
        'epsilon_final': 0.1,
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'batch_size': 32
    },
}


def unpack_batch(batch):
    """
    To avoid the special
    handling of such cases, for terminal transitions we store the initial state in the
    last_states array. To make our calculations of the Bellman update correct,
    we'll mask such batch entries during the loss calculation using the dones
    array. Another solution would be to calculate the value of last states only for
    non-terminal transitions, but it would make our loss function logic a bit more
    complicated
    :param batch:
    :return:
    """
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), \
           np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
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
        :param gamma:
        :param device:
        :return:
    """
    states, actions, rewards, dones, new_states = unpack_batch(batch)
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

    expected_state_action_values = next_state_values * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward
        self.mean_reward = None

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        self.mean_reward = float(np.mean(self.total_rewards[-100:]))
        epsilon_str = "" if epsilon is None else "., eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s %s"
              % (frame, len(self.total_rewards), self.mean_reward, speed, epsilon_str))

        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", self.mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if self.mean_reward > self.stop_reward:
            print("Solve in %d frames" % frame)
            return True
        return False

    def get_mean_reward(self):
        return self.mean_reward


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self.epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        """
        It updates the value of
        epsilon according to the standard DQN epsilon decay schedule: linearly
        decreasing it for the first epsilon_frames steps and then keeping it constant.
        :param frame:
        :return:
        """
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)


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