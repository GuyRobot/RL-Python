import gym
import ptan
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from RL.Library import Common as common
from RL.Library import DeepQNetModel as dqn_model


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
#    params['epsilon_frames'] = 200000
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-basic")
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    # Here we create our agent, which needs a network to convert observations into
    # the action values and an action selector to decide which action to take. For
    # the action selector, we use epsilon-greedy policy with epsilon decayed
    # according to our schedule defined by hyperparams.
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # which is one-step
    # ExperienceSourceFirstLast and experience replay buffer, which will store
    # a fixed amount of transitions.
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

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
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()