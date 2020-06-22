"""
    Add a noise to the weights of fully connected layers of the network and adjust the parameters of this noise
    during training using backpropagation
    Two approaches:
        1. Independent Gaussian noise: For every weight in a fully-connected
            layer, we have a random value that we draw from the normal
            distribution. Parameters of the noise μ and σ are stored inside the layer
            and get trained using backpropagation, the same way that we train
            weights of the standard linear layer. The output of such a 'noisy layer' is
            calculated in the same way as in a linear layer.
        2. Factorized Gaussian noise: To minimize the amount of random values
            to be sampled, the authors proposed keeping only two random vectors,
            one with the size of input and another with the size of the output of the
            layer. Then, a random matrix for the layer is created by calculating the
            outer product of the vectors.
"""
import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import gym
import ptan
from tensorboardX import SummaryWriter
from RL.Library import Common as common, DeepQNetModel as dqn_model

import argparse


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weights = nn.Parameter(torch.full((out_features, in_features),
                                                     sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.epsilon_weight * self.epsilon_bias, bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """

    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.noisy_layers = [
            NoisyLinear(conv_out_size, 512),
            NoisyLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        """
        Get conv out size use fake data (all 0)
        :param shape:
        :return:
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        The final piece of the model is the forward() function, which accepts the 4D
        input tensor (the first dimension is batch size, the second is the color channel,
        which is our stack of subsequent frames, while the third and fourth are image
        dimensions). The application of transformations is done in two steps: first we
        apply the convolution layer to the input and then we obtain a 4D tensor on
        output. This result is flattened to have two dimensions: a batch size and all
        the parameters returned by the convolution for this batch entry as one long
        vector of numbers. This is done by the view() function of the tensors, which
        lets one single dimension be a -1 argument as a wildcard for the rest of the
        parameters. For example, if we have a tensor T of shape (2, 3, 4), which is
        a 3D tensor of 24 elements, we can reshape it into a 2D tensor with six rows
        and four columns using T.view(6, 4). This operation doesn't create a new
        memory object or move the data in memory, it just changes the higher-level
        shape of the tensor. The same result could be obtained by T.view(-1, 4) or
        T.view(6, -1), which is very convenient when your tensor has a batch size
        in the first dimension. Finally, we pass this flattened 2D tensor to our fully
        connected layers to obtain Q-values for every batch input.
        :param x:
        :return: output tuple after forward computation
        """
        x = x.float() / 256
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        """
        (SNR) of our noisy
        layers, which is a ratio of RMS(μ) / RMS(σ), where RMS is the root mean
        square of the corresponding weights. In our case, SNR shows how many
        times the stationary component of the noisy layer is larger than the injected
        noise.
        :return: SNR
        """
        return [((layer.weight ** 2).mean().sqrt() /
                 (layer.sigma_weight ** 2).mean().sqrt()).data.cpu().numpy()[0]
                for layer in self.noisy_layers]


if __name__ == "__main__":
    # Almost the same just query SNR values for noisy layers from network
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-noisy-net")
    net = NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    agent = ptan.agent.DQNAgent(net, ptan.actions.ArgmaxActionSelector(), device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx):
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

            if frame_idx % 500 == 0:
                for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                    writer.add_scalar("sigma_snr_layer_%d" % (layer_idx+1),
                                      sigma_l2, frame_idx)