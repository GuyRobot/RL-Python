"""
    The
    architecture difference from the classic DQN network is shown on the picture
    below. The classic DQN network (top) takes features from the convolution
    layer and, using fully-connected layers, transforms them into a vector of Qvalues, one for each action. On the other hand, dueling DQN (bottom) takes
    convolution features and processes them using two independent paths: one
    path is responsible for V(s) prediction, which is just a single number, and
    another path predicts individual advantage values, having the same
    dimension as Q-values in the classic case. After that, we add V(s) to every
    value of A(s, a) to obtain the Q(s, a), which is used and trained as normal.
    See in images folder
    We have yet another constraint to be set: we want the mean
    value of the advantage of any state to be zero

    subtracting from the Q expression in the network the mean value
    of the advantage, which effectively pulls the mean for advantage to zero:
        Q(s,a) = V(s) + A(s,a) - 1/N * sum_k(A(s,k))
"""


import torch
import torch.nn as nn
import numpy as np


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

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
        # Instead of defining a single path of fully connected layers, we create two
        # different transformations: one for advantages and one for value prediction.
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
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
        # we calculate value and advantage for our batch of
        # samples and add them together, subtracting the mean of advantage to obtain
        # the final Q-values
        x = x.float() / 256
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()


# Train is the same
