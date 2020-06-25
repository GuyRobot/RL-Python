"""
    Observation will include the following information:
        N past bars, where each have open, high, low, and close prices
        An indication that the share was bought some time ago (it will be
            possible to have only one share at a time)
        Profit or loss we currently have from our current position (the share
            bought)

    Actions:
        Do nothing: Skip the bar without taking actions
        Buy a share: If the agent has already got the share, nothing will be
            bought, otherwise we’ll pay the commission, which is usually some
            small percentage of the current price
        Close the position: If we’ve got no share previously bought, nothing
            will happen, otherwise we’ll pay the commission for the trade
    The reward that the agent receives could be expressed in various ways. On
    the one hand, we can split the reward into multiple steps during our
    ownership of the share. In that case, the reward on every step will be equal to
    the last bar’s movement. On the other hand, the agent can receive reward
    only after the close action and receive full reward at once. At the first sight,
    both variants should have the same final result, but maybe with different
    convergence speed. However, in practice, the difference could be dramatic.
"""
import argparse
import numpy as np

from RL.RealLifeProblem.Lib import DataHandler, Models, Enviroment

import torch
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use("Agg")


EPSILON = 0.02


