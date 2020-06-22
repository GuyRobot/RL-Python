import gym
import ptan
import numpy as np
import torch.nn as nn

env = gym.make("CartPole-v0")
net = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 256),
    nn.ReLU(),
    nn.Linear(256, env.action_space.n)
)

action_selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=0.1)
agent = ptan.agent.DQNAgent(net, action_selector)

"""
    Experience buffer
    In case of a DQN, we rarely want to learn from the experience once we get
        it. We usually store it in some large buffer and perform a random sample
        from it to obtain the minibatch to train on. This scenario is supported by the
        ptan.experience.ExperienceReplayBuffer class, which is very similar to
        the implementation we've seen in the previous chapter. To construct it, we
        need to pass the experience source and size of the buffer. By calling the
        populate(n) method, we ask the buffer to pull n examples from the
        experience source and store them in the buffer. The sample(batch_size)
        method returns a random sample of the given size from the current buffer
        contents
"""