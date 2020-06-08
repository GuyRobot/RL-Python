import random


class Agent:
    def __init__(self):
        self.total_reward = .0

    def step(self, env):
        """
        The step function take environment (env) as argument and perform
        following actions:
            Observe the environment
            Make a decision about the action to take based on the observations
            Submit the action to the environment
            Get the reward for the current step
        :param env: environment instance
        :return:
        """
        current_obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward
