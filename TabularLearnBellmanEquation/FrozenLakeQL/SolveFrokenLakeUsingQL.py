"""
    Almost same as Bellman
"""

import gym
import collections
from tensorboardX import SummaryWriter
from RL.TabularLearnBellmanEquation.FrozenLakeQL.SolveFrozenLakeUsingValueFunc import Agent as BellmanAgent, run_main

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent(BellmanAgent):
    def __init__(self):
        """
        Almost same just modify self.values key to (state, action)
        """
        super(Agent, self).__init__()

    def value_iteration(self):
        """
        . For the given state and action, it needs
            to calculate the value of this action using statistics about target states that
            we've reached with the action. To calculate this value, we use the Bellman
            equation and our counters, which allow us to approximate the probability of
            the target state. However, in Bellman's equation we have the value of the
            state and now we need to calculate it differently. Before, we had it stored in
            the value table (as we approximated the value of states), so we just took it
            from this table. We can't do this anymore, so we have to call the
            select_action method, which will choose for us the action with the largest
            Q-value, and then we take this Q-value as the value of the target state. Of
            course, we can implement another function which could calculate for us this
            value of state, but select_action does almost everything we need, so we
            will reuse it here.
            There is another piece of this example
        :return:
        """
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transitions[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state, action, tgt_state)]
                    best_action = self.select_action(tgt_state)
                    action_value += (count / total) * \
                                    (reward + GAMMA * self.values[(tgt_state, best_action)])
                self.values[(state, action)] = action_value

    def select_action(self, state):
        """
        Almost same thing as Bellman
        :param state: env state
        :return: best action
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action


if __name__ == '__main__':
    run_main(agent=Agent(), comment="-q-learning")




