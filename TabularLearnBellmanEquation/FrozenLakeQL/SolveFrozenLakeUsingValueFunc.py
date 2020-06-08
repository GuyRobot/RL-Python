"""
    The central data structures in this example are as follows:
    Reward table: A dictionary with the composite key "source state" +
        "action" + "target state". The value is obtained from the immediate
        reward.
    Transitions table: A dictionary keeping counters of the experienced
        transitions. The key is the composite "state" + "action" and the value is
        another dictionary that maps the target state into a count of times that
        we've seen it. For example, if in state 0 we execute action 1 ten times,
        after three times it leads us to state 4 and after seven times to state 5.
        Entry with the key (0, 1) in this table will be a dict {4: 3, 5: 7}. We
        use this table to estimate the probabilities of our transitions.
    Value table: A dictionary that maps a state into the calculated value of
        this state.
    The overall logic of our code is simple: in the loop, we play 100 random
    steps from the environment, populating the reward and transition tables. After
    those 100 steps, we perform a value iteration loop over all states, updating
    our value table. Then we play several full episodes to check our
    improvements using the updated value table. If the average reward for those
    test episodes is above the 0.8 boundary, then we stop training. During test
    episodes, we also update our reward and transition tables to use all data from
    the environment.

        Graph:
            Play random to get experience (construct 2 dict tables above - transition and reward)
            Value Iteration to construct values table
            Play with environment:
                Play episode:
                    Choose the best action using max value function (fun select_action)
                        In select action calculate values using dict values has saved
                    Take the reward from best action
                    Update 3 tables dict

            Repeat util solve!
"""
import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        # {(0, 0, 0): 0.0, (0, 3, 0): 0.0, (0, 1, 1): 0.0, (1, 0, 0): 0.0,...}
        # (source_state, action, target_state) : reward
        self.rewards = collections.defaultdict(float)
        # {(0, 1): Counter({4: 146, 0: 146, 1: 124}), (4, 3): Counter({0: 31, 4: 27, 5: 22}), ...}
        # (state, action) : count num time exec {state1: num time exist, state2: num time exist, ...}
        self.transitions = collections.defaultdict(collections.Counter)
        # {4: 0.09031095088988002, 0: 0.07002148262932646, 1: 0.06382471255154518, 5: 0.0, 2: 0.07480238713555348,
        # 3: 0.056943860481139336, ...}
        # state: calc values
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        """
        This function is used to gather random experience from the environment and
        update reward and transition tables.
        :param count:
        :return:
        """
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transitions[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        """
            1. We extract transition counters for the given state and action from the
                transition table. Counters in this table have a form of dict, with target
                states as key and a count of experienced transitions as value. We sum all
                counters to obtain the total count of times we've executed the action
                from the state. We will use this total value later to go from an individual
                counter to probability.
            2. Then we iterate every target state that our action has landed on and
                calculate its contribution into the total action value using the Bellman
                equation @see Theory.py. This contribution equals to immediate reward plus discounted
                value for the target state. We multiply this sum to the probability of this
                transition and add the result to the final action value.
            See images/Q_learning_transitions

        """
        target_counts = self.transitions[(state, action)]
        total = sum(target_counts.values())
        action_values = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_values += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_values

    def select_action(self, state):
        """
        iterates over all possible
        actions in the environment and calculates value for every action. The action
        with the largest value wins and is returned as the action to take.
        :param state:
        :return:
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        """
        The play_episode function uses select_action to find the best action to
        take and plays one full episode using the provided environment. This
        function is used to play test episodes,

        Loop over states
        accumulating reward for one episode:

        :param env:
        :return:
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transitions[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        """
        loop over all states in the environment, then for every state we calculate the
        values for the states reachable from it, obtaining candidates for the value of
        the state. Then we update the value of our current state with the maximum
        value of the action available from the state

        :return:
        """
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action) for action in
                            range(self.env.action_space.n)]
            self.values[state] = max(state_values)


def run_main(agent, comment="-v-learning"):
    test_env = gym.make(ENV_NAME)
    writer = SummaryWriter(comment=comment)

    epoch = 0
    best_reward = 0.0
    while True:
        epoch += 1
        """
            First, we perform 100 random steps to fill our reward and transition
                tables with fresh data and then we run value iteration over all states. The rest
                of the code plays test episodes using the value table as our policy, then writes
                data into TensorBoard, tracks the best average reward, and checks for the
                training loop stop condition.
        """
        agent.play_n_random_steps(100)
        agent.value_iteration()
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar("reward", reward, epoch)
        if reward > best_reward:
            print("Best reward updated: %.3f -> %.3f"
                  % (best_reward, reward))
            best_reward = reward
        if reward > 0.8:
            print("Solved in %d iterations!" % epoch)
            break
    writer.close()


if __name__ == '__main__':
    run_main(agent=Agent())
