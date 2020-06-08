import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        """
        The preceding method is used to obtain the next transition from the
        environment. We sample a random action from the action space and return
        the tuple of the old state, action taken, reward obtained, and the new state.
        The tuple will be used in the training loop later.

        :return: tuple of old_state, action, reward, new_state
        """
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        """
        The next method receives the state of the environment and finds the best
        action to take from this state by taking the action with the largest value that
        we have in the table
        This method will be used two times: first,
        in the test method that plays one episode using our current values table (to
        evaluate our policy quality), and the second, in the method that performs the
        value update to get the value of the next state.

        :param state:
        :return: best_value, best_action
        """
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            # print(action_value)
            # print(self.values.get((state, action)))
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, state, action, reward, next_state):
        """
        We're calculating the Bellman approximation for our state s and action a
        by summing the immediate reward with the discounted value of the next
        state. Then we obtain the previous value of the state and action pair, and
        blend these values together using the learning rate. The result is the new
        approximation for the value of state s and action a, which is stored in our
        table
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return: none
        """
        best_value, _ = self.best_value_and_action(next_state)
        new_val = reward + GAMMA * best_value
        old_val = self.values[(state, action)]
        self.values[(state, action)] = old_val * (1 - ALPHA) + ALPHA * new_val

    def play_episode(self, env):
        """
        Plays one full episode using the provided
        test environment. The action on every step is taken using our current value
        table of Q-values. This method is used to evaluate our current policy to check
        the progress of learning. Note that this method doesn't alter our value table: it
        only uses it to find the best action to take.
        :param env: env to play
        :return: total reward
        """
        total_reward = 0.0
        state = env.reset()
        while True:
            _, best_action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(best_action)
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward


if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment='-q-learning')

    epoch = 0
    best_reward = 0.0
    while True:
        epoch += 1
        state, action, reward, next_state = agent.sample_env()

        agent.value_update(state, action, reward, next_state)

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
            print("Solved! in %d episode" % epoch)
            break
    writer.close()




