import time
import numpy as np
import argparse
import torch
import gym.wrappers
from RL.DeepQLearning import Wrapper
from RL.Library import DeepQNetModel

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "—model", required=True, default="PongNoFrameskip-v4-best.dat",
                        help="Model file to load")
    parser.add_argument("-e", "—env", default=DEFAULT_ENV_NAME,
                        help="Environment name to use, default="
                             + DEFAULT_ENV_NAME)
    parser.add_argument("-r", "—record", help="Directory to store video recording")
    args = parser.parse_args()

    env = Wrapper.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = DeepQNetModel.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model))

    state = env.reset()
    total_reward = 0.0
    while True:
        start_ts = time.time()
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)

        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        delta = 1 / FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
    print("Total reward: %.2f" % total_reward)

