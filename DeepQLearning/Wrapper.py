"""
    Converting individual lives in the game into separate episodes. In
        general, an episode contains all the steps from the beginning of the game
        until the "Game over" screen appears?, which can last for thousands of
        game steps (observations and actions). Usually, in arcade games, the
        player is given several lives, which provide several attempts in the
        game. This transformation splits a full episode into individual small
        episodes for every life that a player has. Not all games support this
        feature (for example, Pong doesn't), but for the supported environments,
        it usually helps to speed up convergence as our episodes become shorter.
        In the beginning of the game, performing a random amount (up to 30) of
        no-op actions. This should stabilize training, but there is no proper
        explanation why it is the case.
    Making an action decision every K steps, where K is usually 4 or 3. On
        intermediate frames, the chosen action is simply repeated. This allows
        training to speed up significantly, as processing every frame with a
        neural network is quite a demanding operation, but the difference
        between consequent frames is usually minor.
        Taking the maximum of every pixel in the last two frames and using it
        as an observation. Some Atari games have a flickering effect, which is
        due to the platform's limitation (Atari has a limited amount of sprites
        that can be shown on a single frame). For a human eye, such quick
        changes are not visible, but they can confuse neural networks.
    Pressing FIRE in the beginning of the game. Some games (including
        Pong and Breakout) require a user to press the FIRE button to start the
        game. In theory, it's possible for a neural network to learn to press FIRE
        itself, but it will require much more episodes to be played. So, we press
        FIRE in the wrapper.
    Scaling every frame down from 210 × 160, with three color frames, into
        a single-color 84 × 84 image. Different approaches are possible. For
        example, the DeepMind paper describes this transformation as taking
        the Y-color channel from the YCbCr color space and then rescaling the
        full image to an 84 × 84 resolution. Some other researchers do grayscale
        transformation, cropping non-relevant parts of the image and then
        scaling down. In the Baselines repository (and in the following example
        code), the latter approach is used.
    Stacking several (usually four) subsequent frames together to give the
        network the information about the dynamics of the game's objects.
    Clipping the reward to −1, 0, and 1 values. The obtained score can vary
        wildly among the games. For example, in Pong you get a score of 1 for
        every ball that your opponent passes behind you. However, in some
        games, like KungFu, you get a reward of 100 for every enemy killed.
        This spread in reward values makes our loss have completely different
        scales between the games, which makes it harder to find common
        hyperparameters for a set of games. To fix this, reward just gets clipped
        to the range [−1...1].
    Converting observations from unsigned bytes to float32 values. The
        screen obtained from the emulator is encoded as a tensor of bytes with
        values from 0 to 255, which is not the best representation for a neural
        network. So, we need to convert the image into floats and rescale the
        values to the range [0.0…1.0].
"""

import cv2
import gym
import gym.spaces
import numpy as np
import collections


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """
        Return only every skip frame
        :param env:
        :param skip:
        """
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        info = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def _reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    """
    The goal of this wrapper is to convert input observations from the emulator,
    which normally has a resolution of 210 × 160 pixels with RGB color
    channels, to a grayscale 84 × 84 image. It does this using a colorimetric
    grayscale conversion (which is closer to human color perception than a
    simple averaging of color channels), resizing the image and cropping the top
    and bottom parts of the result.
    """
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1),
                                                dtype=np.uint8)

    def observation(self, observation):
        return ProcessFrame84.process(observation)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, (210, 160, 3)).astype(np.float32)
        else:
            assert False, "Unknown resolution"
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 \
              + img[:, :, 2] * .114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    """
    This class creates a stack of subsequent frames along the first dimension and
    returns them as an observation. The purpose is to give the network an idea
    about the dynamics of the objects, such as the speed and direction of the ball
    in Pong or how enemies are moving. This is very important information,
    which it is not possible to obtain from a single image.
    """
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0),
                                                dtype=dtype)

    def reset(self, **kwargs):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    """
        This simple wrapper changes the shape of the observation from HWC to the
        CHW format required by PyTorch. The input shape of the tensor has a color
        channel as the last dimension, but PyTorch's convolution layers assume the
        color channel to be the first dimension.

    """
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=1.0,
                                                shape=(old_shape[-1],
                                                       old_shape[0],
                                                       old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)