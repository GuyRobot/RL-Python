import enum
import gym
from gym.utils import seeding
import numpy as np
from RL.RealLifeProblem.Lib import DataHandler as data

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(self, bars_count, commission_percent,
                 reset_on_close, reward_on_close=True, volumes=True):
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_percent, float)
        assert commission_percent >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_percent = commission_percent
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self, prices, offset):
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self):
        # [h, l, c] * bars + position_flag + rel_profit (since open)
        if self.volumes:
            return (4 * self.bars_count + 1 + 1,)

        return (3 * self.bars_count + 1 + 1,)

    def encode(self):
        """
        Convert to numpy array
        Eencodes prices at the current offset into a NumPy array,
        which will be the observation of the agent.

        :return:
        """
        res = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for bar_idx in range(-self.bars_count + 1, 1):
            res[shift] = self._prices.high[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.low[self._offset + bar_idx]
            shift += 1
            res[shift] = self._prices.close[self._offset + bar_idx]
            shift += 1
            if self.volumes:
                res[shift] = self._prices.volume[self._offset + bar_idx]
                shift += 1
        res[shift] = float(self.have_position)
        shift += 1
        if not self.have_position:
            res[shift] = 0.0
        else:
            res[shift] = (self._cur_close() - self.open_price) / self.open_price
        return res

    def _cur_close(self):
        """
        Calculate real close price for the current bar
        This helper method calculates the current bar’s close price. Prices passed to
        the State class have the relative form in respect to open price: the high, low,
        and close components are relative ratios to the open price
        :return:
        """
        open = self._prices.open[self._offset]
        rel_close = self._prices.close[self._offset]
        return open * (1.0 + rel_close)

    def step(self, action):
        """
        Perform one step in our price, adjust offset, check for the end of prices
        and handle position change
        :param action:
        :return:
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self._cur_close()
        # If the agent has decided to buy a share, we change our state and pay the
        # commission. In our state, we assume the instant order execution at the current
        # bar’s close price, which is a simplification on our side, as, normally, order
        # could be executed on a different price, which is called "price slippage".
        if action == Actions.Buy and not self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_percent
        # If we have a position and the agent asks us to close it, we pay commission
        # again, change the done flag if we’re in reset_on_close mode, give a final
        # reward for the whole position, and change our state.
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_percent
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
            self.have_position = False
            self.open_price = 0.0

        self._offset += 1
        prev_close = close
        close = self._cur_close()
        done |= self._offset >= self._prices.close.shape[0] - 1

        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close

        return reward, done


class State1D(State):
    @property
    def shape(self):
        if self.volumes:
            return 6, self.bars_count
        return 5, self.bars_count

    def encode(self):
        res = np.zeros(shape=self.shape, dtype=np.float32)
        ofs = self.bars_count - 1
        res[0] = self._prices.high[self._offset - ofs:self._offset + 1]
        res[1] = self._prices.low[self._offset - ofs:self._offset + 1]
        res[2] = self._prices.close[self._offset - ofs:self._offset + 1]
        if self.volumes:
            res[3] = self._prices.close[self._offset - ofs:self._offset + 1]
            dst = 4
        else:
            dst = 3
        if self.have_position:
            res[dst] = 1.0
            res[dst + 1] = (self._cur_close() - self.open_price) / self.open_price
        return res


class StocksEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, prices, bars_count=DEFAULT_BARS_COUNT,
                 commission=DEFAULT_COMMISSION_PERC, reset_on_close=True,
                 state_1d=False, random_ofs_on_reset=True,
                 reward_on_close=False, volumes=False):
        """

        :param prices: Contains one or more stock prices for one or more instruments
        as a dict, where keys are the instrument’s name and value is a container
        object data.Prices which holds price data arrays.

        :param bars_count: The count of bars that we pass in observation. By default,
        this is 10 bars.

        :param commission: The percentage of the stock price we have to pay to the
        broker on buying and selling the stock. By default, it’s 0.1%.
        reset_on_close: If this parameter is set to True, which it is by default,
        every time the agent asks us to close the existing position (in other
        words, sell a share), we stop the episode. Otherwise, the episode will
        continue until the end of our time series, which is one year of data.

        :param reset_on_close: If the parameter is True (by default), on every
        reset of the environment, the random offset in time series will be chosen.

        :param state_1d: This boolean argument, switches between different
        representations of price data in the observation passed to the agent. If it
        is set to True, observations have a 2D shape, with different price
        components for subsequent bars organized in rows. For example, high
        prices (max price for the bar) are placed on the first row, low prices on
        the second and close prices on the third. This representation is suitable
        for doing 1D convolution on time series, where every row in the data
        has the same meaning as different color planes (red, green, or blue) in
        Atari 2D images. If we set this option to False, we have one single array
        of data with every bar’s components placed together. This organization
        is convenient for fully-connected network architecture. Both
        representations are illustrated in Figure 2.

        :param random_ofs_on_reset: : If the parameter is True (by default), on every
        reset of the environment, the random offset in time series will be chosen

        :param reward_on_close: This Boolean parameter switches between two
        reward schemes discussed above. If it is set to True, the agent will
        receive reward only on the “close” action issue. Otherwise, we’ll give a
        small reward every bar, corresponding to price movement during that
        bar.
        :param volumes: This argument switches on volumes in observations and is
        disabled by default.

        """
        assert isinstance(prices, dict)
        self._prices = prices
        if state_1d:
            self._state = State1D(bars_count, commission, reset_on_close,
                                  reward_on_close, volumes)
        else:
            self._state = State(bars_count, commission, reset_on_close,
                                reward_on_close, volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=self._state.shape)
        self.random_ofs_on_reset = random_ofs_on_reset
        self.seed()

    def reset(self):
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.random_ofs_on_reset:
            offset = self.np_random.choice(prices.high.shape[0] - bars * 10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.encode()

    def step(self, action):
        action = Actions(action)
        reward, done = self._state.step(action)
        obs = self._state.encode()
        info = {'instrument': self._instrument, "offset": self._state._offset}
        return obs, reward, done, info

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def from_dir(cls, data_dir, **kwargs):
        prices = {file: data.load_relative(file)
                  for file in data.price_files(data_dir)}
        return StocksEnv(prices, **kwargs)
