import sys
import logging

from six import StringIO

import gym
from gym import spaces, envs
import numpy as np
import pandas as pd


class AssetTimeSerieEnv(gym.Env):
    '''gym interface that implements a stock time series'''
    metadata = {'render.modes': ['human', 'ansi']}
    reward_range = (-np.inf, np.inf)

    all_actions = {
        'long_only': {
            0: 'buy',
            1: 'hold',
            2: 'sell',
        },
        'short_only': {
            0: 'short_sell',
            1: 'hold',
            2: 'buy_to_cover',
        },
        'long_short': {
            0: 'buy',
            1: 'hold',
            2: 'sell',
            3: 'short_sell',
            4: 'buy_to_cover',
        },
    }

    def __init__(self, data_df, input_cols, price_col, low_cols, high_cols,
                 with_long=True, with_short=False,
                 date_start=None, date_end=None,
                 reward_type='pct', sort_index=False, std=False):
        super(AssetTimeSerieEnv, self).__init__()

        if with_long and with_short:
            self.action_key = 'long_short'
            position_min = -1
            position_max = 1
        elif with_long and not with_short:
            self.action_key = 'long_only'
            position_min = 0
            position_max = 1
        elif with_short and not with_long:
            self.action_key = 'short_only'
            position_min = -1
            position_max = 0
        else:
            raise Exception('Mode no valid')

        self.df = data_df
        if sort_index:
            self.df = self.df.sort_index()

        self.input_cols = input_cols
        self.price_col = price_col
        self.with_long = with_long
        self.with_short = with_short
        self.reward_type = reward_type
        self.std = std

        if self.std:
            new_colnames = ['%s_std' % col for col in input_cols]
            df_std = (self.df[input_cols] - self.df[input_cols].mean()) / \
                self.df[input_cols].std()
            self.df[new_colnames] = df_std
            self.input_cols = new_colnames

        if date_start is None:
            self.date_start = self.df.iloc[0].name
        else:
            self.date_start = self.df.index[self.df.index >= date_start][0]

        if date_end is None:
            self.date_end = self.df.iloc[-1].name
        else:
            self.date_end = self.df.index[self.df.index <= date_end][-1]

        if self.df.loc[self.date_start:self.date_end].shape[0] < 2:
            raise Exception('At least 2 rows are need')

        self.actions = self.all_actions[self.action_key]
        self.action_space = spaces.Discrete(len(self.actions))
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([position_min] + low_cols),
            high=np.array([position_max] + high_cols),
            dtype=np.float32,
        )

        self.current_date = None
        self.position = 0
        self.acum = 0
        self._iter = None

    def reset(self):
        self.position = 0
        self.buy_price = None
        self.df_iter = self.df.loc[self.date_start:self.date_end].iterrows()

        self.current_date, self.current_data = next(self.df_iter)

        new_status = np.array(
            [0] +  # Current position on asset
            [self.current_data[col] for col in self.input_cols]
        )
        return new_status

    def step(self, action):
        action_str = self.actions[action]
        next_date, next_data = next(self.df_iter)

        # if action_str == 'hold':
        #     # Nothing to do
        #     pass
        # elif action_str == 'buy':
        #     if self.position == 0:
        #         self.position = 1
        #         self.buy_price = self.current_data[self.price_col]
        #     else:
        #         # Not possible
        #         pass
        # elif action_str == 'sell':
        #     if self.position == 1:
        #         self.position = 0
        #         self.buy_price = None
        #     else:
        #         # Not possible
        #         pass
        # elif action_str == 'short_sell':
        #     if self.position == 0:
        #         self.position = -1
        #         self.buy_price = self.current_data[self.price_col]
        #     else:
        #         # not possible
        #         pass
        # elif action_str == 'buy_to_cover':
        #     if self.position == -1:
        #         self.position = 0
        #         self.buy_price = None

        if action_str == 'hold':
            # Nothing to do
            pass
        elif action_str == 'buy':
            self.position = 1
            self.buy_price = self.current_data[self.price_col]
        elif action_str == 'sell':
            if self.position == 1:
                self.position = 0
                self.buy_price = None
            else:
                # Not possible
                pass
        elif action_str == 'short_sell':
            self.position = -1
            self.buy_price = self.current_data[self.price_col]
        elif action_str == 'buy_to_cover':
            if self.position == -1:
                self.position = 0
                self.buy_price = None

        reward = self._compute_reward(next_data)
        self.current_date, self.current_data = next_date, next_data
        done = self.current_date.date() == self.date_end.date()

        new_status = np.array(
            [self.position] +  # Current position on asset
            [self.current_data[col] for col in self.input_cols]
        )
        return new_status, reward, done, {}

    def _compute_reward(self, next_data):
        if self.position == 0:
            reward = 0
        elif self.reward_type == 'price':
            reward = self.position * (
                next_data[self.price_col] - self.current_data[self.price_col])
        elif self.reward_type == 'pct':
                reward = self.position * (next_data[self.price_col] - self.current_data[self.price_col]) / self.current_data[self.price_col]
        else:
            raise Exception('reward_type %s not implemented.' % self.reward_type)

        return reward

    def render(self, mode='human'):
        if mode == 'ansi':
            outfile = StringIO()
        elif mode == 'human':
            outfile = sys.stdout
        elif mode == 'log':
            pass
        else:
            raise Exception('Render mode %s is not supported' % mode)

        render_text_01 = ['date: %s' % self.current_date, 'position: %d' % self.position]
        render_text_02 = ['%s: %f' % (col, self.current_data[col]) for col in self.input_cols]
        render_text_03 = ['%s: %s' % (self.price_col, self.current_data[self.price_col])]
        final_text = ' - '.join(render_text_01 + render_text_02 + render_text_03)

        if mode != 'log':
            outfile.write(final_text)
            outfile.write('\n')
        else:
            logging.debug(final_text)


if __name__ == '__main__':

    # from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines import A2C
    from stable_baselines import DQN

    df = pd.read_csv('~/SPY.csv')
    df.rename(
        {
            'Date': 'date',
            'Open': 'open',
            'Low': 'low',
            'High': 'high',
            'Adj Close': 'close_adj',
            'Close': 'close',
            'Volume': 'volume'
        },
        axis=1,
        inplace=True,
    )
    df['date'] = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)
    # reward_type = 'pct'
    reward_type = 'price'

    input_cols = ['open', 'high', 'low', 'close', 'volume']
    env_train = AssetTimeSerieEnv(
        data_df=df,
        input_cols=input_cols,
        price_col='close_adj',
        low_cols=[0] * len(input_cols),
        high_cols=[np.inf] * len(input_cols),
        with_long=True,
        with_short=True,
        date_start=pd.to_datetime('2015-01-01'),
        date_end=pd.to_datetime('2017-12-31'),
        reward_type=reward_type,
        std=True,
    )
    # initial = env_train.reset()
    env_train = DummyVecEnv([lambda: env_train])

    # model = A2C(MlpPolicy, env, ent_coef=0.1, verbose=1)
    model = DQN(MlpPolicy, env_train, verbose=1)
    model.learn(total_timesteps=500000)

    env_test = AssetTimeSerieEnv(
        data_df=df,
        input_cols=input_cols,
        price_col='close_adj',
        low_cols=[0] * len(input_cols),
        high_cols=[np.inf] * len(input_cols),
        with_long=True,
        with_short=True,
        date_start=pd.to_datetime('2018-01-01'),
        date_end=pd.to_datetime('2018-12-31'),
        reward_type=reward_type,
        std=True,
    )
    env_test = DummyVecEnv([lambda: env_test])

    # Enjoy trained agent
    obs = env_test.reset()
    env_test.render()
    dones = [False]
    if reward_type == 'pct':
        reward_acum = 0
    elif reward_type == 'price':
        reward_acum = 0
    else:
        raise Exception('reward_type %s not supported' % reward_type)
    while not dones[0]:
        action, _states = model.predict(obs)
        print('Action steps=%s' % env_test.envs[0].actions[action[0]])
        obs, rewards, dones, info = env_test.step(action)

        if reward_type == 'pct':
            reward_acum = (1 + reward_acum) * (1 + rewards[0]) - 1
        else:
            reward_acum += rewards[0]

        env_test.render()
        print('reward_acum = %f' % reward_acum)
