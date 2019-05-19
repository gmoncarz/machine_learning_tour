#!/usr/bin/env python

'''
This library helps to backtest the performance of a time series using
reinforcement learning models.

It uses an environment that implements the gym inteface and works with
any reinforcement learning model that implements of the baseline_stable
interface.

All environments and reinforcement learning model and parameters are adjusted
throught this library params.
'''

import logging
import os
from dateutil.relativedelta import relativedelta
import re

import pandas as pd
from toolz.itertoolz import sliding_window

from stable_baselines.common.vec_env import DummyVecEnv


class RLBacktester:
    def __init__(self, rl_env_train_class, rl_env_test_class,
                 rl_env_train_params, rl_env_test_params,
                 env_date_start_param, env_date_end_param,
                 rl_class, rl_params={},
                 rl_learn_params={}, rl_predict_params={},
                 model_path='./', save_model=False,
                 model_filename_prefix='',
                 model_update_frequency='M',
                 train_history_period=relativedelta(months=1),
                 ):
        '''Backtester constructor

        :param rl_env_train_class: class to instanciate a training RL environment.
        :type rl_env_train_class: type
        :param rl_env_test_class: class to instanciate a testing RL environment.
        :type rl_env_train_class: type
        :param rl_env_train_params: parameters to instanciate rl_env_train_class.
          It will be called as rl_env_train_class(**rl_env_train_params)
        :type rl_env_train_params: dict
        :param rl_env_test_params: parameters to instanciate rl_env_train_class.
          It will be called as rl_env_test_class(**rl_env_test_params)
        :type rl_env_test_params: dict
        :param env_date_start_param: The testing and training dates changes
            during the backtest. For that reason the environment has to work
            during a period. This param specify what is the environment
            parameter that set the environment starting date. For example, in
            `AssetTimeSerieEnv` the param `date_start`.
        :type env_date_start_param: str
        :param env_date_end_param: Same that env_date_start_param, but for
            end date. For `AssetTimeSerieEnv` the param is `date_end`.
        :type env_date_end_param: str
        :param rl_class: Reinforcement learnring class that implements the
            stable_baseline interface
        :type rl_class: type
        :param rl_params: Reinforcement learning parametersr to call rl_class.
        It will be instanciated like rl_class(**rl_params)
        :param rl_params: dict
        :param rl_learn_params: parameters to call the stable_baselines
        learn method. For example, it will be called as:
            model.learn(**rl_learn_params)
        :type rl_learn_params: dict
        :param rl_predict_params: Parameters to call the stable_baselines
        predict method. For example, it will be called as:
            model.predict(**rl_predict_params)
        :type rl_predict_params: dict
        :param save_model: set if the model is persisted in the file system.
        :bool save_model: bool
        :param model_path: Set the path to save the model
        :type model_path: str
        :param model_filename_prefix: prefix filename to save the model
        :type model_filename_prefix: str
        :param model_update_frequency: Specify how frequent the RL model
            is updated. Check
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        :type model_update_frequency: str
        :param train_history_period: Set how much history is used to train
            a model.
        :type train_history_period: dateutil.relativedelta.relativedelta
        '''
        self.rl_env_train_class = rl_env_train_class
        self.rl_env_test_class = rl_env_test_class
        self.rl_env_train_params = rl_env_train_params
        self.rl_env_test_params = rl_env_test_params
        self.env_date_start_param = env_date_start_param
        self.env_date_end_param = env_date_end_param

        self.rl_class = rl_class
        self.rl_params = rl_params
        self.rl_learn_params = rl_learn_params
        self.rl_predict_params = rl_predict_params

        self.model_path = model_path
        self.save_model = save_model
        self.model_filename_prefix = model_filename_prefix
        self.model_update_frequency = model_update_frequency
        self.train_history_period = train_history_period

        self._filename_regexp = re.compile('[^a-zA-Z0-9.\-_]')

    def train_and_backtest(self, backtest_start, backtest_end, force_train=False,
                           model_filename_prefix=''):
        logger = logging.getLogger(__name__)

        periods = pd.date_range(backtest_start, backtest_end, freq=self.model_update_frequency)
        periods = periods.insert(0, backtest_start)
        periods = periods.insert(periods.shape[-1], backtest_end)
        periods = periods.drop_duplicates()

        logger.debug('%d periods to backtest: %s.', len(periods), list(map(str, periods.date)))

        backtest_trans_list = []

        for period_start, period_end in sliding_window(2, periods):
            logger.info('Training a model to be tested between %s and %s.',
                        period_start.date(), period_end.date())

            date_train_end = period_start - relativedelta(days=1)
            date_train_start = date_train_end - self.train_history_period

            env_train_params = self.rl_env_train_params.copy()
            env_train_params[self.env_date_start_param] = date_train_start
            env_train_params[self.env_date_end_param] = date_train_end

            env_train = self.rl_env_train_class(**env_train_params)
            env_train = DummyVecEnv([lambda: env_train])

            model = None
            if not force_train:
                model = self._load_model(period_start, period_end, model_filename_prefix)

            if model is None:
                logger.debug('Training model from %s to %s', date_train_start.date(),
                             date_train_end.date())

                model = self.rl_class(env=env_train, **self.rl_params)
                model.learn(**self.rl_learn_params)
                if self.save_model:
                    filename = self._get_filename(
                        period_start,
                        period_end,
                        model_filename_prefix
                    )
                    model.save(filename)
                    logger.info('Model saved: %s' % filename)

            if model is None:
                raise Exception("Model couldn' be trained")

            current_backtest = self._backtest_model(model, period_start, period_end)
            backtest_trans_list.append(current_backtest)

        # Summarize backtest
        df_backtest = pd.concat(backtest_trans_list).reset_index(drop=True)

        return df_backtest

    def _backtest_model(self, model, date_start, date_end):
        # Backtest
        logger = logging.getLogger(__name__)

        env_test_params = self.rl_env_test_params.copy()
        env_test_params[self.env_date_start_param] = date_start
        env_test_params[self.env_date_end_param] = date_end

        env_test = self.rl_env_test_class(**env_test_params)
        env_test = DummyVecEnv([lambda: env_test])

        obs = env_test.reset()
        dones = [False]

        backtest_data_list = []
        while not dones[0]:
            try:
                current_date = env_test.envs[0].get_date()
            except Exception:
                # This is not a mandatory method to be implemented by gym
                current_date = None

            action, _states = model.predict(obs, **self.rl_predict_params)
            # logger.debug('Action steps=%s' % env_test.envs[0].actions[action[0]])
            obs, rewards, dones, info = env_test.step(action)

            try:
                action_descr = env_test.envs[0].get_action_str(action[0])
            except Exception:
                # This is not a mandatory method to be implemented by gym
                action_descr = None

            backtest_data = [current_date, action[0], action_descr, rewards[0]]
            backtest_data_list.append(backtest_data)

        df_opers = pd.DataFrame(
            backtest_data_list,
            columns=['date', 'action', 'action_descr', 'reward']
        )
        df_opers['date'] = pd.to_datetime(df_opers['date'])

        return df_opers

    def _load_model(self, period_start, period_end, custom_prefix=None):
        logger = logging.getLogger(__name__)

        filename = self._get_filename(period_start, period_end, custom_prefix)

        try:
            model = self.rl_class.load(filename)
            logger.debug('Model %s read successfully.', filename)
        except Exception:
            logger.debug("Model %s couldn't be read.", filename)
            model = None

        return model

    def _get_filename(self, period_start, period_end, custom_prefix):
        prefix = custom_prefix if custom_prefix else self.model_filename_prefix

        filename = '%s_%s_%s_%s_%s.pickle' % (prefix, self.model_update_frequency,
            self.train_history_period, period_start.date(), period_end.date())

        filename = self._filename_regexp.sub('_', filename)

        fullpath = os.path.join(self.model_path, filename)

        return fullpath


if __name__ == '__main__':
    import numpy as np
    import stable_baselines.deepq.policies
    from stable_baselines import DQN

    from env_asset_by_ts import AssetTimeSerieEnv

    logging_format = '%(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logging_format)

    df = pd.read_csv('../../data/SPY_postprocess_adj.csv.gz')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', drop=False, inplace=True)

    rl_env_train_class = AssetTimeSerieEnv
    rl_env_test_class = AssetTimeSerieEnv

    input_cols = ['open_adj', 'low_adj', 'high_adj', 'close_adj', 'volume']

    rl_class = DQN
    rl_params = {
        'policy': stable_baselines.deepq.policies.MlpPolicy,
        'verbose': 1,
        'policy_kwargs': {
            'layers': [16, 16]
        }
    }

    rl_learn_params = {
        'total_timesteps': 50000,
        'seed': 100,
    }

    rl_predict_params = {
        'deterministic': True,
    }

    rl_env_base_params = {
        'data_df': df,
        'input_cols': input_cols,
        'price_col': 'close_adj',
        'low_cols': [0] * len(input_cols),
        'high_cols': [np.inf] * len(input_cols),
        'with_long': True,
        'with_short': True,
        'std': True,
    }

    rl_env_train_custom_params = {'reward_type': 'price'}
    rl_env_test_custom_params = {'reward_type': 'pct'}

    rl_env_train_params = {**rl_env_base_params, **rl_env_train_custom_params}
    rl_env_test_params = {**rl_env_base_params, **rl_env_test_custom_params}

    backtester = RLBacktester(
        rl_env_train_class=rl_env_train_class,
        rl_env_test_class=rl_env_test_class,
        rl_env_train_params=rl_env_train_params,
        rl_env_test_params=rl_env_test_params,
        env_date_start_param='date_start',
        env_date_end_param='date_end',
        rl_class=rl_class,
        rl_params=rl_params,
        rl_learn_params=rl_learn_params,
        rl_predict_params=rl_predict_params,
        model_update_frequency='M',
        train_history_period=relativedelta(months=6),

        save_model=True,
        model_path='./models/tmp1/',
        model_filename_prefix='tmp1',
    )

    backtest = backtester.train_and_backtest(
        pd.to_datetime('2018-01-01'),
        pd.to_datetime('2018-12-31'),
    )

    pass
