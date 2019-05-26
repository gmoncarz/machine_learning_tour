#!/usr/bin/env python

import logging
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np

from backtest import RLBacktester
from env_asset_by_ts import AssetTimeSerieEnv

from stable_baselines import DQN
import stable_baselines.deepq.policies

if __name__ == '__main__':
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
        reward_type='pct',
    )

    backtest = backtester.train_and_backtest(
        pd.to_datetime('2018-01-01'),
        pd.to_datetime('2018-12-31'),
    )

    # import ipdb; ipdb.set_trace()
    pass
