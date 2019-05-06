import logging
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from toolz.itertoolz import sliding_window
import empyrical

from helpers.machine_learning import train_model, get_trailing_df


def train_model_and_backtest_regressor(df, x_vars, y_var,
        buy_price_col, sell_price_col,
        model_class, model_params,
        backtest_start=None, backtest_end=None,
        model_update_frequency='M',
        train_history_period=relativedelta(months=1),
        col_date='date',
        col_date_shift=None,
        ignore_last_x_training_items=0,
        **kwargs):

    logger = logging.getLogger(__name__)

    dates = sorted(df[col_date])
    if backtest_start is None:
        backtest_start = pd.to_datetime(dates[1])
    else:
        backtest_start = pd.to_datetime(backtest_start)

    if backtest_end is None:
        backtest_end = pd.to_datetime(dates[-1])
    else:
        backtest_end = pd.to_datetime(backtest_end)

    if col_date_shift is None:
        col_date_shift = col_date

    periods = pd.date_range(backtest_start, backtest_end, freq='M')
    periods = periods.insert(0, backtest_start)
    periods = periods.insert(periods.shape[-1], backtest_end)
    periods = periods.drop_duplicates()

    logger.debug('%d periods to backtest: %s.', len(periods), list(map(str, periods.date)))
    pass

    backtest_trans_list = []
    for period_start, period_end in sliding_window(2, periods):
        logger.info('Training a model to be tested between %s and %s.', period_start.date(), period_end.date())

        # get the training dataset
        df_train = get_trailing_df(
            df,
            period_start,
            train_history_period,
            date_col=col_date,
            date_shift_col=col_date_shift,
        )
        if ignore_last_x_training_items:
            train_index = df_train[col_date].sort_values().index[:-ignore_last_x_training_items]
            df_train = df_train.loc[train_index]

        logger.info('Training dataset is between %s and %s.',
                     df_train.date.min().date(), df_train.date.max().date())
        # Get the testing dataset
        if period_end == periods[-1]:
            df_test = df[(df.date>=period_start) & (df.date<=period_end)]
        else:
            df_test = df[(df.date>=period_start) & (df.date<period_end)]


        model = train_model(df_train, x_vars, y_var, model_class, model_params)
        pred = model.predict(df_test[x_vars])

        go_long = df_test['close_adj'] < pred
        df_trans = pd.DataFrame({
            'date': df_test.date,
            'open_price': df_test[buy_price_col],
            'close_price': df_test[sell_price_col],
            'go_long': go_long.astype(int),
            'go_short': (~go_long).astype(int),
            'action': go_long.astype(int) * 2 - 1 # -1 for short ; 1 for long
        })
        df_trans['ret_long'] = ((df_trans.close_price - df_trans.open_price) / df_trans.open_price)
        df_trans['ret_short'] = ((df_trans.open_price - df_trans.close_price) / df_trans.close_price)
        df_trans['ret'] = df_trans.go_long * df_trans.ret_long + df_trans.go_short * df_trans.ret_short
        df_trans['benchmark_ret'] = df_trans.ret_long

        backtest_trans_list.append(df_trans)

    df_all_trans = pd.concat(backtest_trans_list)

    return df_all_trans


def _get_backtest_performance_metrics(ret, benchmark_ret):
    metrics = {
        'alpha': empyrical.alpha(ret, benchmark_ret),
        'beta': empyrical.beta(ret, benchmark_ret),
        'return': empyrical.cum_returns_final(ret),
        'cagr': empyrical.cagr(ret),
        'sharpe': empyrical.sharpe_ratio(ret),
        'max_drawdown': empyrical.max_drawdown(ret),
        'var': empyrical.value_at_risk(ret),
        'volatility': empyrical.annual_volatility(ret),
    }

    return metrics


def get_backtest_performance_metrics(ret, benchmark_ret, with_benchmark=False,
                                     with_delta=False):
    metrics_main = _get_backtest_performance_metrics(ret, benchmark_ret)

    if with_benchmark:
        metrics_benchmark = _get_backtest_performance_metrics(benchmark_ret, benchmark_ret)
        ret = pd.DataFrame(
            {'main': metrics_main, 'benchmark': metrics_benchmark},
            columns=['main', 'benchmark'],
        )

        if with_delta:
            delta = ret.main - ret.benchmark
            delta.alpha = np.nan
            delta.beta = np.nan
            ret['delta'] = delta
    else:
        ret = pd.DataFrame({'main': metrics_main,})

    return ret
