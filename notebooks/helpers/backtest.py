import logging
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
from toolz.itertoolz import sliding_window
import empyrical
import tensorflow as tf

from helpers.machine_learning import train_model, train_tensorflow_model, get_trailing_df
from helpers.machine_learning import train_sequential_tensorflow_model


def train_model_and_backtest_regressor(df, x_vars, y_var,
        buy_price_col, sell_price_col,
        model_class, model_params,
        backtest_start=None, backtest_end=None,
        model_update_frequency='M',
        train_history_period=relativedelta(months=1),
        col_date='date',
        col_date_shift=None,
        ignore_last_x_training_items=0,
        reference_price_col='close_adj',
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

    periods = pd.date_range(backtest_start, backtest_end, freq=model_update_frequency)
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

        # df_train = df_train[~df_train[x_vars].isnull()]
        df_train = df_train[~df_train[x_vars].isnull().any(axis=1)]

        model = train_model(df_train, x_vars, y_var, model_class, model_params)
        pred = model.predict(df_test[x_vars])

        go_long = df_test[reference_price_col] < pred
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


def train_tensorflow_model_and_backtest_regressor(df, x_vars, y_var,
        buy_price_col, sell_price_col,
        model_class, model_params={},
        backtest_start=None, backtest_end=None,
        model_update_frequency='M',
        train_history_period=relativedelta(months=1),
        col_date='date',
        col_date_shift=None,
        ignore_last_x_training_items=0,
        reference_price_col='close_adj',
        fit_kwargs={},
        seed=None,
        tf_config=None,
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

    periods = pd.date_range(backtest_start, backtest_end, freq=model_update_frequency)
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

        # df_train = df_train[~df_train[x_vars].isnull()]
        df_train = df_train[~df_train[x_vars].isnull().any(axis=1)]

        # tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        graph = tf.Graph()

        session = tf.Session(config=tf_config, graph=graph)
        tf.keras.backend.set_session(session)

        with session.as_default():
            with session.graph.as_default():

                if seed:
                    tf.random.set_random_seed(seed)
                    np.random.seed(seed)

                model = train_tensorflow_model(
                    df_train,
                    x_vars,
                    y_var,
                    model_class=model_class,
                    model_params=model_params,
                    fit_params=fit_kwargs,
                )

                pred = model.predict(df_test[x_vars])
                pred = pd.Series(pred.reshape([-1]), index=df_test.index)

        go_long = df_test[reference_price_col] < pred
        df_trans = pd.DataFrame({
            'date': df_test.date,
            'open_price': df_test[buy_price_col],
            'close_price': df_test[sell_price_col],
            'ml_pred': pred,
            'ref_price': df_test[reference_price_col],
            'real_y': df_test[y_var],
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


def _build_sequential_timeseries(df, x_vars, y_vars, time_steps):
    dim_batch = df.shape[0]
    dim_features = len(x_vars)

    mat_x = np.full((dim_batch, time_steps, dim_features), np.nan)
    mat_y = np.full((dim_batch,), np.nan)

    for index in range(time_steps-1, dim_batch):
        mat_x[index] = df[x_vars].values[index-time_steps+1:index+1]
        mat_y[index] = df[y_vars].values[index]

    return mat_x, mat_y


def train_tensorflow_sequencial_model_and_backtest_regressor(
        df,
        x_vars,
        y_var,
        time_steps,
        buy_price_col,
        sell_price_col,
        model_class, model_params={},
        backtest_start=None, backtest_end=None,
        model_update_frequency='M',
        train_history_period=relativedelta(months=1),
        col_date='date',
        col_date_shift=None,
        ignore_last_x_training_items=0,
        reference_price_col='close_adj',
        fit_kwargs={},
        seed=None,
        tf_config=None,
        with_summary=False,
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

    periods = pd.date_range(backtest_start, backtest_end, freq=model_update_frequency)
    periods = periods.insert(0, backtest_start)
    periods = periods.insert(periods.shape[-1], backtest_end)
    periods = periods.drop_duplicates()

    logger.debug('%d periods to backtest: %s.', len(periods), list(map(str, periods.date)))

    # Build the sequantial timeserie.
    # Note: Doing this here is not the most memory-efficient, but it is CPU
    # efficient, because all the sequence is computed once. For this problem,
    # memory is not an issue.
    mat_full_x, mat_full_y = _build_sequential_timeseries(
        df, x_vars, y_var, time_steps)

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

        # Locate the numpy training index
        index_train_from = df.index.get_loc(df_train.index[0])
        index_train_to = df.index.get_loc(df_train.index[-1])
        index_test_from = df.index.get_loc(df_test.index[0])
        index_test_to = df.index.get_loc(df_test.index[-1])

        # Build the final train and test array
        train_x = mat_full_x[index_train_from:index_train_to+1]
        train_y = mat_full_y[index_train_from:index_train_to+1]

        test_x = mat_full_x[index_test_from:index_test_to+1]
        test_y = mat_full_y[index_test_from:index_test_to+1]

        # # df_train = df_train[~df_train[x_vars].isnull()]
        # df_train = df_train[~df_train[x_vars].isnull().any(axis=1)]

        # tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        graph = tf.Graph()

        session = tf.Session(config=tf_config, graph=graph)
        tf.keras.backend.set_session(session)

        with session.as_default():
            with session.graph.as_default():

                if seed:
                    tf.random.set_random_seed(seed)
                    np.random.seed(seed)

                model = train_sequential_tensorflow_model(
                    train_x,
                    train_y,
                    model_class=model_class,
                    model_params=model_params,
                    fit_params=fit_kwargs,
                )

                if with_summary:
                    logging.debug(model.summary())

                pred = model.predict(test_x)
                pred = pd.Series(pred.reshape([-1]), index=df_test.index)

        go_long = df_test[reference_price_col] < pred
        df_trans = pd.DataFrame({
            'date': df_test.date,
            'open_price': df_test[buy_price_col],
            'close_price': df_test[sell_price_col],
            'ml_pred': pred,
            'ref_price': df_test[reference_price_col],
            'real_y': df_test[y_var],
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


def train_model_and_backtest_bool_classifier(df, x_vars, y_var,
        buy_price_col, sell_price_col,
        model_class, model_params,
        backtest_start=None, backtest_end=None,
        model_update_frequency='M',
        train_history_period=relativedelta(months=1),
        col_date='date',
        col_date_shift=None,
        ignore_last_x_training_items=0,
        long_class=1, long_interval=pd.Interval(left=0.5, right=1, closed='both'),
        short_class=-1, short_interval=pd.Interval(left=0.5, right=1, closed='right'),
        add_missing_class_on_uniclass_model=False,
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

    periods = pd.date_range(backtest_start, backtest_end, freq=model_update_frequency)
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

        if add_missing_class_on_uniclass_model:
            unique = df_train[y_var].unique()
            if unique.shape[0] == 1:
                # This is a hack. The training dataset contains a single class,
                # while the model requieres at least two. Add the new class
                # randomly.
                missing = {long_class, short_class} - set(df_train[y_var].unique())

                if missing:
                    logger.warn('Adding randomly the following missing classes: %s',
                                missing)

                for item in missing:
                    # df_train.sample(1)[y_var] = item
                    df_train.loc[df_train.sample(1).index, y_var] = item

        logger.info('Training dataset is between %s and %s.',
                     df_train.date.min().date(), df_train.date.max().date())
        # Get the testing dataset
        if period_end == periods[-1]:
            df_test = df[(df.date>=period_start) & (df.date<=period_end)]
        else:
            df_test = df[(df.date>=period_start) & (df.date<period_end)]

        # Train the model
        model = train_model(df_train, x_vars, y_var, model_class, model_params)

        # Get long and short classes
        long_class_pos = np.where(model.classes_ == long_class)[0][0]
        short_class_pos = np.where(model.classes_ == short_class)[0][0]

        # Get probability prediction
        pred_proba = model.predict_proba(df_test[x_vars])
        long_pred_proba = pred_proba[:, long_class_pos]
        short_pred_proba = pred_proba[:, short_class_pos]

        go_long = [val in long_interval for val in long_pred_proba]
        go_short = [val in short_interval for val in short_pred_proba]
        df_trans = pd.DataFrame({
            'date': df_test.date,
            'open_price': df_test[buy_price_col],
            'close_price': df_test[sell_price_col],
            'go_long': go_long,
            'go_short': go_short,
        })
        df_trans['go_long'] = df_trans['go_long'].astype(int)
        df_trans['go_short'] = df_trans['go_short'].astype(int)

        df_trans['action'] = df_trans.go_long - df_trans.go_short # -1 for short ; 1 for long
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
