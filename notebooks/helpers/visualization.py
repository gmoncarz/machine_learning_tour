from plotly.graph_objs import Candlestick, Figure, Scatter
from plotly.offline import iplot, init_notebook_mode
from types import GeneratorType
import numpy as np


def plot_ohlc_ts(df, x_is_index=True, x_col=None,
        open_col='open', close_col='close', high_col='high', low_col='low',
        title=None, y_axis=None):

    init_notebook_mode()

    data_kwargs = {}
    if x_is_index:
        data_kwargs['x'] = df.index
    if x_col is not None:
        data_kwargs['x'] = df[x_col]
    if open_col:
        data_kwargs['open'] = df[open_col]
    if high_col:
        data_kwargs['high'] = df[high_col]
    if low_col:
        data_kwargs['low'] = df[low_col]
    if close_col:
        data_kwargs['close'] = df[close_col]

    data = [
        Candlestick(**data_kwargs),
    ]

    layout = {}
    if title:
        layout['title'] = title
    if y_axis:
        layout['yaxis'] = {'title': y_axis}

    figure = Figure(data=data, layout=layout)
    iplot(figure)


def plot_return(dates, series, labels=None, accumulate=False,
                scater_params={'opacity': .7}, title=None):
    init_notebook_mode()

    if accumulate:
        series = [(np.array(serie) + 1).cumprod() - 1 for serie in series]

    if not (isinstance(dates, list) or isinstance(dates, tuple) or isinstance(dates, GeneratorType)):
        dates = [dates for date in range(len(series))]

    if not labels:
        labels = [None for _ in range(len(series))]

    data = []
    for serie, date, label in zip(series, dates, labels):
        data_params = scater_params.copy()
        data_params['x'] = date
        data_params['y'] = serie

        if label:
            data_params['name'] = label

        current_data = Scatter(**data_params)
        data.append(current_data)

    layout = {}
    if title:
        layout['title'] = title

    fig = dict(data=data, layout=layout)
    iplot(fig)
