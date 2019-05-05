import pandas as pd


def train_model(df_train, x_vars, y_var, model_class, model_params):
    model = model_class(**model_params)
    model.fit(df_train[x_vars], df_train[y_var])

    return model


def get_trailing_df(df, ref_date, delta_time, date_col='date', date_shift_col=None):
    ref_date = pd.to_datetime(ref_date)

    if date_shift_col is None:
        date_shift_col = date_col

    train_df = df[
        (df[date_shift_col] < ref_date) &
        (df[date_col] >= (ref_date - delta_time))
    ]

    return train_df
