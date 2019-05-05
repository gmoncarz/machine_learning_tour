import pandas as pd


def read_quote_dataset(filename):
    df = pd.read_csv(filename)

    df['date'] = pd.to_datetime(df.date)
    df.sort_values('date', inplace=True)

    return df


def preprocess_quotes(df, vars_to_shift=['close_adj'], shift_periods=[1],
                      vars_for_return = ['close_adj'], return_periods = [1],
                      shift_date=False, col_date='date'):
    df_shift = pd.concat(
        (
            df[vars_to_shift].shift(-shift_period).rename(
                {varname: '%s_shift_%d' % (varname, shift_period) for varname in vars_to_shift},
                axis=1
            )
            for shift_period in shift_periods),
        axis=1,
    )

    if shift_date:
        df_shift_date = pd.concat(
            (
                df[[col_date]].shift(-shift_period).rename(
                    {col_date: '%s_shift_%d' % (col_date, shift_period)},
                    axis=1,
                )
                for shift_period in shift_periods
            ),
            axis=1,
        )
    else:
        df_shift_date = pd.DataFrame()

    df = pd.concat([df, df_shift, df_shift_date], axis=1)

    df_ret = pd.concat((
        ((df[vars_for_return].shift(-return_period) - df[vars_for_return])/df[vars_for_return]).rename(
            {varname: '%s_ret_%d' % (varname, return_period) for varname in vars_for_return},
            axis=1
        )
        for return_period in return_periods),
    axis=1,
    )
    df = pd.concat([df, df_ret], axis=1)

    return df
