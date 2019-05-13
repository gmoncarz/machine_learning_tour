def get_x_vars_classifier(columns):
    x_vars_all = list(filter(lambda varname:
        ('_adj' in varname or 'volume' in varname) and
            not '_shift_' in varname and
            not '_std' in varname and
            not '_norm' in varname and
            not '_ret_' in varname,
        columns,
    ))

    x_vars_slope = list(filter(lambda var: var.startswith('slope_'), x_vars_all))
    x_vars_sma = list(filter(lambda var: var.startswith('sma_'), x_vars_all))
    x_vars_ema = list(filter(lambda var: var.startswith('ema_'), x_vars_all))
    x_vars_lagged = list(filter(lambda var: var.startswith('lag_'), x_vars_all))
    x_vars_ratio_close_adj = list(filter(lambda var: var.startswith('ratio_close_adj_'), x_vars_all))
    x_vars_ratio_volume = list(filter(lambda var: var.startswith('ratio_volume_'), x_vars_all))

    ret = {
        'all': x_vars_all,
        'slope': x_vars_slope,
        'sma': x_vars_sma,
        'ema': x_vars_ema,
        'lagged': x_vars_lagged,
        'ratio_close_adj': x_vars_ratio_close_adj,
        'ratio_volume': x_vars_ratio_volume,
    }

    return ret
