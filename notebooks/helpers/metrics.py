def get_x_vars_classifier(columns):
    x_vars_all = list(filter(lambda varname:
        ('_adj' in varname or 'volume' in varname) and
            not '_shift_' in varname and
            not '_std' in varname and
            not '_norm' in varname and
            not '_ret_' in varname,
        columns,
    ))

    x_vars_all_norm = list(filter(
        lambda varname: ('_adj' in varname or 'volume' in varname) and
            not '_shift_' in varname and
            not '_std' in varname and
            '_norm' in varname and
            not '_ret_' in varname,
        columns)
    )

    x_vars_all_std = list(filter(
        lambda varname: ('_adj' in varname or 'volume' in varname) and
            not '_shift_' in varname and
            '_std' in varname and
            not '_norm' in varname and
            not '_ret_' in varname,
        columns)
    )

    x_vars_slope = list(filter(lambda var: var.startswith('slope_'), x_vars_all))
    x_vars_sma = list(filter(lambda var: var.startswith('sma_'), x_vars_all))
    x_vars_ema = list(filter(lambda var: var.startswith('ema_'), x_vars_all))
    x_vars_lagged = list(filter(lambda var: var.startswith('lag_'), x_vars_all))
    x_vars_ratio_close_adj = list(filter(lambda var: var.startswith('ratio_close_adj_'), x_vars_all))
    x_vars_ratio_volume = list(filter(lambda var: var.startswith('ratio_volume_'), x_vars_all))

    x_vars_slope_norm = list(filter(lambda var: var.startswith('slope_'), x_vars_all_norm))
    x_vars_sma_norm = list(filter(lambda var: var.startswith('sma_'), x_vars_all_norm))
    x_vars_ema_norm = list(filter(lambda var: var.startswith('ema_'), x_vars_all_norm))
    x_vars_lagged_norm = list(filter(lambda var: var.startswith('lag_'), x_vars_all_norm))
    x_vars_ratio_close_adj_norm = list(filter(lambda var: var.startswith('ratio_close_adj_'), x_vars_all_norm))
    x_vars_ratio_volume_norm = list(filter(lambda var: var.startswith('ratio_volume_'), x_vars_all_norm))

    x_vars_slope_std = list(filter(lambda var: var.startswith('slope_'), x_vars_all_std))
    x_vars_sma_std = list(filter(lambda var: var.startswith('sma_'), x_vars_all_std))
    x_vars_ema_std = list(filter(lambda var: var.startswith('ema_'), x_vars_all_std))
    x_vars_lagged_std = list(filter(lambda var: var.startswith('lag_'), x_vars_all_std))
    x_vars_ratio_close_adj_std = list(filter(lambda var: var.startswith('ratio_close_adj_'), x_vars_all_std))
    x_vars_ratio_volume_std = list(filter(lambda var: var.startswith('ratio_volume_'), x_vars_all_std))

    ret = {
        'all': x_vars_all,
        'slope': x_vars_slope,
        'sma': x_vars_sma,
        'ema': x_vars_ema,
        'lagged': x_vars_lagged,
        'ratio_close_adj': x_vars_ratio_close_adj,
        'ratio_volume': x_vars_ratio_volume,

        'all_norm': x_vars_all_norm,
        'slope_norm': x_vars_slope_norm,
        'sma_norm': x_vars_sma_norm,
        'ema_norm': x_vars_ema_norm,
        'lagged_norm': x_vars_lagged_norm,
        'ratio_close_adj_norm': x_vars_ratio_close_adj_norm,
        'ratio_volume_norm': x_vars_ratio_volume_norm,

        'all_std': x_vars_all_std,
        'slope_std': x_vars_slope_std,
        'sma_std': x_vars_sma_std,
        'ema_std': x_vars_ema_std,
        'lagged_std': x_vars_lagged_std,
        'ratio_close_adj_std': x_vars_ratio_close_adj_std,
        'ratio_volume_std': x_vars_ratio_volume_std,
    }

    return ret
