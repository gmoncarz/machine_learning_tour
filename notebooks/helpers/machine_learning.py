import pandas as pd
import tensorflow as tf


def train_model(df_train, x_vars, y_var, model_class, model_params):
    model = model_class(**model_params)
    model.fit(df_train[x_vars], df_train[y_var])

    return model


def train_tensorflow_model(df_train, x_vars, y_var,
                           model_class, model_params={},
                           fit_params={}):
    model = model_class(**model_params)

    model.fit(
        df_train[x_vars].values,
        df_train[y_var].values.reshape([-1, 1]),
        **fit_params,
    )

    return model


def train_sequential_tensorflow_model(mat_x, mat_y, model_class,
        model_params={}, fit_params={}):

    model = model_class(**model_params)

    model.fit(mat_x, mat_y, **fit_params,)

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


def tf_build_dnn(neurons=[10], activations=['relu'],
                 input_shapes=None, regs_kernel_l1=None, regs_kernel_l2=None,
                 optimizer_eval='tf.keras.optimizers.Adam(lr=0.01)',
                 compile_kwargs={'loss': 'mse'},
                ):

    if input_shapes is None:
        input_shapes = [None] * len(neurons)
    elif not isinstance(input_shapes, tuple) and not isinstance(input_shapes, list):
        input_shapes = [input_shapes] * len(neurons)

    if activations is None:
        activations = [None] * len(neurons)
    elif not isinstance(activations, tuple) and not isinstance(activations, list):
        activations = [activations] * len(neurons)

    if regs_kernel_l1 is None:
        regs_kernel_l1 = [None] * len(neurons)
    elif not isinstance(regs_kernel_l1, tuple) and not isinstance(regs_kernel_l1, list):
        regs_kernel_l1 = [regs_kernel_l1] * len(neurons)

    if regs_kernel_l2 is None:
        regs_kernel_l2 = [None] * len(neurons)
    elif not isinstance(regs_kernel_l2, tuple) and not isinstance(regs_kernel_l2, list):
        regs_kernel_l2 = [regs_kernel_l2] * len(neurons)

    model = tf.keras.models.Sequential()

    for (neuron_quantity, activation, reg_l1, reg_l2, input_shape) in zip(
            neurons, activations, regs_kernel_l1, regs_kernel_l2, input_shapes):

        dense_params = {'units': neuron_quantity}

        if input_shape:
            dense_params['input_shape'] = input_shape
        if activation:
            dense_params['activation'] = activation

        if reg_l1 is not None and reg_l2 is not None:
            dense_params['kernel_regularizer'] = tf.keras.regularizers.l1_l2(l1=reg_l1, l2=reg_l2)
        elif reg_l1 is not None:
            dense_params['kernel_regularizer'] = tf.keras.regularizers.l1(l=reg_l1)
        elif reg_l2 is not None:
            dense_params['kernel_regularizer'] = tf.keras.regularizers.l2(l=reg_l2)

        model.add(tf.keras.layers.Dense(**dense_params))


    compile_kwargs = compile_kwargs.copy()

    optimizer = eval(optimizer_eval)
    compile_kwargs['optimizer'] = optimizer

    model.compile(**compile_kwargs)

    return model
