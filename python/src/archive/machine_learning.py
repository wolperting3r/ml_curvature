# MACHINE LEARNING
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback

import io
from contextlib import redirect_stdout

from .data_transformation import transform_data

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)


def relative_mse(y_true, y_pred):
    # https://stackoverflow.com/questions/51700351/valueerror-unknown-metric-function-when-using-custom-metric-in-keras
    return tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))/y_true


def param_filename(parameters):
    # Generate filename string
    filename_string = ''
    for key, value in parameters.items():
        if key == 'layers':
            filename_string = filename_string + '_' + '-'.join(str(e) for e in value)
        elif key == 'stencil_size':
            filename_string = filename_string + '_' + str(value[0]) + 'x' + str(value[1])
        elif key == 'equal_kappa':
            filename_string = filename_string + '_' + ('eqk' if value else 'eqr')
        else:
            filename_string = filename_string + '_' + str(value)
    return filename_string


def build_model_mlp(parameters, shape):
    # Build keras model
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(shape,)))
    for l in parameters['layers']:
        model.add(layers.Dense(l, activation=parameters['activation']))
    model.add(layers.Dense(1, activation='linear'))

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(parameters['learning_rate']),
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


def build_model_cvn(parameters, shape):
    # Build keras model
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(shape[0], shape[1], 1)))
    for l in parameters['layers']:
        model.add(layers.Conv2D(l, (2, 2), activation=parameters['activation']))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation=parameters['activation']))
    model.add(layers.Dense(1, activation='linear'))

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(parameters['learning_rate']),
                  loss='mse',
                  metrics=['mae', 'mse'])

    return model


def train_model(model, train_data, train_labels, parameters, silent, regenerate=True):
    # Build tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(parameters['batch_size'])
    if regenerate:
        # Train Model
        # Early stopping callback
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                                min_delta=10e-8,
                                                                patience=50,
                                                                verbose=0,
                                                                mode='auto',
                                                                baseline=None)
        # History to csv callback
        path = os.path.dirname(os.path.abspath(__file__))
        param_str = parameters['filename']
        file_name = os.path.join(path, '..', 'models', 'history', 'history' + param_str + '.csv')
        csv_logger = tf.keras.callbacks.CSVLogger(file_name, separator=',', append=False)

        # Train model
        model.fit(dataset,
                  shuffle=True,
                  epochs=parameters['epochs'],  # war 10000
                  verbose=0,
                  callbacks=[TqdmCallback(verbose=(0 if silent else 1)),
                             early_stopping_callback,
                             csv_logger])

        # Save model
        path = os.path.dirname(os.path.abspath(__file__))
        param_str = parameters['filename']
        file_name = os.path.join(path, '..', 'models', 'models', 'model' + param_str + '.h5')
        model.save(file_name)
    else:
        path = os.path.dirname(os.path.abspath(__file__))
        param_str = parameters['filename']
        file_name = os.path.join(path, '..', 'models', 'models', 'model' + param_str + '.h5')
        model = tf.keras.models.load_model(file_name)
        print(model.summary())

    return model


def validate_model_loss(model, train_data, train_labels, test_data, test_labels, parameters):
    # Print MSE and MAE
    path = os.path.dirname(os.path.abspath(__file__))
    param_str = parameters['filename']
    file_name = os.path.join(path, '..', 'models', 'logs', 'log' + param_str + '.txt')
    # Catch print output of tensorflow functions
    f = io.StringIO()
    with redirect_stdout(f):
        print(str(model.evaluate(train_data, train_labels, verbose=2)))
        print('\n')
        print(str(model.evaluate(test_data, test_labels, verbose=2)))
        print('\n')
        print(str(model.summary()))
    out = f.getvalue()
    # Write output into logfile
    with open(file_name, 'w') as logfile:
        logfile.write(out)
        # logfile.write('\n')


def validate_model_plot(model, test_data, test_labels, parameters):
    # Validate model
    test_predictions = model.predict(test_data, batch_size=parameters['batch_size']).flatten()
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(test_labels, test_predictions, alpha=0.05)
    ax.set_xlabel('True Values [MPG]')
    ax.set_ylabel('Predictions [MPG]')
    # lims = [min(test_labels), max(test_labels)]
    lims = [-0.2, 4/3+0.2]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims)

    path = os.path.dirname(os.path.abspath(__file__))
    param_str = parameters['filename']
    file_name = os.path.join(path, '..', 'models', 'figures', 'fig' + param_str + '.png')
    fig.tight_layout()
    fig.savefig(file_name, dpi=150)
    # plt.show()
    plt.close()


def learning(filename, parameters, silent=False, regenerate=True, plot=True):
    # Read data
    [[train_labels, train_data], [test_labels, test_data]] =\
            transform_data(filename, reshape=(True if parameters['network'] == 'cvn' else False))
    # Get filename
    parameters['filename'] = param_filename(parameters)
    # Build model
    if parameters['network'] == 'mlp':
        model = build_model_mlp(parameters, shape=test_data.shape[1])
    elif parameters['network'] == 'cvn':
        model = build_model_cvn(parameters, shape=[test_data.shape[1], test_data.shape[2]])
    # Train model
    model = train_model(model, train_data, train_labels, parameters, silent, regenerate=(False if plot else True))
    # Validate model
    if plot:
        validate_model_plot(model, test_data, test_labels, parameters)
    else:
        validate_model_loss(model, train_data, train_labels, test_data, test_labels, parameters)
