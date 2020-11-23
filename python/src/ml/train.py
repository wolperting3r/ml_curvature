import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback
from .utils import param_filename
from .building import custom_loss
from keras import backend as K

import os
import sys


def train_model(model, train_data, train_labels, val_data, val_labels, parameters, silent):
    # Build tensorflow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))\
                    .batch(parameters['batch_size'])\
                    .prefetch(parameters['batch_size']*4)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))\
                    .batch(parameters['batch_size'])\
                    .prefetch(parameters['batch_size']*4)
    # Train Model
    # Early stopping callback
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            min_delta=10e-15, # war 10e-8
                                                            patience=100, # war 5, dann 15
                                                            restore_best_weights=True,
                                                            verbose=0,
                                                            mode='auto',
                                                            baseline=None)
    # History to csv callback
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    param_str = parameters['filename']
    file_name = os.path.join(path, 'models', 'history', 'history' + param_str + '.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(file_name, separator=',', append=False)

    # Disable Multithreading for reproducibility (https://stackoverflow.com/questions/46421258/limit-number-of-cores-used-in-keras, https://determined.ai/blog/reproducibility-in-ml/)
    # K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))
    # Train model
    model.fit(train_dataset,
              validation_data=val_dataset,
              shuffle=False,  # war true
              epochs=parameters['epochs'],  # war 10000
              verbose=0,
              callbacks=[#TqdmCallback(verbose=(0 if silent else 1)),
                         early_stopping_callback,
                         csv_logger])
    # validation_steps=2,

    # Save model
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    param_str = parameters['filename']
    file_name = os.path.join(path, 'models', 'models', 'model' + param_str + '.h5')
    model.save(file_name)
    return model

def load_model(parameters, **kwargs):
    if 'path' in kwargs:
        path = kwargs.get('path')
    else:
        path = os.path.dirname(os.path.abspath(sys.argv[0]))

    # '''
    # param_str = parameters['filename']
    param_str = parameters['filename'] # + '_shift_kappa'
    # '''
    '''
    param_tmp = parameters.copy()
    # param_tmp['data'] = 'all'
    # param_tmp.pop('filename')
    param_str = param_filename(param_tmp)
    # '''
    print(f'param_str:\n{param_str}')

    file_name = os.path.join(path, 'models', 'models', 'model' + param_str + '.h5')
    if parameters['custom_loss']:
        model = tf.keras.models.load_model(file_name, custom_objects={'custom_loss': custom_loss})
    else:
        model = tf.keras.models.load_model(file_name)
    return model
