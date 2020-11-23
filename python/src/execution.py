# from src.d.data_generation import generate_data
from src.d.data_generation import generate_data
from src.ml.machine_learning import learning, saving
from src.ml.utils import param_filename

from itertools import product as itpd
from multiprocessing import Process
import psutil

# Suppress tensorflow logging
import logging
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Execute machine learning
def ml(
    plot,
    network,
    stencil,
    layer,
    activation,
    batch_size=128,
    epochs=25,
    learning_rate=1e-3,
    neg=False,
    angle=False,
    rot=False,
    data=['circle'],
    smearing=False,
    hf='hf',
    hf_correction=False,
    dropout=0,
    plotdata=False,
    flip=False,
    cut=False,
    dshift=0,
    shift=0,
    bias=True,
    interpolate=0,
    edge=0,
    edge2=False,
    custom_loss=0,
    gauss=0,
    unsharp_mask=False,
    amount=0.1,
    load_data='',
    seed=False,
    addstring=False,
    stz_kappa=11,
    normalize=False,
    interpolate_lower=0.5,
    affinity=5,
):



    equal_kappa = True

    # Parameters
    parameters = {
        'plot': plot,
        'network': network,              # Network type
        'epochs': epochs,                # Number of epochs
        'layers': layer,                 # Autoencoder: [n*Encoder Layers, 1*Coding Layer, 1*Feedforward Layer]
        'stencil_size': stencil,         # Stencil size [x, y]
        'equal_kappa': equal_kappa,      # P(kappa) = const. or P(r) = const.
        'learning_rate': learning_rate,  # Learning Rate
        'batch_size': batch_size,        # Batch size
        'activation': activation,        # Activation function
        'negative': neg,                 # Negative values too or only positive
        'angle': angle,                  # Use the angles of the interface too
        'rotate': rot,                   # Rotate the data before learning
        'normalize': normalize,
        'data': data,                    # 'ellipse', 'circle', 'both'
        'smear': smearing,               # Use smeared data
        'stz_kappa': stz_kappa,
        'hf': hf,                        # Use height function
        'hf_correction': hf_correction,  # Use height function as input for NN
        'plotdata': plotdata,
        # 'dropout': dropout               # dropout fraction
        'flip': flip,
        'cut': cut,
        'dshift': dshift,
        'shift': shift,
        'bias': bias,
        'interpolate_lower': interpolate_lower,
        'interpolate': interpolate,
        'edge': edge,
        'edge2': edge2,
        'custom_loss': custom_loss,
        'gauss': gauss,
        'unsharp_mask': unsharp_mask,
        'amount': amount,
        'load_data': load_data,
        'seed': seed,
        'addstring': addstring,
        #'addstring': '_dshift1b_shift_kappa',
    }
    # '''
    affinity = [affinity]
    proc = psutil.Process()  # get self pid
    aff = proc.cpu_affinity()
    # print('Affinity before: {aff}'.format(aff=aff))
    proc.cpu_affinity(affinity)
    aff = proc.cpu_affinity()
    print('Affinity: {aff}'.format(aff=aff))
    # '''

    # Generate filename string
    parameters['filename'] = param_filename(parameters) + parameters['addstring']

    # Execute learning
    if parameters['network'] != 'auto':
        learning(parameters, silent=True, plot=plot)

    elif parameters['network'] == 'auto':
        # Autoencoder
        if not plot:
            parameters['network'] = 'autoencdec'
            parameters['epochs'] = int(parameters['epochs']*2)
            learning(parameters, silent=True, plot=plot)
            parameters['epochs'] = int(parameters['epochs']/2)
        parameters['network'] = 'autoenc'
        learning(parameters, silent=True, plot=plot)

    parameters = None


# Save model to text file
def save(
    plot,
    network,
    stencil,
    layer,
    activation,
    batch_size=128,
    epochs=25,
    learning_rate=1e-3,
    neg=False,
    angle=False,
    rot=False,
    data=['circle'],
    smearing=False,
    hf='hf',
    hf_correction=False,
    dropout=0,
    plotdata=False,
    flip=False,
    cut=False,
    dshift=0,
    shift=0,
    bias=True,
    interpolate=0,
    edge=0,
    edge2=False,
    custom_loss=0,
    gauss=0,
    unsharp_mask=False,
    amount=0.1,
    load_data='',
    seed=False,
    addstring=False,
    stz_kappa=11,
    normalize=False,
    interpolate_lower=0.5
):

    equal_kappa = True

    # Parameters
    parameters = {
        'plot': plot,
        'network': network,              # Network type
        'epochs': epochs,                # Number of epochs
        'layers': layer,                 # Autoencoder: [n*Encoder Layers, 1*Coding Layer, 1*Feedforward Layer]
        'stencil_size': stencil,         # Stencil size [x, y]
        'equal_kappa': equal_kappa,      # P(kappa) = const. or P(r) = const.
        'learning_rate': learning_rate,  # Learning Rate
        'batch_size': batch_size,        # Batch size
        'activation': activation,        # Activation function
        'negative': neg,                 # Negative values too or only positive
        'angle': angle,                  # Use the angles of the interface too
        'rotate': rot,                   # Rotate the data before learning
        'normalize': normalize,
        'data': data,                    # 'ellipse', 'circle', 'both'
        'smear': smearing,               # Use smeared data
        'stz_kappa': stz_kappa,
        'hf': hf,                        # Use height function
        'hf_correction': hf_correction,  # Use height function as input for NN
        # 'dropout': dropout               # dropout fraction
        'plotdata': plotdata,
        'flip': flip,
        'cut': cut,
        'dshift': dshift,
        'shift': shift,
        'bias': bias,
        'interpolate_lower': interpolate_lower,
        'interpolate': interpolate,
        'edge': edge,
        'edge2': edge2,
        'custom_loss': custom_loss,
        'gauss': gauss,
        'unsharp_mask': unsharp_mask,
        'amount': amount,
        'load_data': load_data,
        'seed': seed,
        'addstring': addstring,
    }

    # Generate filename string
    parameters['filename'] = param_filename(parameters)

    # Execute learning
    saving(parameters)


    parameters = None

# Data generation
def exe_dg(**kwargs):
    print(f'kwargs:\n{kwargs}')
    # Sort input keyword arguments
    order = ['N_values', 'stencils', 'ek', 'neg', 'silent', 'geometry', 'smearing', 'usenormal', 'dshift', 'interpolate', 'gauss', 'stz_kappa', 'interpolate_lower']
    kwargs = {k: kwargs[k] for k in order}
    # Create job list according to input arguments
    job_list = list(itpd(*kwargs.values()))
    if len(job_list) > 1:
        # Execute job list with multithreading
        jobs = []
        for job in job_list:
            # Add cpu affinity
            job = job + (int(job_list.index(job)),)
            jobs.append(Process(target=generate_data, args=job))
        [j.start() for j in jobs]
        [j.join() for j in jobs]
    else:
        # Execute job
        generate_data(**dict(zip(kwargs.keys(), job_list[0])))

# Machine Learning
def exe_ml(**kwargs):
    # Sort input keyword arguments
    order = ['plot', 'network', 'stencil', 'layer', 'activation', 'batch_size', 'epochs', 'learning_rate', 'neg', 'angle', 'rot', 'data', 'smearing', 'hf', 'hf_correction', 'dropout', 'plotdata', 'flip', 'cut', 'dshift', 'shift', 'bias', 'interpolate', 'edge', 'edge2', 'custom_loss', 'gauss', 'unsharp_mask', 'amount', 'load_data',  'seed', 'addstring', 'stz_kappa', 'normalize', 'interpolate_lower']
    kwargs = {k: kwargs[k] for k in order}
    plot = kwargs.get('plot')
    # Execute machine learning
    '''
    for job in list(itpd(*kwargs.values())):
        ml(**dict(zip(kwargs.keys(), job)))
    # '''
    # '''
    if not plot[0]: # Execute training job list with multithreading
        kwargs['plotdata'] = [False]
        jobs = []
        job_list = list(itpd(*kwargs.values()))
        for job in job_list:
            # Add cpu affinity
            job = job + (int(job_list.index(job)),)
            jobs.append(Process(target=ml, args=job))
        [j.start() for j in jobs]
        [j.join() for j in jobs]
    elif plot[0]:
        # Execute validation job list with multithreading
        for job in list(itpd(*kwargs.values())):
            ml(**dict(zip(kwargs.keys(), job)))
    # '''

# Save model
def exe_save(**kwargs):
    # Sort input keyword arguments
    order = ['plot', 'network', 'stencil', 'layer', 'activation', 'batch_size', 'epochs', 'learning_rate', 'neg', 'angle', 'rot', 'data', 'smearing', 'hf', 'hf_correction', 'dropout', 'plotdata', 'flip', 'cut', 'dshift', 'shift', 'bias', 'interpolate', 'edge', 'edge2', 'custom_loss', 'gauss', 'unsharp_mask', 'amount', 'load_data',  'seed', 'addstring', 'stz_kappa', 'normalize', 'interpolate_lower']
    kwargs = {k: kwargs[k] for k in order}
    # Execute saving job list with multithreading
    for job in list(itpd(*kwargs.values())):
        save(**dict(zip(kwargs.keys(), job)))
