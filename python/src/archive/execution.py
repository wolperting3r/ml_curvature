from src.data_generation import generate_data
from src.machine_learning import learning

# Suppress tensorflow logging
import logging
import os
import itertools
import multiprocessing

# import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def gendat(inzip):
    print(f'Generating data: {inzip}')
    generate_data(N_values=1e6, st_sz=inzip[0], equal_kappa=inzip[1], silent=True)
    print(f'Generation finished: {inzip}')


def ml(network, stencil, layer, act, plot, epochs=5, batch_size=2048, learning_rate=1e-3, equal_kappa=True):
    # Parameters
    parameters = {'network': network,
                  'epochs': epochs,
                  'layers': layer,
                  'stencil_size': stencil,
                  'equal_kappa': equal_kappa,
                  'learning_rate': learning_rate,
                  'batch_size': batch_size,
                  'activation': act}

    # Get data filename
    print('\nParameters:')
    for key, value in parameters.items():
        print(str(key) + ': ' + str(value))
    filename = 'data_' + \
            str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
            ('eqk' if parameters['equal_kappa'] else 'eqr') + \
            '.feather'

    print('\nImporting ' + filename)
    # Execute learning
    learning(filename, parameters, silent=True, regenerate=(False if plot else True), plot=plot)
    print('Finished:')
    for key, value in parameters.items():
        print(str(key) + ': ' + str(value))

def exe_dg(stencils, ek):
    # Execute data generation
    job_list = list(itertools.product(*(stencils, ek)))
    jobs = []
    for job in job_list:
        process = multiprocessing.Process(target=gendat, args=[job])
        jobs.append(process)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

def exe_ml(network, stencils, layers, activation):
    # Execute machine learning
    # Set plot = False
    plot = [False]
    job_list = list(itertools.product(*(network, stencils, layers, activation, plot)))
    print(f'job_list:\n{job_list}')

    jobs = []
    for job in job_list:
        process = multiprocessing.Process(target=ml, args=job)
        jobs.append(process)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

    # Plot
    plot = [True]
    job_list = list(itertools.product(*(network, stencils, layers, activation, plot)))
    for job in job_list:
        ml(job[0], job[1], job[2], job[3], job[4])
