from src.execution import exe_ml, exe_save
from src.execution import exe_dg
from datetime import datetime
from itertools import product as Product
import os
import sys
if __name__ == '__main__':

    ''' Data Generation '''
    arguments_dg = {
        'stencils':             [[3, 3], [5, 5], [7, 7], [9, 9]],
        'ek':                   [True],
        'neg':                  [True],
        'N_values':             [1e6],
        'silent':               [True],
        'geometry':             ['ellipse'],
        'smearing':             [True],
        'usenormal':            [True],
        'dshift':               [True],
        'gauss':                [False],
        'stz_kappa':            [10],
        'interpolate_lower':    [0],
        'interpolate':          [0],
    }
    # exe_dg(**arguments_dg)

    ''' Machine Learning '''
    arguments={
        'seed':             [1, 2, 3, 4, 5, 6],
        'epochs':           [750],
        'plot':             [False],
        'network':          ['mlp'],
        'stencil':          [[7, 7]],
        'layer':            [[200, 150, 120]],
        'activation':       ['tanh'],
        'batch_size':       [128],
        'learning_rate':    [5e-5],
        'data':             ['ellipse'],
        'stz_kappa':        [10],
        'interpolate_lower':[0],
        'interpolate':      [0],
        'dshift':           [1],
        'shift':            [1],
        'smearing':         [True],
        'rot':              [True],
        'edge2':            [True],
        'normalize':        [True],
        'hf':               ['hf'],
        'hf_correction':    [False],
        'dropout':          [0],
        'plotdata':         ['ellipse'],
        'addstring':        ['_herkules'],
        'flip':             [False],
        'cut':              [False],
        'bias':             [True],
        'custom_loss':      [False],
        'gauss':            [0],
        'load_data':        [''],
        'unsharp_mask':     [False],
        'edge':             [0],
        'amount':           [0],
        'neg':              [True],
        'angle':            [False],
    }
    testrun = False
    # '''
    variable_parameters = {}
    variable_parameters['layer'] = [
            # [25], [50], [100], [150], [200],
            [25, 25], [100, 100], [200, 200], [200, 100], [100, 200], [50, 25], [25, 50],
            [25, 25, 25], [100, 100, 100], [200, 200, 200], [200, 100, 25], [25, 100, 200], [200, 150, 100], [100, 150, 200], [150, 100, 50], [50, 100, 150],
            # [25, 25, 25, 25, 25], [100, 100, 100, 100, 100], [200, 200, 200, 200, 200], [200, 150, 100, 50, 25],
            ]
    variable_parameters['edge2'] = [True, False]

    variable_parameters['stencil'] = [[5, 5], [3, 3]]
    variable_parameters['seed'] = [[7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18]]

    configurations = list(Product(*variable_parameters.values()))

    for i in range(0, 2):
        if i == 0:
            arguments['plot'] = [False]
            for configuration in configurations:
                with open('zwischenstand.txt', 'a') as output_file:
                    output_file.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\t' + str(configuration) + '\n')
                for key in variable_parameters.keys():
                    arguments[key] = (configuration[list(variable_parameters.keys()).index(key)] if (key == 'seed') else [configuration[list(variable_parameters.keys()).index(key)]])
                if not testrun:
                    exe_ml(**arguments)
        if i == 1:
            arguments['plot'] = [True]
            for configuration in configurations:
                with open('zwischenstand.txt', 'a') as output_file:
                    output_file.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\tplot\t' + str(configuration) + '\n')
                for key in variable_parameters.keys():
                    arguments[key] = (configuration[list(variable_parameters.keys()).index(key)] if (key == 'seed') else [configuration[list(variable_parameters.keys()).index(key)]])
                if not testrun:
                    exe_ml(**arguments)
                    exe_save(**arguments)
