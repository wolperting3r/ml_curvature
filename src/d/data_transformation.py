import pandas as pd
import numpy as np
import os
import sys
import time

from sklearn.pipeline import Pipeline
from src.d.transformators import TransformData, FindGradient, FindAngle, Rotate, CDS, HF, TwoLayers, Shift, Edge, Edge2, UnsharpMask


# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)

def get_data(parameters):
    ''' Import data from files '''
    if ((parameters['data'] == 'all') & (len(parameters['load_data']) == 0)):
        # Data file to load
        # Vorher war hier b und unten keins, für 5x5 geändert
        filename_cir = 'data_' + \
            str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
            ('eqk' if parameters['equal_kappa'] else 'eqr') + \
            ('_neg' if parameters['negative'] else '_pos') + \
            '_cir' + \
            ('_smr' if parameters['smear'] else '_nsm') + \
            ('_shift1' if parameters['dshift'] else '') + \
            (('_int' + str(parameters['interpolate'])) if parameters['interpolate'] else '') + \
            ('_g' if parameters['gauss'] else '') + \
            '.feather'
        filename_sin = 'data_' + \
            str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
            ('eqk' if parameters['equal_kappa'] else 'eqr') + \
            ('_neg' if parameters['negative'] else '_pos') + \
            '_sin' + \
            ('_smr' if parameters['smear'] else '_nsm') + \
            ('_shift1' if parameters['dshift'] else '') + \
            (('_int' + str(parameters['interpolate'])) if parameters['interpolate'] else '') + \
            ('_g' if parameters['gauss'] else '') + \
            '.feather'
        filename_ell = 'data_' + \
            str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
            ('eqk' if parameters['equal_kappa'] else 'eqr') + \
            ('_neg' if parameters['negative'] else '_pos') + \
            '_ell' + \
            ('_smr' if parameters['smear'] else '_nsm') + \
            ('_shift1' if parameters['dshift'] else '') + \
            (('_int' + str(parameters['interpolate'])) if parameters['interpolate'] else '') + \
            ('_g' if parameters['gauss'] else '') + \
            '.feather'

        if parameters['dshift'] == '1b':
            filename_cir2 = 'data_' + \
                str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
                ('eqk' if parameters['equal_kappa'] else 'eqr') + \
                ('_neg' if parameters['negative'] else '_pos') + \
                '_cir' + \
                ('_smr' if parameters['smear'] else '_nsm') + \
                ('_int2' if parameters['plot'] else (('_int' + str(parameters['interpolate'])) if parameters['interpolate'] else '')) + \
                '_shift1b' + \
                '.feather'
            filename_sin2 = 'data_' + \
                str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
                ('eqk' if parameters['equal_kappa'] else 'eqr') + \
                ('_neg' if parameters['negative'] else '_pos') + \
                '_sin' + \
                ('_smr' if parameters['smear'] else '_nsm') + \
                '_shift1' + \
                ('_int2' if parameters['plot'] else (('_int' + str(parameters['interpolate'])) if parameters['interpolate'] else '')) + \
                '.feather'
            filename_ell2 = 'data_' + \
                str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
                ('eqk' if parameters['equal_kappa'] else 'eqr') + \
                ('_neg' if parameters['negative'] else '_pos') + \
                '_ell' + \
                ('_smr' if parameters['smear'] else '_nsm') + \
                '_shift1b' + \
                ('_int2' if parameters['plot'] else (('_int' + str(parameters['interpolate'])) if parameters['interpolate'] else '')) + \
                '.feather'

        # Get path
        parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        print(f'Dataset 1:\t{filename_cir}')
        print(f'Dataset 2:\t{filename_ell}')
        print(f'Dataset 3:\t{filename_sin}')

        # Get path for data
        path_cir = os.path.join(parent_path, 'data', 'datasets', filename_cir)
        path_sin = os.path.join(parent_path, 'data', 'datasets', filename_sin)
        path_ell = os.path.join(parent_path, 'data', 'datasets', filename_ell)

        # Load data
        data_cir = pd.read_feather(path_cir)
        data_sin = pd.read_feather(path_sin)
        data_ell = pd.read_feather(path_ell)

        # Cut sin and cir data by half
        data_sin = data_sin[:int(data_sin.shape[0]/2)]
        data_cir = data_sin[:int(data_cir.shape[0]/2)]

        if parameters['dshift'] == '1b':
            print(f'Dataset 1b:\t{filename_cir2}')
            print(f'Dataset 2b:\t{filename_ell2}')
            print(f'Dataset 3b:\t{filename_sin2}')
            path_cir2 = os.path.join(parent_path, 'data', 'datasets', filename_cir2)
            path_sin2 = os.path.join(parent_path, 'data', 'datasets', filename_sin2)
            path_ell2 = os.path.join(parent_path, 'data', 'datasets', filename_ell2)
            data_cir2 = pd.read_feather(path_cir2)
            data_sin2 = pd.read_feather(path_sin2)
            data_ell2 = pd.read_feather(path_ell2)
            data_cir2 = data_sin2[:int(data_cir2.shape[0]/4)]
            data_sin2 = data_sin2[:int(data_sin2.shape[0]/4)]
            data_ell2 = data_ell2[:int(data_ell2.shape[0]/2)]
            data = pd.concat([data_cir, data_sin, data_ell, data_cir2, data_sin2, data_ell2], ignore_index=True)

        # Glue everything together
        data = pd.concat([data_sin, data_ell, data_cir], ignore_index=True)
    else:
        if len(parameters['load_data']) > 0:
            # Load data specified in load_data
            parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
            filename = parameters['load_data'] + '.feather'
            path = os.path.join(parent_path, 'data', 'datasets', filename)
            if os.path.isfile(path):
                print(f'Dataset:\t{filename}')
                data = pd.read_feather(path)
            else:
                raise ValueError(f'Dataset {filename} not found!')
        else:
            if parameters['data'] == 'ellipse':
                geom_str = '_ell'
            elif parameters['data'] == 'sinus':
                geom_str = '_sin'
            elif parameters['data'] == 'circle':
                geom_str = '_cir'
            # Data file to load
            filename = 'data_' + \
                str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
                ('eqk' if parameters['equal_kappa'] else 'eqr') + \
                ('_neg' if parameters['negative'] else '_pos') + \
                geom_str + \
                ('_smr' if parameters['smear'] else '_nsm') + \
                ('_shift1' if parameters['dshift'] else '') + \
                (('_ab' + str(parameters['interpolate_lower'])) if parameters['interpolate'] else  '') + \
                (('_int' + str(parameters['interpolate'])) if parameters['interpolate'] else '') + \
                ('_g' if parameters['gauss'] else '') + \
                ('_' + str(parameters['stz_kappa']) if (parameters['stz_kappa'] != 0) else '')
            parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
            # if os.path.isfile(os.path.join(parent_path, 'data', 'datasets', filename+'_25.feather')):
                # filename = filename+'_25.feather'
            if os.path.isfile(os.path.join(parent_path, 'data', 'datasets', filename+'_eqkmax.feather')):
                filename = filename+'_eqkmax.feather'
            elif not os.path.isfile(os.path.join(parent_path, 'data', 'datasets', filename+'.feather')):
                if os.path.isfile(os.path.join(parent_path, 'data', 'datasets', filename+'_intmin05.feather')):
                    filename = filename+'_intmin05.feather'
            else:
                filename = filename+'.feather'

            # Load data
            print(f'Dataset:\t{filename}')
            path = os.path.join(parent_path, 'data', 'datasets', filename)
            data = pd.read_feather(path)

            if parameters['dshift'] == '1b':
                filename2 = 'data_' + \
                    str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
                    ('eqk' if parameters['equal_kappa'] else 'eqr') + \
                    ('_neg' if parameters['negative'] else '_pos') + \
                    geom_str + \
                    ('_smr' if parameters['smear'] else '_nsm') + \
                    '_shift1b' + \
                    (('_ab' + str(parameters['interpolate_lower'])) if parameters['interpolate'] else  '') + \
                    (('_int' + str(parameters['interpolate'])) if parameters['interpolate'] else '') + \
                    ('_g' if parameters['gauss'] else '') + \
                    '.feather'
                print(f'Dataset2:\t{filename2}')
                path2 = os.path.join(parent_path, 'data', 'datasets', filename2)
                data2 = pd.read_feather(path2)
                data2= data2[:int(data2.shape[0]/2)]
                data = pd.concat([data, data2], ignore_index=True)
    # Only return data with the curvature being below a certain threshold
    # data = data[np.abs(data.iloc[:, 0]) < 0.15]

    # Flip labels
    if parameters['flip']:
        if len(parameters['load_data']) > 0:
            data.iloc[:, 0] = data.iloc[:, 0] # CVOFLS Data
        else:
            data.iloc[:, 0] = -data.iloc[:, 0] # Eigene Netze

    return data.copy()


def split_data(data, ratio, parameters):
    '''
    Create randomized train set and test set based on the ratio
    ---
    Input:
        data[pd df]     data
        ratio[int]      test:train ratio
    Output:
        testdata[pd df]
        traindata[pd df]
    '''
    # Set seed
    # np.random.seed(42)
    np.random.seed(parameters['seed']+1)
    # Generate random indices
    indices = np.random.permutation(len(data))

    if isinstance(ratio, list):
        # [test_ratio, val_ratio, train_ratio] as list
        # Calculate how many entries the test data will have
        test_size = int(len(data)*ratio[0])
        val_size = int(len(data)*ratio[1])
    
        # Get the test indices from the randomly generated indices
        test_indices = indices[:test_size]
        # Get the validation indices from the randomly generated indices
        val_indices = indices[test_size:val_size+test_size]
        # Get the train indices from the randomly generated indices
        train_indices = indices[val_size+test_size:]
        # Return the data corresponding to the indices
        return data.iloc[test_indices], data.iloc[val_indices], data.iloc[train_indices]
    else:
        # test_ratio as float
        # Calculate how many entries the test data will have
        test_size = int(len(data)*ratio)
        # Get the test indices from the randomly generated indices
        test_indices = indices[:test_size]
        # Get the train indices from the randomly generated indices
        train_indices = indices[test_size:]
        # Return the data corresponding to the indices
        return data.iloc[test_indices], data.iloc[train_indices]
    
    
def process_kappa(dataset, parameters, reshape):
    '''
    # Calculate kappa with CV or HF
    # '''

    if parameters['hf'] == 'hf':
        # Calculate kappa with HF
        # Create pipeline
        # '''
        data_pipeline = Pipeline([
            ('transform', TransformData(parameters=parameters, reshape=reshape)),
            ('findgradient', FindGradient(parameters=parameters)),
            ('findkappahf', HF(parameters=parameters)),
        ])
        # Execute pipeline
        [labels, features, kappa] = data_pipeline.fit_transform(dataset)
    elif parameters['hf'] == 'cd':
        # Calculate kappa with CV
        data_pipeline = Pipeline([
            ('transform', TransformData(parameters=parameters, reshape=reshape)),
            ('findkappacds', CDS(parameters=parameters)),
        ])
        # Execute pipeline
        [labels, features, kappa] = data_pipeline.fit_transform(dataset)
    else:
        kappa = 0

    return [labels, kappa]


def process_data(dataset, parameters, reshape):

    # Pre-Processing

    # '''
    # Initialize pipeline list
    pipeline_list = []

    # Add steps to pipeline
    # Transform data to required format
    pipeline_list.append(('transform', TransformData(parameters=parameters, reshape=reshape)))
    # Find gradient in middle cell
    pipeline_list.append(('findgradient', FindGradient(parameters=parameters)))
    # if parameters['hf']:
        # Calculate kappa with HF (angle in output = kappa) ! Needs adjustments !
        # pipeline_list.append(('findkappahf', HF(parameters=parameters)))
    # Calculate normal vector angle ! Needs adjustments !
    pipeline_list.append(('findangle', FindAngle(parameters=parameters)))
    if parameters['shift']:
        # Shift stencils
        pipeline_list.append(('shift', Shift(parameters=parameters, shift=parameters['shift'])))
    if parameters['rotate']:
        # Rotate stencils
        pipeline_list.append(('rotate', Rotate(parameters=parameters)))
    if parameters['edge']:
        # Interface reconstruction with interpolation
        pipeline_list.append(('edge', Edge(parameters=parameters)))
    if parameters['edge2']:
        # Interface reconstruction with interpolation
        pipeline_list.append(('edge2', Edge2(parameters=parameters)))
    if parameters['unsharp_mask']:
        # Interface reconstruction with unsharp mask
        pipeline_list.append(('unsharp', UnsharpMask(parameters=parameters, amount=parameters['amount'])))
    # Build pipeline
    data_pipeline = Pipeline(pipeline_list)

    # Execute pipeline
    [labels, features, angle] = data_pipeline.fit_transform(dataset)

    if parameters['angle']:
        # Stack features and angles
        features = np.hstack((features, angle))

    if parameters['hf_correction']:
        # Kappa is given in angle output.
        kappa = angle 
        if parameters['network'] == 'cvn':
            # Write kappa into second layer on 3rd axis
            data_pipeline = Pipeline([
                ('twolayers', TwoLayers(parameters=parameters)),
            ])
            [labels, features] = data_pipeline.fit_transform([labels, features, kappa])
        else:
            # Stack features and height function
            features = np.concatenate((features, kappa), axis=1)
    # '''
    # Cut concentration above and below threshold
    if parameters['cut']:
        features[np.nonzero(features < 0.005)] = 0
        features[np.nonzero(features > 0.995)] = 1
    if parameters['normalize']:
        kappa_max = 4/parameters['stz_kappa']
        kappa_min = -4/parameters['stz_kappa']
        labels = (2*labels - kappa_max - kappa_min)/(kappa_max - kappa_min)
        features = (2*features - features.max() - features.min())/(features.max() - features.min())

    return [labels, features, angle]


def transform_kappa(parameters, reshape=False, plot=False, **kwargs):
    time0 = time.time()

    parameters_loc = parameters.copy()
    # Load 15x15 data for cds 
    if parameters['hf'] == 'cd':
        parameters_loc['stencil_size'] = [15, 15]
        parameters_loc['smearing'] = False
    # Read data
    if 'data' in kwargs:
        data = kwargs.get('data')
    else:
        data = get_data(parameters_loc)

    if 'data' in kwargs:  # And use all of that data in test set
        # Split data
        test_set = data
        val_set = data
        train_set = data
    else:
        # Split data
        test_set, val_set, train_set = split_data(data, [1.0, 0.0, 0.0], parameters)

    print(f'Dataset Shape:\t{test_set.shape}')

    # Pre-Processing
    [test_labels, test_kappa] = process_kappa(test_set, parameters_loc, reshape)
    # '''
    if not plot:
        [train_labels, train_kappa] = process_kappa(train_set, parameters_loc, reshape)
        [val_labels, val_kappa] = process_kappa(val_set, parameters_loc, reshape)
    else:
        [train_labels, train_kappa] = 0, 0
        [val_labels, val_kappa] = 0, 0
    # '''

    filestring = parameters['filename']
    print(f'Time needed for finding kappa of {filestring}:\t{np.round(time.time()-time0,3)}s')
    print(f'Kappa Shape:\t{test_kappa.shape}')

    return [[train_labels, train_kappa],
            [test_labels, test_kappa],
            [val_labels, val_kappa]]


def transform_data(parameters, reshape=False, plot=False, **kwargs):
    time0 = time.time()
    # Read data
    if 'data' in kwargs:  # If data is passed explicitly, use that (usually for plotting purposes)
        data = kwargs.get('data')
    else:
        data = get_data(parameters)

    # Condition for data
    # data = data[(np.abs(data.iloc[:, 0]) < 0.05)]
    # data = data[:1000]

    if 'data' in kwargs:  # And use all of that data in test set
        # Split data
        test_set = data
        val_set = data
        train_set = data
    else:
        # Split data
        test_set, val_set, train_set = split_data(data, [0.15, 0.15, 0.7], parameters)

    # Pre-Processing
    test_labels, test_data, test_angle = process_data(test_set, parameters, reshape)
    # '''
    if not plot:
        train_labels, train_data, train_angle = process_data(train_set, parameters, reshape)
        val_labels, val_data, val_angle = process_data(val_set, parameters, reshape)
    else:
        train_labels, train_data, train_angle = 0, 0, 0
        val_labels, val_data, val_angle = 0, 0, 0
    # '''


    filestring = parameters['filename']
    print(f'Time needed for pre-processing of {filestring}:\t{np.round(time.time()-time0,3)}s')

    return [
        [train_labels, train_data, train_angle],
        [test_labels, test_data, test_angle],
        [val_labels, val_data, val_angle],
    ]

    '''
    # Test
    ind = 6
    # print_data_orig = np.reshape(test_data, (test_data.shape[0], 5, 5, 1)).transpose((0, 1, 3, 2))[ind]
    print_data_grad = test_data.transpose((0, 1, 3, 2))[ind]
    # print(f'\nOriginal:\n{print_data_orig}')
    print(f'\nGedreht:\n{print_data_grad}')
    # '''

