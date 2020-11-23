import os
import sys
import re
import pandas as pd
import numpy as np
from progressbar import *
from scipy import ndimage

import matplotlib.pyplot as plt

import itertools
from multiprocessing import Process

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=np.inf, threshold=np.inf)


def tecplot2data(f, oszb, st_sz, filtering, filter):
    print(f'Generating data from {f} with stencil {st_sz[0]}x{st_sz[1]}')
    with open(os.path.join(f, 'res', 'staticbubble.dat'), 'r') as myfile:
        data = myfile.read()
        # Append 'zone t' to file for capturing blocks later
        data = data + '\nZONE T'
        # Get variables
        variables = re.split(r'"\n"', re.search(r'(?<=VARIABLES = ")\w+("\n"\w+)+(?=")', data)[0])
        # Bei StaticBubble müssen die nächsten beiden Zeilen auskommentiert werden
        # variables.remove('X')
        # variables.remove('Y')
        n_vars = len(variables)
        # Get gs
        gs = [int(i) for i in re.findall(r'\d+', re.search(r'I=\d+, J=\d+, K=\d+', data)[0])]
        length = gs[0]**2
        # Get all timesteps (blocks)
        blocks = re.findall(r'ZONE\sT[\d\D]+?(?=ZONE\sT)', data)
        print(f'len(blocks):\t{len(blocks)}')
        # Remove first block (no information)
        # blocks = blocks[1:]

        # Get x/y from first block 
        # coordinates = {}
        block = blocks[0]
        numbers = np.array(re.findall(r'(\d\.\d+E[\+\-]\d{2})', block))
        # print(f'len(numbers):\t{len(numbers)}')
        coordinates = np.empty((2, gs[0], gs[1], gs[2]))
        # Get x coordinates
        coordinates[0, :, :, :] = np.reshape(numbers[:np.prod(gs)], (gs[0], gs[1], gs[2]))
        # Get y coordinates
        coordinates[1, :, :, :] = np.reshape(numbers[np.prod(gs):2*np.prod(gs)], (gs[0], gs[1], gs[2]))
        max_x = np.max(coordinates[0, :, :, :])
        max_y = np.max(coordinates[1, :, :, :])
        delta = max_x/(gs[0]-1)
        coordinates = np.reshape(coordinates[:, :, :, 0], (2, gs[0], gs[1]))

        # Coordinates are messed up, make own coordinate matrix
        coordinates = np.empty((2, gs[0]-1, gs[1]-1))
        cord_vec = np.reshape(np.array(range(0, gs[0]-1)*delta), (gs[0]-1, 1))
        coordinates[0, :, :] = cord_vec.T
        coordinates[1, :, :] = np.flip(cord_vec)

        # Remove timestep 0
        blocks.pop(0)

        # print('Blocks abgeschnitten!')
        # blocks = blocks[:1]

        # Progressbar
        widgets = ['Getting values : ', Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=len(blocks))
        pbar.start()

        bl = {}
        # Get variables of every timestep
        for block in blocks:
            pbar.update(blocks.index(block))
            # Numpy array with values for all variables
            if oszb:
                values = np.empty((n_vars, gs[0]-1, gs[1]-1))
            else:
                values = np.empty((n_vars-2, gs[0]-1, gs[1]-1))
            # values = np.empty((n_vars, gs[0]-1, gs[1]-1, gs[2]-1))
            # Get all values of block
            numbers = re.findall(r'(\-?\d\.\d+E[\+\-]\d{2})', block)
            if not oszb:
                # Get rid of x and y coordinates
                numbers = list(np.array(numbers)[gs[0]*gs[1]*2*2:]) # From 33282=129^2*2(z=2)*2(x, y) on
            for v in variables:
                # j = 0: concentration, j = 1: curvature
                j = variables.index(v)
                # Assign next 128*128 values to variable v
                if oszb: # Oscillating Bubble
                    values[j, :, :] = np.reshape(numbers[j*(gs[0]-1)*(gs[1]-1):j*(gs[0]-1)*(gs[1]-1)+(gs[0]-1)*(gs[1]-1)], (gs[0]-1, gs[1]-1))
                else: # Static Bubble
                    if j >= 2:
                        values[j-2, :, :] = np.reshape(numbers[
                            (j-2)*(gs[0]-1)*(gs[1]-1) : (j-2)*(gs[0]-1)*(gs[1]-1)+(gs[0]-1)*(gs[1]-1)
                        ], (gs[0]-1, gs[1]-1))
            bl[blocks.index(block)] = values

            '''
            fig, ax = plt.subplots()
            # ax.imshow(values[1, :, :], cmap='Greys_r')
            ax.imshow(coordinates[0, :, :], cmap='Greys_r')
            plt.show()
            # '''
        pbar.finish()

    if filtering > 0:
        # Apply filter to curvature
        if filter == 'f':
            kernel = np.array(
                [[0, 1, 0],
                 [1, 4, 1],
                 [0, 1, 0]])
        elif filter == 'g':
            kernel = np.array(
                [[1, 2, 1],
                 [2, 4, 2],
                 [1, 2, 1]])

        # Progressbar
        widgets = ['Convolving : ', Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=len(bl.keys())*filtering)
        pbar.start()

        for i in range(filtering):
            for b in bl.keys():
                pbar.update(list(bl.keys()).index(b))
                # Get curvature field
                values = bl[b]
                # Get mask where curvature != 0
                mask = np.where(values[1, :, :] != 0, 1, 0)
                # Convolute kernel over curvature field
                values[1, :, :] = ndimage.convolve(values[1, :, :], kernel, mode='constant', cval=0.0)
                # Convolute mask over curvature field to get divident
                conv_mask = ndimage.convolve(mask, kernel, mode='constant', cval=0.0)*mask
                # Devide convoluted values by convoluted mask (gives error because conv_mask = 0 in some points, but that does not matter)
                bl[b][1, :, :] = np.where(conv_mask!=0, values[1, :, :]/conv_mask, 0)
        pbar.finish()

    # Initialize output array 
    output = np.empty((0, np.prod(st_sz)+1+2))

    # Progressbar
    widgets = ['Generating stencils: ', Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=len(bl.keys()))
    pbar.start()

    # Get stencils
    for b in bl.keys():
        pbar.update(list(bl.keys()).index(b))
        values = bl[b]
        # Set kappa = 0 near the edges so stencils can be obtained
        values[1, 0:int((st_sz[0]-1)/2), :] = 0
        values[1, -int((st_sz[0]-1)/2):, :] = 0
        values[1, :, 0:int((st_sz[0]-1)/2)] = 0
        values[1, :, -int((st_sz[0]-1)/2):] = 0

        # Get indices where curvature is not 0
        indices = np.nonzero(values[1, :, :] != 0)
        # Get curvature
        if oszb:
            kappa = np.reshape(values[1, indices[0], indices[1]], (indices[0].shape[0], 1))*max_x/(gs[1]-1)
        else:
            kappa = np.reshape(values[1, indices[0], indices[1]], (indices[0].shape[0], 1))

        x = np.reshape(coordinates[0, indices[0], indices[1]], (indices[0].shape[0], 1))
        y = np.reshape(coordinates[1, indices[0], indices[1]], (indices[0].shape[0], 1))

        # Get corresponding stencil
        stencils = np.empty((indices[0].shape[0], st_sz[0], st_sz[1]))
        for i in range(st_sz[1]):
            for j in range(st_sz[0]):
                stencils[:, j, i] = values[0, indices[0]+(j-int((st_sz[0]-1)/2)), indices[1]+(i-int((st_sz[1]-1)/2))]
        print(f'kappa.shape:\t{kappa.shape}')
        print(f'stencils.shape:\t{stencils.shape}')
        print(f'x.shape:\t{x.shape}')
        print(f'y.shape:\t{y.shape}')
        # Flatten stencil and write curvature in front
        new_data = np.concatenate((kappa, np.reshape(stencils, (stencils.shape[0], stencils.shape[1]*stencils.shape[2])), x, y), axis = 1)
        # Append to data array
        output = np.concatenate((output, new_data), axis=0)
    pbar.finish()
    output_df = pd.DataFrame(output)
    output_df.columns = output_df.columns.astype(str)
    output_df = output_df.rename(columns={'0':'Curvature'})
    # Write output dataframe to feather file
    parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    # Create file name
    # file_name = os.path.join(parent_path, 'data_CVOFLS_'+str(st_sz[0])+'x'+str(st_sz[1])+'.feather')
    file_name = os.path.join(f, 'res', 'staticbubble.feather')
    print(f'File:\n{file_name}')
    # print('NO OUTPUT WRITTEN')
    output_df.reset_index(drop=True).to_feather(file_name)
    print(f'Generated {output_df.shape[0]} tuples with:\nStencil size:\t{st_sz}')

if __name__ == '__main__':
    gridsize = 128
    files = [
        # f'2007181652 staticBubble {gridsize} ml mit w+g',
        # f'2007181652 staticBubble {gridsize} ml ohne w+g',
        # f'2007181652 staticBubble {gridsize} cds mit w+g', 
        # f'2007181652 staticBubble {gridsize} cds ohne w+g', 
        # f'2007181652 staticBubble {gridsize} cvofls',
        # f'2008130957 staticBubble 128 ml sharp 9x9',
        '2008151036 staticBubble 64 HF',
        '2008151036 staticBubble 128 HF',
        '2008151036 staticBubble 256 HF',
        '2008161150 staticBubble 64 ml sharp 9x9',
        '2008161150 staticBubble 256 ml sharp 9x9',
    ]

    # st_sz = [[5, 5], [7, 7], [9, 9]]
    st_sz = [[7, 7]]
    # st_sz = [[5, 5]]

    filtering = [0]
    filter = ['g']

    oszb = [False]

    kwargs = {'f': files, 'oszb': oszb, 'st_sz': st_sz, 'filtering': filtering, 'filter': filter}

    job_list = list(itertools.product(*kwargs.values()))
    '''
    # Single Process (for plotting)
    for job in job_list:
        tecplot2data(**dict(zip(kwargs.keys(), job)))
    # '''
    # '''
    # Multiprocessing
    jobs = []
    [jobs.append(Process(target=tecplot2data, args=job)) for job in job_list]
    [j.start() for j in jobs]
    [j.join() for j in jobs]
    # '''
