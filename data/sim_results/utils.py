from src.ml.training import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from src.ml.building import custom_loss


def get_predictions(data, basepath, **kwargs):
    #parameters = {'filename': '_mlp_1000_100-80_7x7_eqk_0.0001_128_relu_neg_nag_nrt_all_smr_nhc'}
    #parameters = {'filename': '_mlp_1000_100-80_7x7_eqk_0.0001_128_relu_neg_nag_rot_all_smr_nhc_old'}
    #parameters = {'filename': '_mlp_1000_100-80_7x7_eqk_0.0001_128_relu_neg_nag_nrt_circle_smr_nhc'}
    if 'filename' in kwargs:
        parameters = {'filename': kwargs.get('filename')}    
    else:
        parameters = {'filename': '_mlp_1000_100-80_7x7_eqk_0.0001_128_relu_neg_nag_rot_all_smr_nhc_old'}

    if 'custom_objects' in kwargs:
        custom_objects = kwargs.get('custom_objects')
        # model = load_model(parameters, path=basepath, custom_objects=custom_objects)

        param_str = parameters['filename']# + '_shift_kappa'
        file_name = os.path.join(basepath, 'models', 'models', 'model' + param_str + '.h5')
        model = tf.keras.models.load_model(file_name, custom_objects=custom_objects)
    else:
        # model = load_model(parameters, path=basepath)

        param_str = parameters['filename']# + '_shift_kappa'
        file_name = os.path.join(basepath, 'models', 'models', 'model' + param_str + '.h5')
        model = tf.keras.models.load_model(file_name)
    test_predictions = model.predict(data, batch_size=128).flatten()
    return test_predictions


def plot_vofs(features, number_of_plots, **kwargs):
    if 'per_row' in kwargs:
        per_row = kwargs.get('per_row')
    else:
        per_row = 5
    rows = int(number_of_plots/per_row)
    print()
    if 'size' in kwargs:
        size = 150*kwargs.get('size')
    else:
        size = 150
    fig, ax = plt.subplots(rows, per_row, figsize=(3*per_row, 3*number_of_plots/per_row), dpi=size)
    if rows > 1:
        for idx_st, a_stack in enumerate(ax):
            for idx, a in enumerate(a_stack):
                a.imshow(features[idx+int(idx_st*per_row)], cmap=('Greys_r' if not 'cmap' in kwargs else kwargs.get('cmap')))
                a.get_xaxis().set_ticks([])
                a.get_yaxis().set_ticks([])
                if 'labels' in kwargs:
                    labels=kwargs.get('labels')
                    kappa = labels[idx + int(idx_st*per_row)]

                    if 'hf' in kwargs:
                        hf = kwargs.get('hf')
                        hf = hf[idx + int(idx_st*per_row)]
                        a.set_title(f'hf: {np.round(hf, 3)}, k: {np.round(kappa, 3)}', fontsize=14)
                    else:
                        # a.set_title(f'kappa = {np.round(kappa, 3)}', fontsize=22)
                        a.set_title("kappa = %.3f" % kappa, fontsize=22)
    else:
        idx_st = 0
        if isinstance(ax, np.ndarray):
            for idx, a in enumerate(ax):
                    a.imshow(features[idx+int(idx_st*per_row)], cmap=('Greys_r' if not 'cmap' in kwargs else kwargs.get('cmap')))
                    a.get_xaxis().set_ticks([])
                    a.get_yaxis().set_ticks([])
                    if 'labels' in kwargs:
                        labels=kwargs.get('labels')
                        kappa = labels[idx + int(idx_st*per_row)]

                        if 'hf' in kwargs:
                            hf = kwargs.get('hf')
                            hf = hf[idx + int(idx_st*per_row)]
                            a.set_title(f'hf: {np.round(hf, 3)}, k: {np.round(kappa, 3)}', fontsize=12)
                        else:
                            #a.set_title(f'kappa = {np.round(kappa, 3)}', fontsize=22)
                            a.set_title("kappa = %.3f" % kappa, fontsize=22)
        else:
            a = ax
            if 'cmap' in kwargs:
                vmax = max(np.amax(features), -np.amin(features))
                vmin = -vmax
            else:
                vmax = np.amax(features)
                vmin = np.amin(features)

            axis = a.imshow(features, cmap=('Greys_r' if not 'cmap' in kwargs else kwargs.get('cmap')), vmin=vmin, vmax=vmax)
            a.get_xaxis().set_ticks([])
            a.get_yaxis().set_ticks([])
            if 'cmap' in kwargs:
                cbar = plt.colorbar(axis)
            if 'labels' in kwargs:
                labels=kwargs.get('labels')
                kappa = labels

                if 'hf' in kwargs:
                    hf = kwargs.get('hf')
                    a.set_title(f'hf: {np.round(hf, 3)}, k: {np.round(kappa, 3)}', fontsize=12)
                else:
                    #a.set_title(f'kappa = {np.round(kappa, 3)}', fontsize=22)
                    a.set_title("kappa = %.3f" % kappa, fontsize=22)

    fig.tight_layout()
    


def create_plot(labels, predictions, color, parameters, **kwargs):
    # Create plot
    fgsz = (10*kwargs.get('scale') if 'scale' in kwargs else 10)
    fig, ax = plt.subplots(1, 1, figsize=(fgsz, fgsz))

    # Create scatterplot test_predictions vs test_labels
    alpha = 0.1
    marker = kwargs.get('marker') if 'marker' in kwargs else ','
    size = (kwargs.get('size') if 'size' in kwargs else 1)
    size = (size*kwargs.get('scale') if 'scale' in kwargs else size)
    plt.scatter(labels, predictions, alpha=alpha, color=color, edgecolors='none', marker=marker, s=size)  # darkseagreen
    
    if 'axis' in kwargs:
        axis = kwargs.get('axis')
        if axis == 'equal':
            lims = [[min(predictions), max(predictions)], [min(predictions), max(predictions)]]
            #lims = [[min(labels), max(labels)], [min(labels), max(labels)]]
            # Plot the 45 degree line
            ax.plot(lims[0], lims[0], color='gray')
        else:
            lims = [[min(labels), max(labels)], [min(predictions), max(predictions)]]
        
    else:
        lims = [[min(predictions), max(predictions)], [min(predictions), max(predictions)]]
        # Plot the 45 degree line
        ax.plot(lims[0], lims[0], color='gray')
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
 
    fig.tight_layout()

    return fig, ax
