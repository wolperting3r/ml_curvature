import matplotlib.pyplot as plt
import os
import io
import sys
import numpy as np
from contextlib import redirect_stdout
from decimal import Decimal
from .utils import param_filename


def validate_model_loss(model, train_data, train_labels, test_data, test_labels, parameters):
    # Print MSE and MAE
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    param_str = parameters['filename']
    file_name = os.path.join(path, 'models', 'logs', 'log' + param_str + '.txt')
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


def as_si(x, ndp):
    # Number formatting
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    if 'e' in s:
        m, e = s.split('e')
        st = r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))
    else:
        st = x
    return st

def create_plot(labels, predictions, color, file_name, parameters, hf, hf_labels=False):
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    # Create scatterplot test_predictions vs test_labels
    alpha = 0.1
    marker = ','
    size = 0.5
    hf_labels = -hf_labels
    plt.scatter((hf_labels if hf else labels), predictions, alpha=alpha, color=color, edgecolors='none', marker=marker, s=size)  # darkseagreen


    # lims = ([-0.2, 0.42+0.2] if not parameters['negative'] else [-0.5, 0.5])
    if parameters['stencil_size'][0] == 5:
        lims = [-0.6, 0.6]
    elif parameters['stencil_size'][0] == 7:
        lims = [-0.5, 0.5]
    elif parameters['stencil_size'][0] == 9:
        lims = [-0.5, 0.5]
    else:
        lims = [-1, 1]
    lims = [-0.6, 0.6]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predictions')
    if hf:
        # Plot the 45 degree line
        ax.plot(lims, lims, color='gray')
    else:
        # Make everything except the plot white
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white') 
        ax.spines['right'].set_color('white')
        ax.spines['left'].set_color('white')

    # Calculate L2 and Linf error
    error = labels - predictions
    L2 = 1/max(labels) * np.sqrt( np.sum(np.multiply(error, error))/labels.shape[0])
    Li = 1/max(labels) * max(np.abs(error, error))
    MSE = 1/labels.shape[0] * np.sum(np.multiply(error, error)) 

    L2 = as_si(L2, 2)
    Li = as_si(Li, 2)
    MSE = as_si(MSE, 2)
    prt_str1 = '$L_{2}$'
    prt_str2 = '$\,=' + f'{L2}$'
    prt_str3 = '$L_{\infty}$'
    prt_str4 = '$\,=' + f'{Li}$'
    prt_str5 = '$MSE$'
    prt_str6 = '$\,=' + f'{MSE}$'
    # Print error on plot
    if hf:
        ax.text(0.80, 0.225, 'HF:', transform=ax.transAxes, color=color, ha='left')
        ax.text(0.82, 0.20, prt_str1, transform=ax.transAxes, color='k', ha='right')
        ax.text(0.82, 0.20, prt_str2, transform=ax.transAxes, color='k', ha='left')
        ax.text(0.82, 0.175, prt_str3, transform=ax.transAxes, color='k', ha='right')
        ax.text(0.82, 0.175, prt_str4, transform=ax.transAxes, color='k', ha='left')
        ax.text(0.82, 0.15, prt_str5, transform=ax.transAxes, color='k', ha='right')
        ax.text(0.82, 0.15, prt_str6, transform=ax.transAxes, color='k', ha='left')
    else:
        ax.text(0.80, 0.11, 'ML:', transform=ax.transAxes, color='darkturquoise', ha='left')
        ax.text(0.82, 0.085, prt_str1, transform=ax.transAxes, color='k', ha='right')
        ax.text(0.82, 0.085, prt_str2, transform=ax.transAxes, color='k', ha='left')
        ax.text(0.82, 0.06, prt_str3, transform=ax.transAxes, color='k', ha='right')
        ax.text(0.82, 0.06, prt_str4, transform=ax.transAxes, color='k', ha='left')
        ax.text(0.82, 0.035, prt_str5, transform=ax.transAxes, color='k', ha='right')
        ax.text(0.82, 0.035, prt_str6, transform=ax.transAxes, color='k', ha='left')

    # Save plot
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    param_str = parameters['filename'] + parameters['addstring']
    fig.tight_layout()
    fig.savefig(file_name, dpi=150)
    # plt.show()
    plt.close()


def validate_model_plot(model, test_data, test_labels, parameters, test_kappa=False, test_k_labels=False):

    # Get predictions for test data
    test_predictions = model.predict(test_data, batch_size=parameters['batch_size']).flatten()
    if parameters['normalize']:
        kappa_max = 4/parameters['stz_kappa']
        kappa_min = -4/parameters['stz_kappa']
        test_predictions = (test_predictions*(kappa_max-kappa_min)+(kappa_max+kappa_min))/2
        test_labels = (test_labels*(kappa_max-kappa_min)+(kappa_max+kappa_min))/2

    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    param_str = param_filename(parameters, include_plotdata=True) + parameters['addstring']
    # param_str = parameters['filename']

    if (parameters['hf'] == 'hf') or (parameters['hf'] == 'cd'):
        # Calculate error norm (L2, L infinity)
        error_ml = test_labels - test_predictions
        error_hf = test_k_labels - test_kappa
        L2_ml = 1/max(test_labels) * np.sqrt( np.sum(np.multiply(error_ml, error_ml))/test_labels.shape[0])
        L2_hf = 1/max(test_labels) * np.sqrt( np.sum(np.multiply(error_hf, error_hf))/test_labels.shape[0])
        Linf_ml = 1/max(test_labels) * max(np.abs(error_ml, error_ml))
        Linf_hf = 1/max(test_labels) * max(np.abs(error_hf, error_hf))
        # Create ML Plot
        file_name_ml = os.path.join(path, 'models', 'figures', 'fig' + param_str + '_ml.png')
        create_plot(labels=test_labels, predictions=test_predictions, color='aqua', file_name=file_name_ml, parameters=parameters, hf=False)  # steelblue

        # Create HF Plot
        file_name_hf = os.path.join(path, 'models', 'figures', 'fig' + param_str + '_hf.png')
        create_plot(labels=test_k_labels, predictions=test_kappa, color='deeppink', file_name=file_name_hf, parameters=parameters, hf=True, hf_labels = test_k_labels)  # darkgoldenrod

        # Blend the two plots with image magick
        file_name = os.path.join(path, 'models', 'figures', 'fig' + param_str + '.png')
        # blend = 'stamp'
        blend = 'plus'
        os.system(f"convert {file_name_ml} {file_name_hf} -compose {blend} -composite {file_name}")
        # Delete temp files
        os.system(f"rm {file_name_ml}")
        os.system(f"rm {file_name_hf}")
    else:
        # Create Plot
        file_name = os.path.join(path, 'models', 'figures', 'fig' + param_str + '.png')
        create_plot(labels=test_labels, predictions=test_predictions, color='b', file_name=file_name, parameters=parameters)
