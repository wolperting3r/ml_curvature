import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib as tkz
import regex
import re
from scipy import signal
from pprint import pprint

def cal_ypos(path):
    filepath = os.path.join(path, 'y_pos_re.txt')
    filepath_ext = os.path.join('/Volumes','Daten','Archive','Masterarbeit','fortran_test','dynamic','model_mlp_1000_100-80_7x7_eqk_0.0001_128_relu_neg_nag_rot_all_smr_nhc_128_allstc', path, 'y_pos_re.txt')
    if os.path.isfile(filepath):
        data = pd.read_csv(filepath, skiprows=1, names=['it', 'x', 'y', 'c', 'cm'])
    elif os.path.isfile(filepath_ext):
        data = pd.read_csv(filepath_ext, skiprows=1, names=['it', 'x', 'y', 'c', 'cm'])
    else:
        print(f'file {path} not found!')

    diffpos = 0
    diffpos = data['it'].value_counts().iloc[0]

    # Cut after endtime
    data = data.iloc[:np.int32(diffpos*(endtime+0.5)*1/timestep)+diffpos, :]

    # Convert to numpy array
    data = data.values

    # Reshape cm into 2D-array
    c = np.reshape(data[:, 4], (int(data.shape[0]/diffpos), diffpos))
    # Cut off below and above threshold
    tolerance = 1e-2
    c[np.nonzero(c < tolerance)] = 0
    c[np.nonzero(c > 1-tolerance)] = 1

    # Reshape y into 2D-array
    y = np.reshape(data[:, 2], (int(data.shape[0]/diffpos), diffpos))

    # Find closest value to c=0.5 where c>0.5
    indices = np.argmin(np.where((c-0.5) < 0, c.max(), c), axis=1)
    c_upper = c[range(c.shape[0]), indices]
    y_upper = y[range(y.shape[0]), indices]

    # Find closest value to c=0.5 where c<0.5
    indices = np.argmax(np.where(-(c-0.5) <= 0, 0, c), axis=1)
    c_lower = c[range(c.shape[0]), indices]
    y_lower = y[range(y.shape[0]), indices]

    # Make array with right dimensions filled with 0.5
    c_mid = c_lower.copy()
    c_mid[:] = 0.5

    # Get y where c = 0.5 by linear interpolation
    return (y_lower + (c_mid-c_lower)/(c_upper-c_lower)*(y_upper-y_lower))

if __name__ == '__main__':
    # ld = ['int0', 'int1', 'int2', 'fnb', 'cvofls', 'cvofls_edge', 'fnb_64', 'fnb_256', 'fnb_9x9_128', 'fnb_9x9_64', 'fnb_9x9_256', 'hfcv', '1680int', '1680g']
    ld = ['edge2_200']
    gridsize = 128
    percentiles = True
    testrun = False
    metrics = False # Calculate metrics (and do not plot)
    for loaddata in ld:
        if not metrics:
            if not testrun:
                # fig, ax = plt.subplots(1, 1, figsize=(10,5))
                # Make figure without border
                fig = plt.figure(frameon=False)
                fig.set_size_inches(10,5)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
            else:
                fig, ax = plt.subplots(figsize=(10,5))

        # '''
        n_folders = 16
        # n_folders = 3
        if loaddata == 'int0':
            # Vergleich selbes Netz int 0
            basepath = '2007212130 int 0 s'
        elif loaddata == 'int1':
            # Vergleich selbes Netz int 1
            basepath = '2007221249 int 1 s'
        elif loaddata == 'int2':
            # Vergleich selbes Netz int 2
            basepath = '2007211250 int 2 s'
        elif loaddata == 'edge2_128':
            basepath = '2010221138 Edge2 7x7 epsilon 1e-10 1e-3 1.5 s'
            n_folders = 4
        elif loaddata == 'edge2_200':
            basepath = '2010271111 Edge2 200 s'
            n_folders = 4
        elif loaddata == 'fnb':
            basepath = '2008031931 Edge neu Relaxation FNB Ellipse s'
        elif loaddata == 'fnb_64':
            basepath = '2008111045 Edge neu Relaxation FNB Ellipse 64 s'
        elif loaddata == 'fnb_256':
            basepath = '2008111652 Edge neu Relaxation FNB Ellipse 256 7x7 s'
            n_folders = 4
        elif loaddata == 'fnb_9x9_64':
            basepath = '2008131200 Edge neu Relaxation FNB Ellipse 64 9x9 s'
        elif loaddata == 'fnb_9x9_128':
            basepath = '2008121704 Edge neu Relaxation FNB Ellipse 128 9x9 s'
            n_folders = 15
        elif loaddata == 'fnb_9x9_256':
            basepath = '2008120839 Edge neu Relaxation FNB Ellipse 256 9x9 s'
        elif loaddata == 'cvofls':
            basepath = '2008071152 CVOFLS ML s'
        elif loaddata == 'cvofls_edge':
            basepath = '2008072025 CVOFLS ML edge s'
        elif loaddata == '1680int':
            basepath = '2008141034 16x80 int Test s'
            n_folders = 4
        elif loaddata == '1680g':
            basepath = '2008141825 16x80 Test g s'

        if loaddata == 'hfcv':
            paths = ['2008131109 CV no w+g 128', '2006022000 cds Vergleich']
        else:
            paths = []
            if loaddata != 'no':
                stencils = range(n_folders)
                # stencils = [2, 8, 10, 12, 13, 14, 15] # 64 ohne Hügel
                for i in stencils:
                    paths.append(f'{basepath}{i+1}')

        if gridsize == 64:
            paths.append('2008161613 HF 64')
            paths.append('2008161722 CVOFLS 64')
        elif gridsize == 128:
            paths.append('2008131109 HF 128')
            paths.append('2006031403 CVOFLS')
        elif gridsize == 256:
            paths.append('2008161613 HF 256')
            paths.append('2008161722 CVOFLS 256')

        colors=iter(plt.cm.rainbow(np.linspace(0,1,(len(paths)))))
        reds = ['tomato', 'red', 'maroon', 'brown']
        j=0
        k=0
        # endtime = 2  # in s
        endtime = 2
        timestep = 0.000625

        refdata = pd.read_csv('Strubelj_128.txt', skiprows=1, names=['t', 'CVOFLS Paper'])
        refdata['t'] = refdata['t']*1/timestep
        refdata = refdata.set_index('t')
        # refdata.plot(ax=ax, label='CVOFLS (Paper)', color='darkgray', lw=2, ls='-')

        y_05s = {}
        # Get y-position of interface for all paths
        for path in paths:
            if (('CVOFLS' in path) & (not 'ML' in path)):
                cvofls_pos = cal_ypos(path)
            elif ('HF' in path):
                hf_pos = cal_ypos(path)
            else:
                print(f'path:\t{path}')
                # y_05s[paths.index(path)] = cal_ypos(path)
                y_05s[path] = cal_ypos(path)

        # Make numpy array with all values
        # y_pos = np.empty((y_05s[0].shape[0], len(y_05s.keys())))
        y_pos = np.empty((cvofls_pos.shape[0], len(y_05s.keys())))
        for key in y_05s.keys():
            y_pos[:, list(y_05s.keys()).index(key)] = y_05s[key]

        if metrics:
            # Time window for first low and first high in s:
            t_lo = [int(0.5/timestep), int(1/timestep)]
            t_hi = [int(1.25/timestep), int(1.75/timestep)]
            # Initialize arrays
            lo = np.empty((len(y_05s.keys()), 1))
            hi = np.empty((len(y_05s.keys()), 1))
            keys = list(y_05s.keys())
            # Find times of first low/first high
            for key in y_05s.keys():
                i = keys.index(key)
                lo[i] = np.argmin(y_05s[key][t_lo[0]:t_lo[1]]) + t_lo[0]
                hi[i] = np.argmax(y_05s[key][t_hi[0]:t_hi[1]]) + t_hi[0]
            cvofls_lo = (np.argmin(cvofls_pos[t_lo[0]:t_lo[1]]) + t_lo[0])*timestep
            cvofls_hi = (np.argmax(cvofls_pos[t_hi[0]:t_hi[1]]) + t_hi[0])*timestep
            hf_lo = (np.argmin(hf_pos[t_lo[0]:t_lo[1]]) + t_lo[0])*timestep
            hf_hi = (np.argmax(hf_pos[t_hi[0]:t_hi[1]]) + t_hi[0])*timestep
            # Get average time
            lo_avg = np.mean(lo)*timestep
            hi_avg = np.mean(hi)*timestep
            # print(f'lo_avg:\t{lo_avg}')
            print(f'Average period time ML:\t\t{np.round(hi_avg, 3)} s')
            print(f'Average period time CVOFLS:\t{np.round(cvofls_hi, 3)} s')
            print(f'Average period time HF:\t\t{np.round(hf_hi, 3)} s')

            
        else:
            color_cvofls = 'olivedrab'
            color_hf = 'cadetblue'
            if loaddata == 'int2':
                color_ml = 'chocolate'  # int 2
            elif loaddata == 'int1':
                color_ml = 'goldenrod'  # int 1
            elif loaddata == 'int0':
                color_ml = 'lightseagreen'  # int 0
            elif loaddata == 'fnb_9x9_256':
                color_ml = 'orchid'
            elif loaddata == 'fnb_9x9_128':
                color_ml = 'mediumorchid'
            elif loaddata == 'fnb_9x9_64':
                color_ml = 'darkorchid'
            elif loaddata == 'fnb':
                color_ml = 'orchid'
            elif loaddata == 'fnb_64':
                color_ml = 'mediumorchid'
            elif loaddata == 'fnb_256':
                color_ml = 'darkorchid'
            elif loaddata == 'cvofls':
                color_ml = 'royalblue'
            elif loaddata == 'cvofls_edge':
                color_ml = 'deepskyblue'
            elif loaddata == '1680int':
                color_ml = 'crimson'
            elif loaddata == '1680g':
                color_ml = 'crimson'
            elif loaddata == 'edge2_128':
                color_ml = 'orchid'
            elif loaddata == 'edge2_200':
                color_ml = 'orchid'
            else:
                print(f'No color assigned for {path}')
                color_ml = 'k'

            # Get filter parameters
            b, a = signal.butter(3, 0.005, btype='low', analog=False)

            # Calculate, filter and plot median
            median = np.median(y_pos, axis=1)
            median = signal.filtfilt(b, a, median)

            lsp = np.linspace(0, median.shape[0], median.shape[0])
            n_val = 10
            if percentiles:
                labels = iter(['10%-Percentile'] + ['_nolabel_']*(n_val))
                for i in range(n_val):
                    # Calculate, filter and plot lower to upper percentil
                    up_perc = np.percentile(y_pos, 100-(i)/n_val*50, axis=1, interpolation='midpoint')
                    lo_perc = np.percentile(y_pos, (i)/n_val*50, axis=1, interpolation='midpoint')

                    up_perc = signal.filtfilt(b, a, up_perc)
                    lo_perc = signal.filtfilt(b, a, lo_perc)

                    # plt.fill_between(lsp, lo_perc, up_perc, alpha=0.8/n_val, color=color_ml, label=next(labels))
                    plt.fill_between(lsp, lo_perc, up_perc, alpha=0.8/n_val, color=color_ml)

            if not testrun:
                labels = iter(['ANNs'] + ['_nolabel_']*(len(y_05s.keys())))
            else:
                lenlab = len(paths)
                labels = []
                for i in range(lenlab-1):
                    labels.append(str(i))
                labels.append('CV')
                labels.append('CVOFLS')
                labels = iter(labels)
            # '''
            for key in y_05s.keys():
                # Plot y-positions
                if testrun:
                    color_ml = next(colors)
                if percentiles:
                    plt.plot(y_05s[key], c=color_ml, alpha = 0.3, lw=0.5, label=next(labels))  # Percentile
                else:
                    alpha = 0.5
                    lw = 1
                    if '2008131109 CV' in key:
                        color_ml = 'deepskyblue'
                        alpha = 1
                        lw = 3
                    elif '2008131109 HF' in key:
                        color_ml = 'darkorchid'
                        alpha = 1
                        lw = 3
                    plt.plot(y_05s[key], c=color_ml, alpha = (1 if testrun else alpha), lw=lw, label=next(labels))  # Nur Linien

            # '''

            # Plot median (50%-Percentil)
            line_width = 3
            if percentiles:
                plt.plot(median, c=color_ml, lw=line_width, label='ANN Median')
            plt.plot(cvofls_pos, c=color_cvofls, lw=line_width, label='CVOFLS')
            plt.plot(hf_pos, c=color_hf, lw=line_width, label='HF')


            '''
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x*timestep}'))
            plt.xticks(np.arange(start=0, step=1.5888/(2*timestep), stop=int((endtime))/timestep))

            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '%.2E' % np.round((y-0.075/2)*0.1, 4)))
            plt.yticks(np.arange(start=(0.002*10)+0.075/2, step=0.005 , stop=(0.003*10)+0.075/2))
            ax.set_ylabel('y-Pos [m]')
            ax.set_xlabel('Zeit [s]')
            # '''
            ax.set_ylim([0.0540,0.0661])
            # ax.set_ylim([0.053,0.068])
            ax.set_xlim([0, ((endtime)/timestep)])

            for i in range(int(np.floor(2*endtime/1.5888)+1)):
                plt.axvline(x=i/2*1.5888/timestep, color='k', lw=0.5)

            # fig.tight_layout()
            # Set opacity of entries in legend to 1
            '''
            leg = plt.legend()
            for lh in leg.legendHandles: 
                # pprint(vars(lh))
                if len(lh.get_label()) == 0:  # Percentile
                    lh.set_alpha(0.3)
                else:
                    lh.set_alpha(1)
            # '''
            # '''
            # '''
            # Export tikz file
            # tkz.save('result.tex', axis_height='7cm', axis_width='15cm', extra_axis_parameters={'scaled y ticks=manual:{$\cdot10^{-3}$}{\pgfmathparse{#1-1}}'})
            # '''
            tkz.save('result.tex', axis_height='7cm', axis_width='15cm')
            with open('result.tex', 'r') as myfile:
                filedata = myfile.read()
                filedata = re.sub('semithick', 'ultra thick', filedata)
                filedata = regex.sub(r'(?<=yticklabels\=\{.*)(\d\.\d)\d*E\-03', r'\1', filedata)

            with open('result.tex', 'w') as myfile:
                myfile.write(filedata)

            plt.savefig('Streuung_'+loaddata+'.svg', dpi=150)
            os.system('inkscape -D Streuung_'+loaddata+'.svg -o Streuung_'+loaddata+('_noperc' if not percentiles else '') + '.pdf --export-latex')
            # '''
            if testrun:
                ax.legend()
                plt.show()
