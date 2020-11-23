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
        # print(f'file {path} not found!')
        return [-1]

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
    ld = [ '200-100-25_worst', '200-100-25_best'] 
    # ld.remove('fastest')
    # ld.remove('tanh_edge2_noint_p100_splitseed')
    endtime = 3
    gridsize = 128
    percentiles = False
    testrun = True
    metrics = False # Calculate metrics (and do not plot)

    if metrics and 'fastest' in ld:
        ld.remove('fastest')
    colors=iter(plt.cm.rainbow(np.linspace(0,1,(len(ld)))))
    reds = ['tomato', 'red', 'maroon', 'brown']
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
            fig, ax = plt.subplots(figsize=(20,10))
    for loaddata in ld:
        # '''
        n_folders = 16
        # n_folders = 3

        if loaddata == 'int0':
            basepath = '2007212130 int 0 s'
            color_ml = 'lightseagreen'  # int 0
        elif loaddata == 'int1':
            basepath = '2007221249 int 1 s'
            color_ml = 'goldenrod'  # int 1
        elif loaddata == 'int2':
            basepath = '2007211250 int 2 s'
            color_ml = 'chocolate'  # int 2
        elif loaddata == 'fnb':
            basepath = '2008031931 Edge neu Relaxation FNB Ellipse s'
            color_ml = next(colors)
        elif loaddata == 'fnb_64':
            basepath = '2008111045 Edge neu Relaxation FNB Ellipse 64 s'
            color_ml = 'mediumorchid'
        elif loaddata == 'fnb_256':
            basepath = '2008111652 Edge neu Relaxation FNB Ellipse 256 7x7 s'
            color_ml = 'darkorchid'
            n_folders = 4
        elif loaddata == 'fnb_9x9_64':
            basepath = '2008131200 Edge neu Relaxation FNB Ellipse 64 9x9 s'
            color_ml = 'darkorchid'
        elif loaddata == 'fnb_9x9_128':
            # color_ml = 'mediumorchid'
            color_ml = next(colors)
            basepath = '2008121704 Edge neu Relaxation FNB Ellipse 128 9x9 s'
            n_folders = 15
        elif loaddata == 'fnb_9x9_256':
            basepath = '2008120839 Edge neu Relaxation FNB Ellipse 256 9x9 s'
            color_ml = 'orchid'
        else:
            color_ml = next(colors)
            n_folders = 4
            if loaddata == 'edge2_15s':
                basepath = '2010221138 Edge2 7x7 epsilon 1e-10 1e-3 1.5 s'
            elif loaddata == 'edge2_4.11_int1':
                basepath = '2011021022 Edge2 4.11 int 1 s'
            elif loaddata == 'edge2_4.11_int1.5':
                basepath = '2011021022 Edge2 4.11 int 1.5 s'
            elif loaddata == 'edge2_4.11_int2':
                basepath = '2011021022 Edge2 4.11 int 2 s'
            elif loaddata == 'edge2_4.25_int1':
                basepath = '2011021022 Edge2 4.25 int 1 s'
            elif loaddata == 'edge2_4.25_int1.5':
                basepath = '2011021022 Edge2 4.25 int 1.5 s'
            elif loaddata == 'edge2_4.25_int2':
                basepath = '2011021022 Edge2 4.25 int 2 s'
            elif loaddata == 'edge2_ab1_int1.5':
                basepath = '2011031906 Edge2 ab1 int 1.5 s'
                n_folders = 8
            elif loaddata == 'edge2_ab1_int2':
                basepath = '2011030829 Edge2 ab1 int 2 s'
            elif loaddata == 'edge2_ab0.75_int1.5':
                basepath = '2011040911 Edge2 ab0.75 int 1.5 s'
            elif loaddata == 'edge2_ab0.5_int1.5':
                basepath = '2011041208 Edge2 ab0.5 int 1.5 s'
                n_foldes = 3
            elif loaddata == 'edge2_ab0.5_int1.5_e-10':
                basepath = '2011041605 Edge2 ab0.5 int 1.5 e-10 s'
            elif loaddata == 'edge2_tanh':
                basepath = '2011041324 Edge2 tanh int 0.5-1.5 s'
            elif loaddata == 'edge2_tanh_e-10':
                basepath = '2011041324 Edge2 tanh int 0.5-1.5 e-10 s'
            elif loaddata == 'edge2_tanh_e-10_15':
                basepath = '2011041324 Edge2 tanh int 0.5-1.5 e-10 15 s'
            elif loaddata == 'edge2_tanh_noint':
                basepath = '2011051017 tanh ohne int mit edge2 s'
            elif loaddata == 'edge2_tanh_noint_15':
                basepath = '2011051017 tanh ohne int mit edge2 15s s'
                n_folders = 8
            elif loaddata == 'noedge2_tanh_int':
                basepath = '2011051017 tanh mit int ohne edge2 s'
            elif loaddata == 'noedge2_tanh_noint':
                basepath = '2011051017 tanh ohne int ohne edge2 s'
            elif loaddata == 'tanh_9x9':
                basepath = '2011051742 tanh 9x9 s'
            elif loaddata == 'tanh_20':
                basepath = '2011052032 tanh 20 ohne int mit edge2 s'
            elif loaddata == 'relu_20_noedge':
                basepath = '2010301918 Tensorflow 20 s'
            elif loaddata == 'relu_20_edge':
                basepath = '2010301420 Edge2 Tensorflow 20 s'
            elif loaddata == 'tanh_25_noedge':
                basepath = '2011111756 25 no edge2 s'
            elif loaddata == 'tanh_25_edge':
                basepath = '2011111756 25 edge2 s'
            elif loaddata == 'tanh_200-150-100_edge':
                basepath = '2011121450 200-150-100 edge2 s'
            elif loaddata == 'tanh_edge2_noint_p100':
                basepath = '2011061758 tanh ohne int mit edge2 5e-5 p100 s'
                n_folders = 16
            elif loaddata == 'tanh_edge2_noint_p100_splitseed':
                basepath = '2011070955 tanh ohne int mit edge2 5e-5 p100 splitseed s'
                n_folders = 4
            elif loaddata == '200-100-25_best':
                basepath = '2011171529 200-100-25 edge2 beste s'
                n_folders = 18
            elif loaddata == '200-100-25_worst':
                basepath = '2011171641 200-100-25 edge2 schlechteste s'
                n_folders = 18
            elif loaddata == 'fastest':
                basepath = 'FASTEST_'

        if loaddata == 'hfcv':
            paths = ['2008131109 CV no w+g 128', '2006022000 cds Vergleich']
        else:
            paths = []
            if loaddata != 'no':
                stencils = range(n_folders)
                # stencils = [2, 8, 10, 12, 13, 14, 15] # 64 ohne Hügel
                for i in stencils:
                    if basepath != '2011041208 Edge2 ab0.5 int 1.5 s':
                        paths.append(f'{basepath}{i+1}')
                    elif i < 3:
                        paths.append(f'{basepath}{i+2}')

        if gridsize == 64:
            paths.append('2008161613 HF 64')
            paths.append('2008161722 CVOFLS 64')
        elif gridsize == 128:
            paths.append('2008131109 HF 128')
            paths.append('2006031403 CVOFLS')
        elif gridsize == 256:
            paths.append('2008161613 HF 256')
            paths.append('2008161722 CVOFLS 256')

        j=0
        k=0
        # endtime = 2  # in s
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
                # y_05s[paths.index(path)] = cal_ypos(path)
                ypos_return = cal_ypos(path)
                if not len(ypos_return) == 1:
                    print(f'path:\t{path}')
                    y_05s[path] = ypos_return

        # Make numpy array with all values (legth = shortest length)
        length = 1e20
        for key in y_05s.keys():
            length = min(y_05s[key].shape[0], length)
        y_pos = np.empty((length, len(y_05s.keys())))
        for key in y_05s.keys():
            y_pos[:, list(y_05s.keys()).index(key)] = y_05s[key][:length]

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
            hi_std = np.std(hi)*timestep
            # Get standard deviation at high point:
            values = np.empty((len(y_05s.keys()), 1))
            for key in y_05s.keys():
                i = keys.index(key)
                values[i] = y_05s[key][int(hi_avg/timestep)]
            stddv = np.std(values)
            mean = np.mean(values)
            print(f'Standard deviation:\t\t{np.round(stddv, 7):.2E}')
            print(f'Mean Amplitude:\t\t\t{np.round(mean, 5)}')
            print(f'CVOFLS Amplitude:\t\t{np.round(cvofls_pos[int(cvofls_hi/timestep)], 5)}')
            print(f'Average period time ML:\t\t{np.round(hi_avg, 3)} s')
            print(f'ML period time standard dev.:\t{np.round(hi_std, 3)} s')
            print(f'Average period time CVOFLS:\t{np.round(cvofls_hi, 3)} s')
            print(f'Average period time HF:\t\t{np.round(hf_hi, 3)} s')
            print(f'Analytical period time:\t\t1.589 s')

            
        else:
            color_cvofls = 'olivedrab'
            color_hf = 'cadetblue'
            if percentiles:
                # Get filter parameters
                b, a = signal.butter(3, 0.005, btype='low', analog=False)

                # Calculate, filter and plot median
                median = np.median(y_pos, axis=1)
                median = signal.filtfilt(b, a, median)

                lsp = np.linspace(0, median.shape[0], median.shape[0])
                n_val = 10
                labels = iter(['10%-Percentile'] + ['_nolabel_']*(n_val))
                for i in range(n_val):
                    # Calculate, filter and plot lower to upper percentil
                    up_perc = np.percentile(y_pos, 100-(i)/n_val*50, axis=1, interpolation='midpoint')
                    lo_perc = np.percentile(y_pos, (i)/n_val*50, axis=1, interpolation='midpoint')

                    up_perc = signal.filtfilt(b, a, up_perc)
                    lo_perc = signal.filtfilt(b, a, lo_perc)

                    # plt.fill_between(lsp, lo_perc, up_perc, alpha=0.8/n_val, color=color_ml, label=next(labels))
                    plt.fill_between(lsp, lo_perc, up_perc, alpha=0.8/n_val, color=color_ml)

            # if not testrun:
            # labels = iter(['ANNs'] + ['_nolabel_']*(len(y_05s.keys())))
            '''
            else:
                lenlab = len(paths)
                labels = []
                for i in range(lenlab-1):
                    labels.append(str(i))
                labels.append('CV')
                labels.append('CVOFLS')
                labels = iter(labels)
            # '''
            if percentiles:
                labels = iter(['_nolabel_']*(len(y_05s.keys())+1))
            else:
                labels = iter([loaddata] + ['_nolabel_']*(len(y_05s.keys())))
            # '''
            for key in y_05s.keys():
                # Plot y-positions
                # if testrun:
                    # color_ml = next(colors)
                if percentiles:
                    plt.plot(y_05s[key], c=color_ml, alpha = 0.3, lw=0.5, label=next(labels))  # Percentile
                else:
                    # alpha = 0.5
                    alpha = 1
                    lw = 2
                    if '2008131109 CV' in key:
                        color_ml = 'deepskyblue'
                        alpha = 1
                        lw = 3
                    elif '2008131109 HF' in key:
                        color_ml = 'darkorchid'
                        alpha = 1
                        lw = 3
                    plt.plot(y_05s[key], c=color_ml, alpha = alpha, lw=lw, label=next(labels))  # Nur Linien

            # '''

            labels = iter([loaddata] + ['_nolabel_']*(len(y_05s.keys())))
            # Plot median (50%-Percentil)
            line_width = 3
            if percentiles:
                plt.plot(median, c=color_ml, lw=line_width, label=next(labels))

    if not metrics:
        plt.plot(cvofls_pos, c=color_cvofls, lw=line_width, label='CVOFLS')
        plt.plot(hf_pos, c=color_hf, lw=line_width, label='HF')

        # '''
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

        '''
        tkz.save('result.tex', axis_height='7cm', axis_width='15cm')
        with open('result.tex', 'r') as myfile:
            filedata = myfile.read()
            filedata = re.sub('semithick', 'ultra thick', filedata)
            filedata = regex.sub(r'(?<=yticklabels\=\{.*)(\d\.\d)\d*E\-03', r'\1', filedata)

        with open('result.tex', 'w') as myfile:
            myfile.write(filedata)
        # '''

        plt.savefig('Streuung_'+loaddata+'.svg', dpi=150)
        os.system('inkscape -D Streuung_'+loaddata+'.svg -o Streuung_'+loaddata+('_noperc' if not percentiles else '') + '.pdf --export-latex')
        # '''
        if testrun:
            ax.legend()
            plt.show()
