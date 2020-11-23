def param_filename(parameters, include_plotdata=False, plotdata_as_data=False):
    # Generate filename string
    filename_string = ''
    for key, value in parameters.items():
        # Network related keys
        if key == 'layers':
            filename_string = filename_string + '_' + '-'.join(str(e) for e in value)
        elif key == 'angle':
            filename_string = filename_string
        elif key == 'rotate':
            filename_string = filename_string + '_' + ('rot' if value else 'nrt')
        elif key == 'addstring':
            filename_string = filename_string
        elif key == 'hf_correction':
            filename_string = filename_string
        elif key == 'hf':
            filename_string = filename_string
        elif (key == 'plotdata') and (not include_plotdata):
            filename_string = filename_string
        elif key == 'filename':
            filename_string = filename_string
        elif key == 'epochs':
            filename_string = filename_string + '_' + str(value)
        elif key == 'network':
            filename_string = filename_string + '_' + str(value)
        elif key == 'bias':
            # filename_string = filename_string + '_' + ('bia' if value else 'nbi')
            filename_string = filename_string
        elif key == 'cut':
            # filename_string = filename_string + '_' + ('cut' if value else 'nct')
            filename_string = filename_string
        elif key == 'custom_loss':
            # filename_string = filename_string + '_' + ('cls' if value else 'ncl')
            filename_string = filename_string
        elif ((key == 'shift') and (parameters['shift'] != 0)):
            filename_string = filename_string + '_' + 'shift' + str(value)
        elif key == 'edge':
            if ((not parameters['edge2']) and (not parameters['unsharp_mask'])):
                filename_string = filename_string + '_' + ('edg' if value else 'ned')
        elif key == 'edge2':
            if ((not parameters['edge']) and (not parameters['unsharp_mask'])):
                filename_string = filename_string + '_' + ('ed2' if value else 'ne2')
        elif key == 'unsharp_mask':
            if ((not parameters['edge']) and (not parameters['edge2'])):
                filename_string = filename_string + '_' + ('usm' if value else 'num')
        elif (key == 'batch_size') and (parameters['batch_size'] != 128):
            # filename_string = filename_string + '_' + 'bs' + str(value)
            filename_string = filename_string
        elif (key == 'seed') and (parameters['seed']):
            filename_string = filename_string + '_' + 's' + str(value)
        elif key == 'flip':
            # filename_string = filename_string + '_' + ('flp' if value else 'nfp')
            filename_string = filename_string
        elif key == 'stencil_size':
            filename_string = filename_string + '_' + str(value[0]) + 'x' + str(value[1])
        elif key == 'stz_kappa':
            filename_string = filename_string + '_' + str(value)
        elif (key == 'normalize') and (parameters['normalize'] == True):
            filename_string = filename_string + '_norm'
        elif (key == 'learning_rate') and (parameters['learning_rate'] != 5e-5):
            filename_string = filename_string + '_' + str(value)
            # filename_string = filename_string
        elif (key == 'amount') and (parameters['unsharp_mask']):
            filename_string = filename_string + '_a' + str(value)
        elif key == 'labels_min':
            filename_string = filename_string
        elif key == 'labels_max':
            filename_string = filename_string
        else:
            filename_string = filename_string


        if len(parameters['load_data']) > 0:
            if (key == 'load_data') and (len(parameters['load_data']) > 0):
                filename_string = filename_string + '_' + str(value)
        else:
            # Data related keys (not needed when data is explicitly set)
            if key == 'equal_kappa':
                filename_string = filename_string
            elif key == 'negative':
                filename_string = filename_string
            elif key == 'smear':
                filename_string = filename_string
            elif (key == 'data') and (plotdata_as_data):
                filename_string = filename_string
                # filename_string = filename_string + '_' + parameters['plotdata']
            elif (key == 'data') and (not (plotdata_as_data)):
                filename_string = filename_string
                # filename_string = filename_string + '_' + parameters['data']
            elif key == 'gauss':
                filename_string = filename_string + ('_g' if value else '')
            elif ((key == 'dshift') and (parameters['dshift'] != '0')):
                filename_string = filename_string + '_' + 'dshift' + str(value)
            elif ((key == 'interpolate_lower') and (parameters['interpolate'] != 0)):
                filename_string = filename_string + '_' + 'ab' + str(value)
            elif ((key == 'interpolate') and (parameters['interpolate'] != 0)):
                filename_string = filename_string + '_' + 'int' + str(value)
    return filename_string
