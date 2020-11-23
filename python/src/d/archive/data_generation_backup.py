import numpy as np
import itertools
import time
import pandas as pd
from progressbar import *
import matplotlib.pyplot as plt
import os
import sys

from .utils import (
    gt,
    plot_circle,
    plot_ellipse,
    plot_vof,
    u,
    pm,
)


def axismatrix(n_grid, axis):
    # Generate matrix where values increase along the given axis starting at 0
    n_grid = int(n_grid)
    return_matrix = np.tile(np.array(range(n_grid))/n_grid, (n_grid, 1))
    if axis == 0:
        return return_matrix.transpose((1, 0))[::-1]
    elif axis == 1:
        return return_matrix


def cross(mid_pt, max_pt, rev_y=False):
    # Generate cross values mid_pt - max_pt to mid_pt + max_pt in both axis
    # Generate points in direction of both axis

    # rev_y: reverse order in y-direction (required for cross_point_origins)
    if rev_y:
        # First take negative y values to get sorting right
        mid_pt = [-mid_pt[0], mid_pt[1]]

    # Get limits in x-direction
    points_x = np.array([np.array([mid_pt[0]]), mid_pt[1]+max_pt[1]])
    # Get limits in y-direction
    points_y = np.array([mid_pt[0]+max_pt[0], np.array([mid_pt[1]])])
    # Get points in x- and y-direction
    cr_x = np.array(list(itertools.product(*points_x)))
    cr_y = np.array(list(itertools.product(*points_y)))
    # Combine them into a list
    cross_points = np.unique(np.concatenate((cr_y, cr_x), axis=0), axis=0)

    if rev_y:
        # Now invert y values again and transpose it to original shape: np.array().transpose((1, 0))
        cross_points = np.array([-cross_points[:, 0], cross_points[:, 1]]).transpose((1, 0))

    # Get list of all unique cross points
    return cross_points


def generate_data(N_values, stencils, ek, neg, silent, ellipse, smearing):
    '''
    print(f'N_values:\t{N_values}')
    print(f'stencils:\t{stencils}')
    print(f'ek:\t{ek}')
    print(f'neg:\t{neg}')
    print(f'silent:\t{silent}')
    print(f'ellipse:\t{ellipse}')
    print(f'smearing:\t{smearing}')
    # '''
    print(f'Generating data:\nEllipse:\t{ellipse}\nStencil:\t{stencils}\nKappa:\t\t{ek}\nNeg. Values\t{neg}\nN_values:\t{int(N_values)}\nSmearing:\t{smearing}')
    time0 = time.time()

    # Script
    N_values = int(N_values)
    visualize = True if (N_values == 1) else False
    debug = False

    # Initialize progress bar
    widgets = ['Data generation: ', Percentage(), ' ', Bar(marker='=',left='[',right=']'), ' ', ETA()]
    if not silent:
        pbar = ProgressBar(widgets=widgets, maxval=N_values)
        pbar.start()

    # Grid
    Delta = 1/1000
    Delta_vof = 1/32
    L = 1

    # Stencil
    st_sz = stencils   # y, x
    cr_sz = [3, 3]  # y, x
    if smearing:
        # Increase stencil size by two until smearing is applied
        st_sz = np.add(st_sz, [2, 2])

    # Geometry
    R_min = max(st_sz)/2*Delta
    R_max = 0.5

    # kappa_min = L*Delta*2/R_max
    kappa_min = 1e-5
    kappa_max = L*Delta*2/R_min
    equal_kappa = ek

    e_min = 1.0000001
    e_max = 10
    # e_maxmin = 5
    # e_maxmax = 10


    # Calculate midpoints of stencil and cross
    st_mid = [int((st_sz[0]-1)/2), int((st_sz[1]-1)/2)]
    cr_mid = [int((cr_sz[0]-1)/2), int((cr_sz[1]-1)/2)]

    # Generate x and y for origins of local grid for cross points relative to local point
    st_crp = np.array([np.arange(-cr_mid[0], (cr_mid[0]+1)),
                       np.arange(-cr_mid[1], (cr_mid[1]+1))])*Delta
    # Generate x and y for origins of local grid for stencils points relative to local point
    st_stp = np.array([np.arange(-st_mid[0], (st_mid[0]+1)),
                       np.arange(-st_mid[1], (st_mid[1]+1))])*Delta
    # Generate local grid (y from bottom to top, x from left to right)
    local_grid = np.array([axismatrix(1/Delta_vof, 0),
                           axismatrix(1/Delta_vof, 1)])*Delta

    # Initialize list for output vectors
    output_list = []

    for n in range(N_values):
        if not silent:
            # Update progress bar
            pbar.update(n)
        ''' Get random curvature and radius '''
        if ellipse:
            # Get random curvature
            curvature = -(kappa_min + u()*(kappa_max - kappa_min))
            # Get random side ratio
            # e_max = e_maxmax + (curvature/(-kappa_max))**0.5*(e_maxmin - e_maxmax)
            # e_max = 2
            e = e_min+u()*(e_max-e_min)
            # Calculate ellipse radius
            # r = -L*Delta*2/curvature*e**(-1 +(curvature/(-kappa_max)) + (3-(curvature/(-kappa_max)))*u()**1.5)
            # '''
            r_min = max([2*L*Delta/(-curvature*e), (max(st_sz)*Delta*e**2)/2])
            r_max = 2*e**2*L*Delta/(-curvature)
            r = r_min + u()**1* (r_max - r_min)
            '''
            r_min_b = 2*L*Delta/(-curvature*e)
            print(f'curvature:\t{curvature}')
            print(f'e:\t\t{e}')
            print(f'\nr_max:\t\t{np.round(r_max, 5)}')
            print(f'r_min:\t\t{np.round(r_min, 5)}')
            print(f'r_min_b:\t{np.round(r_min_b, 5)}')
            print(f'r:\t\t{np.round(r, 5)}')
            # '''
            # '''
            # r = -L*Delta*2/curvature*e**(-1 +(curvature/(-kappa_max)) + (3-(curvature/(-kappa_max)))*u()**1.5)
        else:
            # Get random curvature
            curvature = -(kappa_min + u()*(kappa_max - kappa_min))
            # Calculate radius
            r = -L*Delta*2/curvature

        # Move midpoint by random amount inside one cell
        x_c = np.array([u(), u()])*Delta
        # x_c = np.array([0, 0])

        ''' Get random point on geometry '''
        if ellipse:
            # Get x and y coordinates of point on ellipse with the given curvature
            # (and flip x and y coordinates randomly with pm)
            nenner = ((-e**2*r**2*L*Delta*2)/curvature)**(2/3)-r**2
            # print(f'nenner:\n{nenner}')
            pt_x = pm()*np.sqrt((((-e**2*r**2*L*Delta*2)/curvature)**(2/3)-r**2)/(e**4-e**2))
            pt_y = pm()*np.sqrt(r**2-e**2*(pt_x)**2)

            # Rotate with random angle
            rot = u()*2*np.pi
            rot_matrix = [[np.cos(rot), -np.sin(rot)],
                          [np.sin(rot), np.cos(rot)]]
            [pt_x, pt_y] = np.matmul(rot_matrix, [pt_x, pt_y])

            # Make x array and add random shift of origin x_c
            x = np.array([pt_y, pt_x])
            x = x+x_c
        else:
            # Get random spherical angle
            theta = 2*np.pi*u()
            # Get cartesian coordinates on sphere surface
            x_rel = np.array([r*np.sin(theta),   # y
                              r*np.cos(theta)])  # x
            x = np.array([x_c[0]+x_rel[0],
                          x_c[1]+x_rel[1]])

        # Round point to get origin of local coordinates in global coordinates relative to geometry origin
        round_point = np.floor(x*1/Delta*L)*Delta/L

        ''' Plot Ellipse/Circle '''
        if visualize:
            # Initialize plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
            if ellipse:
                # Plot ellipse
                plot_ellipse(ax1, r, e, x, x_c, rot, curvature)
            else:
                # Plot circle
                plot_circle(ax1, r, x_c, x)

        ''' 1. Evaluate VOF values on cross around origin to get gradient (if stencil is not quadratic) '''
        if st_sz[0] != st_sz[1]:
            # Initialize vof_array and fill it with nan
            vof_array = np.empty((cr_sz[0], cr_sz[1]))
            vof_array[:] = np.nan

            if visualize:
                # Create pandas dataframe to fetch shape of geometry in local coordinates
                vof_df = pd.DataFrame(index=range(cr_sz[0]), columns=range(cr_sz[1]))

            # Generate cross points in x and y direction
            cross_point_origins = cross(round_point, st_crp, rev_y=True)
            cross_point_indices = cross(cr_mid, st_crp/Delta).astype(int)
            for idx, ill in enumerate(cross_point_indices):
                # Get origin coordinates too
                lo = cross_point_origins[idx]
                # Values of local grid relative to geometry origin
                # Local grid origin in lower left corner, x: l.t.r., y: b.t.t.
                # (!! y on global grid = local grid origins is from top to bottom!)
                [y_l, x_l] = np.array([local_grid[0] + lo[0] - x_c[0],
                                       local_grid[1] + lo[1] - x_c[1]])
                if ellipse:
                    # Rotate geometry back to calculate ellipse
                    x_ltmp = x_l.copy()
                    y_ltmp = y_l.copy()
                    x_l = x_ltmp*np.cos(rot)+y_ltmp*np.sin(rot)
                    y_l = -x_ltmp*np.sin(rot)+y_ltmp*np.cos(rot)

                    # Get radii on local grid (np.multiply way faster than np.power) r^2 = e^2*x^2 + y^2
                    r_sqr = e**2*np.multiply(x_l, x_l) + np.multiply(y_l, y_l)
                # Calculate 1s and 0s on local grid
                r_area = np.where(r_sqr <= r*r, 1, 0)
                # Get VOF values by integration over local grid
                vof = np.sum(r_area)/r_area.size
                # Write vof value into stencil value array
                vof_array[ill[0], ill[1]] = vof
                if visualize:
                    # Save the r_area array (containing the shape of the geometry) for plotting
                    vof_df.iloc[ill[0], ill[1]] = r_area

            # Calculate gradient with central finite difference:
            grad_y = vof_array[cr_mid[0]+1, cr_mid[1]]-vof_array[cr_mid[0]-1, cr_mid[1]]
            grad_x = vof_array[cr_mid[0], cr_mid[1]+1]-vof_array[cr_mid[0], cr_mid[1]-1]
            # Calculate normal vector
            normal = -1/np.sqrt(grad_y*grad_y+grad_x*grad_x)*np.array([grad_y, grad_x])

            # Extend the stencil in the direction the normal vector points to
            if np.abs(normal[0]) > np.abs(normal[1]):
                # If gradient points more to y-direction
                # Set direction to 0 (y)
                direction = 0
                # Leave stencil as it is
                st_stp_tmp = st_stp
                st_sz_tmp = st_sz
                st_mid_tmp = st_mid
                if visualize:
                    # Extend index of dataframe by padding
                    # First shift index/columns by half the difference between new and old dimensions, so the cross dataframe stays in the middle
                    vof_df.index = vof_df.index+(st_sz_tmp[0]-cr_sz[0])/2
                    vof_df.columns = vof_df.columns+(st_sz_tmp[1]-cr_sz[1])/2
                    # Then reindex the dataframe to create the indices before and after the old indices
                    vof_df = vof_df.reindex(range(st_sz_tmp[0]))
                    vof_df = vof_df.reindex(range(st_sz_tmp[1]), axis='columns').astype(object)
            else:
                # If gradient points more to x_direction
                # Set direction to 1 (x)
                direction = 1
                # Rotate stencil by 90 degrees (flip x and y dimensions)
                st_stp_tmp = np.flip(st_stp)
                st_sz_tmp = np.flip(st_sz)
                st_mid_tmp = np.flip(st_mid)
                if visualize:
                    # Extend columns of dataframe by padding
                    vof_df.index = vof_df.index+(st_sz_tmp[0]-cr_sz[0])/2
                    vof_df.columns = vof_df.columns+(st_sz_tmp[1]-cr_sz[1])/2

                    vof_df = vof_df.reindex(range(st_sz_tmp[0]))
                    vof_df = vof_df.reindex(range(st_sz_tmp[1]), axis='columns').astype(object)
            # Pad vof_array so it fits the stencil dimensions
            pad_y = int((st_sz_tmp[0] - vof_array.shape[0])*1/2)
            pad_x = int((st_sz_tmp[1] - vof_array.shape[1])*1/2)
            vof_array = np.pad(vof_array,
                               [[pad_y, pad_y], [pad_x, pad_x]],
                               mode='constant', constant_values=np.nan)
        # If the stencil is quadratic
        else:
            # Initialize vof_array and vof_df with stencil size
            vof_array = np.empty((st_sz[0], st_sz[1]))
            vof_array[:] = np.nan
            if visualize:
                # Create pandas dataframe to fetch shape of geometry in local coordinates
                vof_df = pd.DataFrame(index=range(st_sz[0]), columns=range(st_sz[1]))
            # Pass on stencil values as they are
            st_stp_tmp = st_stp
            st_sz_tmp = st_sz
            st_mid_tmp = st_mid

        ''' 2. Evaluate VOF values on whole stencil '''
        # Get origins of local coordinates of stencil points
        local_origin_points = np.array([round_point[0]+st_stp_tmp[0],
                                        round_point[1]+st_stp_tmp[1]])
        # Get list of all origins in stencil
        local_origins = np.array(list(itertools.product(
            *[local_origin_points[0][::-1],  # [::-1] to get sorting right
              local_origin_points[1]]
        )))
        # Get list of all stencil indices combinations
        local_indices = np.array(list(itertools.product(range(st_sz_tmp[0]), range(st_sz_tmp[1]))))  # g

        # Iterate over all stencil indices combinations
        for idx, ill in enumerate(local_indices):
            # Skip values that were already calculated for the gradient (= not nan)
            if np.isnan(vof_array[ill[0], ill[1]]):
                # Get origin coordinates too
                lo = local_origins[idx]
                # Values of local grid relative to geometry origin (see above for note on order)
                [y_l, x_l] = np.array([local_grid[0] + lo[0] - x_c[0],
                                       local_grid[1] + lo[1] - x_c[1]])
                if ellipse:
                    # Rotate geometry back to calculate ellipse
                    x_ltmp = x_l.copy()
                    y_ltmp = y_l.copy()
                    x_l = x_ltmp*np.cos(rot)+y_ltmp*np.sin(rot)
                    y_l = -x_ltmp*np.sin(rot)+y_ltmp*np.cos(rot)

                    # Get radii on local grid (np.multiply way faster than np.power) r^2 = e^2*x^2 + y^2
                    r_sqr = e**2*np.multiply(x_l, x_l) + np.multiply(y_l, y_l)
                else:
                    # Get radii on local grid (np.multiply way faster than np.power) r^2 = x^2 + y^2 + z^2
                    r_sqr = np.multiply(x_l, x_l) + np.multiply(y_l, y_l)
                # Calculate 1s and 0s on local grid
                r_area = np.where(r_sqr <= r*r, 1, 0)
                # Get VOF values by integration over local grid
                vof = np.sum(r_area)/r_area.size
                # Write vof value into stencil value array
                vof_array[ill[0], ill[1]] = vof
                if visualize:
                    # Save the r_area array for plotting
                    vof_df.iloc[ill[0], ill[1]] = r_area
        # Apply smearing
        if smearing:
            # Define smearing kernel
            kernel = [[0, 1, 0], [1, 4, 1], [0, 1, 0]]  # FNB
            # kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]  # Gauß
            vof_array_smear = vof_array.copy()
            vof_array_smear[:] = np.nan
            for column in range(1, st_sz_tmp[0]-1):
                for row in range(1, st_sz_tmp[1]-1):
                    # Calculate smeared vof field: sum(weights * vof_array_slice)/sum(weights)
                    vof_array_smear[column, row] = np.sum(np.multiply(
                        kernel,
                        vof_array[column-1:column+2, row-1:row+2]
                    ))/np.sum(kernel)
            # Cut edges of vof_array
            vof_array = vof_array_smear[1:st_sz_tmp[0]-1, 1:st_sz_tmp[1]-1]
            if visualize:
                # Cut vof_df too
                vof_df = vof_df.iloc[1:-1, 1:-1]
            # Shrink st_sz
            st_sz_tmp = np.add(st_sz_tmp, [-2, -2])
            st_mid_tmp = np.add(st_mid_tmp, [-1, -1])

        ''' Plot VoF field and show plot '''
        if visualize:
            # Plot vof
            plot_vof(ax2, vof_df, vof_array, st_sz_tmp, Delta_vof)
            # Show plot
            fig.tight_layout()
            plt.show()

        ''' Invert values by 50% chance '''
        # Only proceed if data is valid (invalid = middle point of stencil does not contain interface)
        # Invalid values are created when the interface is flat and exactly between two cells
        if (vof_array[st_mid_tmp[0], st_mid_tmp[1]] > 0) & (vof_array[st_mid_tmp[0], st_mid_tmp[1]] < 1):
            '''
                e = 100
            if True:
            # '''
            if neg:
                # Invert values by 50% chance
                if u() > 0.5:
                    curvature = -curvature
                    vof_array = 1-vof_array
            # Reshape vof_array from mxn array to m*nx1 vector
            output_array = np.reshape(vof_array, (1, np.prod(st_sz_tmp)))[0].tolist()
            # Insert curvature value at first position
            output_array.insert(0, curvature)
            # DEBUGGING: Insert r, e, pt_x, pt_y
            if debug:
                output_array.insert(0, r)
                output_array.insert(0, e)
                output_array.insert(0, pt_y)
                output_array.insert(0, pt_x)
            # Append list to output list
            output_list.append(output_array)
        else:
            if debug:
                print(f'thrown away')
    if not silent:
        pbar.finish()
    if not visualize:
        if smearing:
            # Shrink st_sz to create right filename
            st_sz = np.add(st_sz, [-2, -2])
        ''' Export data to feather file '''
        # Convert output list to pandas dataframe
        output_df = pd.DataFrame(output_list)
        # Reformat column names as string and rename curvature column
        output_df.columns = output_df.columns.astype(str)
        if debug:
            output_df = output_df.rename(columns={'0':'pt_x', '1': 'pt_y', '2': 'e', '3': 'r', '4': 'curvature'})
        else:
            output_df = output_df.rename(columns={'0':'Curvature'})
        # Write output dataframe to feather file
        parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        file_name = os.path.join(parent_path, 'data', 'datasets', 'data_'+str(st_sz[0])+'x'+str(st_sz[1])+('_eqk' if equal_kappa else '_eqr')+('_neg' if neg else '_pos')+('_ell' if ellipse else '_cir')+('_smr' if smearing else '_nsm')+'_opt.feather')
        print(f'File:\n{file_name}')
        # file_name = os.path.join(parent_path, 'data', 'datasets', 'data_'+str(st_sz[0])+'x'+str(st_sz[1])+('_eqk' if equal_kappa else '_eqr')+('_neg' if neg else '_pos')+('_ell' if ellipse else '_cir')+'_flat_e'+'.feather')
        output_df.reset_index(drop=True).to_feather(file_name)
        # Print string with a summary
        geometry = ('Ellipse' if ellipse else 'Circle')
        print(f'Generated {output_df.shape[0]} tuples in {gt(time0)} with:\nGeometry:\t{geometry}\nGrid:\t\t{int(1/Delta)}x{int(1/Delta)}\nStencil size:\t{st_sz}\nVOF Grid:\t{int(1/Delta_vof)}x{int(1/Delta_vof)}\nVOF Accuracy:\t{np.round(100*Delta_vof**2,3)}%\nNeg. Values:\t{neg}\nSmearing:\t{smearing}')
