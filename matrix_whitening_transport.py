# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 16:19:36 2016

@author: jonyoung
"""

data_dir_1 = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/'
data_dir_2 = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC2/matrix_unsmooth/'
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/kernels/'

# include negatively weighted edges or not
include_negative_weights = True

# standardise data with a z-transform
standardise_data = False

# read in connectivity data and labels
connectivity_data_1 = utils.load_connectivity_data(data_dir_1, standardise_data)
labels_1 = np.array([utils.load_labels(data_dir_1), ])
connectivity_data_2 = utils.load_connectivity_data(data_dir_2, standardise_data)
labels_2 = np.array([utils.load_labels(data_dir_2), ])


connectivity_data = np.vstack((connectivity_data_1, connectivity_data_2))
labels = np.hstack((labels_1, labels_2))

# save connectivity data
np.savetxt(kernel_dir + 'connectivity_data.csv', connectivity_data, delimiter = ',')

# map lower triangles of connectivities to an array
edge_data = np.apply_along_axis(lambda x: x[np.ravel_multi_index(np.tril_indices(90, k=-1), (90, 90))], 1, connectivity_data)

if not include_negative_weights :

    # set negative connectivities to 0
    edge_data = np.apply_along_axis(lambda x: [0 if element < 0 else element for element in x], 1, edge_data)

# caclulate the kernel values
K = np.dot(edge_data, np.transpose(edge_data))

# attach the labels and save
K = np.hstack((np.transpose(labels), K))

if include_negative_weights :
    
    if standardise_data :

        np.savetxt(kernel_dir + 'K_edge_standardised.csv', K, delimiter=',')
        
    else :

        np.savetxt(kernel_dir + 'K_edge.csv', K, delimiter=',')

else :
    
    if standardise_data :

        np.savetxt(kernel_dir + 'K_edge_standardised_positive.csv', K, delimiter=',')
        
    else :

        np.savetxt(kernel_dir + 'K_edge_positive.csv', K, delimiter=',')