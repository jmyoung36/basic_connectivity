# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:56:45 2016

@author: jonyoung
"""

# import what we need
import numpy as np
import connectivity_utils as utils
from scipy.linalg import logm

# set directories
data_dir_1 = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/'
data_dir_2 = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC2/matrix_unsmooth/'
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/kernels/'

# include negatively weighted edges or not
include_negative_weights = True

# standardise data with a z-transform
standardise_data = True

# read in connectivity data and labels
connectivity_data_1, connectivity_files = utils.load_connectivity_data(data_dir_1, standardise_data)
labels_1 = np.array([utils.load_labels(data_dir_1), ])
connectivity_data_2 = utils.load_connectivity_data(data_dir_2, standardise_data)
labels_2 = np.array([utils.load_labels(data_dir_2), ])

#connectivity_data = np.vstack((connectivity_data_1, connectivity_data_2))
connectivity_data = connectivity_data_1
labels = np.hstack((labels_1, labels_2))

edge_data = connectivity_data

if not include_negative_weights :

    # set negative connectivities to 0
    edge_data = np.apply_along_axis(lambda x: [0 if element < 0 else element for element in x], 1, edge_data)
    

# allocate memory for transformed edge data
LE_transformed_edge_data = np.zeros(np.shape(edge_data))

# transform the data
for i in range(len(edge_data)) :
    
    subject_edge_data = edge_data[i, :]
    edge_data_matrix = np.matrix(np.reshape(subject_edge_data, (90, 90)))
    #edge_data_matrix[np.diag_indices(90)] = 1
    
    # symmetrise matrix
    #edge_data_matrix = (edge_data_matrix + np.transpose(edge_data_matrix)) / 2
    #print np.shape(edge_data_matrix)
    print edge_data_matrix[:5, :5]
    #print edge_data_matrix[:5,:5] - np.transpose(edge_data_matrix)[:5, :5]
    LE_transformed_edge_data_matrix = logm(edge_data_matrix[:5, :5])
    print LE_transformed_edge_data_matrix

print connectivity_files

# save connectivity data
#np.savetxt(kernel_dir + 'connectivity_data_Arsigny_transformed.csv', connectivity_data, delimiter = ',')

# caclulate the kernel values
K = np.dot(edge_data, np.transpose(edge_data))

# attach the labels and save
K = np.hstack((np.transpose(labels), K))

#if include_negative_weights :
#    
#    if standardise_data :
#
#        #np.savetxt(kernel_dir + 'K_edge_standardised_Arsigny.csv', K, delimiter=',')
#        
#    else :
#
#        #np.savetxt(kernel_dir + 'K_edge_Arsigny.csv', K, delimiter=',')
#
#else :
#    
#    if standardise_data :
#
#        #np.savetxt(kernel_dir + 'K_edge_standardised_positive_Arsigny.csv', K, delimiter=',')
#        
#    else :
#
#        #np.savetxt(kernel_dir + 'K_edge_positive_Arsigny.csv', K, delimiter=',')

    
