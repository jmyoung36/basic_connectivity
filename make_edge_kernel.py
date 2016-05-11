# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:39:51 2016

@author: jonyoung
"""

# import what we need
import numpy as np
import connectivity_utils as utils

# set directories
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/'
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/kernels/'

# read in connectivity data and labels
connectivity_data = utils.load_connectivity_data(data_dir)
labels = np.array([utils.load_labels(data_dir), ])

# map lower triangles of connectivities to an array
edge_data = np.apply_along_axis(lambda x: x[np.ravel_multi_index(np.tril_indices(90, k=-1), (90, 90))], 1, connectivity_data)

# caclulate the kernel values
K = np.dot(edge_data, np.transpose(edge_data))

# attach the labels and save
K = np.hstack((np.transpose(labels), K))
np.savetxt(kernel_dir + 'K_edge.csv', K, delimiter=',')



    
