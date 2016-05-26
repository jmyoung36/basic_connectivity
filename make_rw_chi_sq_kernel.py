# -*- coding: utf-8 -*-
"""
Created on Wed May 25 09:34:03 2016

@author: jonyoung
"""

def update_matrix_row(matrix_row, power) :
    
    # reshape row into connectivity matrix
    M = np.reshape(matrix_row, (90, 90))
#    print power
#    print 'Matrix sample:'
#    print M[:5, :5]
#    print np.sum(M[:5, :], axis=1)
    
    # raise the matrix to the specified power
    M_exp = np.linalg.matrix_power(M, power)
#    print 'Exponentiated matrix sample:'
#    print M_exp[:5, :5]
#    print np.sum(M_exp[:5, :], axis=1)

    
    # squash the exponentiated matrix back to a vector and return it    
    return  np.reshape(M_exp, (8100,))
    
def normalise_matrix_row(matrix_row) :
    
    # reshape row into connectivity matrix
    M = np.reshape(matrix_row, (90, 90))
    
    # raise the matrix to the ith power
    M_norm = normalize(M, axis=1, norm='l1')
    
    # squash the normalised matrix back to a vector and return it    
    return  np.reshape(M_norm, (8100,))

       
# import what we need
import numpy as np
import connectivity_utils as utils
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import chi2_kernel, additive_chi2_kernel

# set directories
data_dir_1 = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/'
data_dir_2 = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC2/matrix_unsmooth/'
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/kernels/'

# number of subjects
n = 333

# read in connectivity data and labels
connectivity_data_1 = utils.load_connectivity_data(data_dir_1)
labels_1 = np.array([utils.load_labels(data_dir_1), ])
connectivity_data_2 = utils.load_connectivity_data(data_dir_2)
labels_2 = np.array([utils.load_labels(data_dir_2), ])

connectivity_data = np.vstack((connectivity_data_1, connectivity_data_2))
labels = np.hstack((labels_1, labels_2))

# set negative connectivities to 0
edge_data = np.apply_along_axis(lambda x: [0 if element < 0 else element for element in x], 1, connectivity_data)

# normalise the edge data
edge_data_norm = np.apply_along_axis(lambda x: normalise_matrix_row(x), 1, edge_data)

# roll through grid of parameters :
for i in range(-10, 10) :
    
    for j in range(1,20) :
        
        # calculate kernel parameter values
        gamma = 2 ** i
        decay_coefficient = 0.05 * j
        print 'gamma = ' + str(gamma)
        print 'decay coefficient = ' + str(decay_coefficient)
        
        # initialise kernel matrix
        K = np.zeros((n, n))
        
        # 100 iterations
        # calculate chi-sq distance between martrices raised to the kth power
        # weight with decay and accumulate for a weighted sum of chi-sq distances
        for k in range(1, 101) :
            
            decay = decay_coefficient ** k
            edge_data_exp = np.apply_along_axis(lambda x: update_matrix_row(x, k), 1, edge_data_norm)
            #K_update = decay * chi2_kernel(edge_data_exp, gamma=gamma)
           # print K_update[:5, :5]
            K_update = decay * additive_chi2_kernel(edge_data_exp)
            #print K_update[:5, :5]
            #print decay
            #print K_update[:5, :5]
            K = K + K_update
        
        # convert to Gaussian kernel
        print K[:5, :5]
        K = np.exp(gamma * K)
        print K[:5, :5]
         
        # attach the labels and save
        K = np.hstack((np.transpose(labels), K)) 
        np.savetxt(kernel_dir + 'K_rw_chisq_ ' + str(i) + '_' + str(j) + '.csv', K, delimiter=',')
            
        














# calculate the kernel using pdist
#K = squareform(pdist(edge_data, rw_overlap_kernel))
#K[np.diag_indices(140)] = 1
#K = np.hstack((np.transpose(labels), K))
#
#np.savetxt(kernel_dir + 'K_rw_overlap.csv', K, delimiter=',')