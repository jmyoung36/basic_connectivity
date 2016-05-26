# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:21:37 2016

@author: jonyoung
"""

def rw_overlap_kernel(C1, C2):
    
    #l = 1.0/np.exp(1.0)
    l = 0.5    
    
    
    k = 0
    c = 0
    
    # reshape rows into kernel matrices
    M1 = np.reshape(C1, (90, 90))
    M2 = np.reshape(C2, (90, 90))
    
    # normalise so rows sum to 1
    M1_norm = normalize(M1, axis=1, norm='l1')
    M2_norm = normalize(M2, axis=1, norm='l1')
    
    for i in range(1, 101) :
        
        M1_exp = np.linalg.matrix_power(M1_norm, i)
        M2_exp = np.linalg.matrix_power(M2_norm, i)
        
        #overlap = np.sum(np.minimum(M1_exp, M2_exp))
        overlap = np.sum(np.sqrt(np.multiply(M1_exp, M2_exp)))
    
        #k = k + ((np.exp(-i) ) * overlap)
        #c = c + ((np.exp(-i)) * 90)
        k = k + ((l ** i) * overlap)
        c = c + ((l ** i) * 90)
    
    return  k/c
       
# import what we need
import numpy as np
import connectivity_utils as utils
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import normalize

# set directories
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/'
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/kernels/'

# read in connectivity data and labels
connectivity_data = utils.load_connectivity_data(data_dir)
labels = np.array([utils.load_labels(data_dir), ])

# set negative connectivities to 0
edge_data = np.apply_along_axis(lambda x: [0 if element < 0 else element for element in x], 1, connectivity_data)
#edge_data = edge_data[:6, :]

# calculate the kernel using pdist
K = squareform(pdist(edge_data, rw_overlap_kernel))
K[np.diag_indices(140)] = 1
K = np.hstack((np.transpose(labels), K))

np.savetxt(kernel_dir + 'K_rw_overlap.csv', K, delimiter=',')

#C1 = edge_data[0, :]
#C2 = edge_data[1, :]
#
#M1 = np.reshape(C1, (90, 90))
#M2 = np.reshape(C2, (90, 90))
#    
## normalise so rows sum to 1
#M1_norm = normalize(M1, axis=1, norm='l1')
#M2_norm = normalize(M2, axis=1, norm='l1')
#
#k = 0
#c = 0
#
#for i in range(1, 101) :
#        
#    M1_exp = np.linalg.matrix_power(M1_norm, i)
#    M2_exp = np.linalg.matrix_power(M2_norm, i)
#        
#    overlap = np.sum(np.minimum(M1_exp, M2_exp))
#    
#    k = k + (np.exp(-i) * overlap)
#    c = c + (np.exp(-i) * 90)
#    
#    print k
#    print c
#    print overlap
