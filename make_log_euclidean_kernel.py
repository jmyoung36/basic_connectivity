# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:50:18 2016

@author: jonyoung
"""

# calculate the log-eudclidean kernel of a set of connectivity matrices
# based on:  Dodero, L., Ha Quang, M., San Biagio, M., Murino, V., Sona, D.:  Kernel-based
# classification for brain connectivity graphs on the riemannian manifold of positive
# definite matrices, International Symposium of Biomedical Imaging ISBI 2015.

# define function giving log-euclidean kernel between a pair of SPD connectvity matrices
def log_euclidean_kernel(C1, C2, gamma, sigma):
    
    M1 = np.reshape(C1, (90, 90))
    M2 = np.reshape(C2, (90, 90))
    D1 = np.diag(np.sum(M1, axis=1))
    D2 = np.diag(np.sum(M2, axis=1))
    L1 = D1 - M1
    L2 = D2 - M2
    S1 = L1 + (gamma * np.eye(90))
    S2 = L2 + (gamma * np.eye(90))
    
    return  np.exp((-1 * la.norm((la.logm(S1) - la.logm(S2)), ord='fro')) / (sigma^2))
       
# import what we need
import numpy as np
import connectivity_utils as utils
import scipy.linalg as la
from scipy.spatial.distance import pdist

# set directories
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/matrix_unsmooth/'
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/kernels/'

# set parameters: regularisation strength gamma and kernel bandwidth sigma
gamma = 2^-2
sigma = 2^-2

# read in connectivity data and labels
connectivity_data = utils.load_connectivity_data(data_dir)
labels = np.array([utils.load_labels(data_dir), ])

# set negative connectivities to 0
edge_data = np.apply_along_axis(lambda x: x[x < 0] == 0, 1, connectivity_data)

# calculate the kernel using pdist
K = pdist(connectivity_data, lambda C1, C2:log_euclidean_kernel(C1, C2, gamma, sigma))

# attach the labels and save
K = np.hstack((np.transpose(labels), K))
np.savetxt(kernel_dir + 'K_log_euclidean_' + str(gamma) + '_' + str(sigma) + '.csv', K, delimiter=',')