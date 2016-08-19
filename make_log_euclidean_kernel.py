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
    
    # remove negative connectivities
    C1[C1 < 0] = 0;
    C2[C2 < 0] = 0;        
    M1 = np.reshape(C1, (90, 90))
    M2 = np.reshape(C2, (90, 90))
    D1 = np.diag(np.sum(M1, axis=1))
    D2 = np.diag(np.sum(M2, axis=1))
    L1 = D1 - M1
    L2 = D2 - M2
    S1 = L1 + (gamma * np.eye(90))
    S2 = L2 + (gamma * np.eye(90))
    
    return  np.exp((-1 * (la.norm((la.logm(S1) - la.logm(S2)), ord='fro')) ** 2) / (sigma**2))
       
# import what we need
import numpy as np
import connectivity_utils as utils
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import glob

# set directories
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/'
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/kernels/'
timecourse_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC_Unsmooth_TimeCourse/'

# read in connectivity data and labels - for original connectivity matrices
#connectivity_data = utils.load_connectivity_data(data_dir)
#labels = np.array([utils.load_labels(data_dir), ])

# for sparse inverse covariance matrices generated from raw timecourses
connectivity_data = np.genfromtxt(timecourse_dir + 'sparse_inverse_covariance_data.csv', delimiter=',')

timecourse_files = pd.read_csv(timecourse_dir + 'sparse_inverse_covariance_files.csv').T.index.values
timecourse_IDs = map(lambda x: int(x.split('/')[-1].split('_')[1][0:-4]), timecourse_files) 
edge_data = connectivity_data
labels = np.array([utils.load_labels(data_dir), ])[0]
connectivity_files = glob.glob(data_dir + '*.txt')
connectivity_IDs = map(lambda x: int(x.split('/')[-1][0:3]), connectivity_files)
connectivity_IDs.sort()
connectivity_in_timecourse = np.array([True if ID in timecourse_IDs else False for ID in connectivity_IDs])
timecourse_in_connectivity = np.array([True if ID in connectivity_IDs else False for ID in timecourse_IDs])
labels = labels[np.array(connectivity_in_timecourse)]
labels = np.expand_dims(labels, axis=1)
edge_data = edge_data[timecourse_in_connectivity, :]

# sanity check
# pull out lower triangles
#from sklearn import svm, cross_validation
#lotril_ind = np.ravel_multi_index(np.tril_indices(90, k=-1), (90, 90))
#edge_data = edge_data[:, lotril_ind]
#K = np.dot(edge_data, np.transpose(edge_data))
#clf = svm.SVC(kernel='precomputed')
#foo =  cross_validation.cross_val_score(clf, K, labels, cv=10)
#print foo
#print np.mean(foo)
#print np.sum(labels)


# set negative connectivities to 0
#edge_data = np.apply_along_axis(lambda x: [0 if element < 0 else element for element in x], 1, connectivity_data)

#for i in range(-10, 10) :
for i in range(-10, 10) :
    
    #for j in range(-10, 10) :
    for j in range(-10, 10) :

        # set parameters: regularisation strength gamma and kernel bandwidth sigma
        gamma = 2**i
        sigma = 2**j

        # calculate the kernel using pdist
        #K = squareform(pdist(edge_data, lambda C1, C2: log_euclidean_kernel(C1, C2, gamma, sigma)))
        #K[np.diag_indices(140)] = 1

        # ugly way
        K = np.zeros((100, 100))
        for k in range(0, 100) :
            for l in range(0, k+1) :
                val = log_euclidean_kernel(edge_data[k, :], edge_data[l,:], gamma, sigma)
                K[k, l] = val
                K[l, k] = val
                if k % 10 == 0 and l == 0:
                    
                    print 'k = ' + str(k)
                
        
        print 'gamma = ' + str(gamma)     
        print 'sigma = ' + str(sigma)
        
        # attach the labels and save


        print np.shape(K)
        print np.shape(labels)
        K = np.hstack((labels, K))
        np.savetxt(kernel_dir + 'K_log_euclidean_sparse_inverse_cov_pos_only_' + str(i) + '_' + str(j) + '.csv', K, delimiter=',')
    