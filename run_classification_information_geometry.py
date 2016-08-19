# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 16:41:50 2016

@author: jonyoung
"""

# import what we need
import numpy as np
import connectivity_utils as utils
import scipy.linalg as la
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import glob
from sklearn import svm, cross_validation, metrics
from pyriemann.classification import TSclassifier, MDM

# set directories
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/'
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/kernels/'
timecourse_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC_Unsmooth_TimeCourse/'

# SQUARED DISTANCE between two covariance matrices according to eq'n 2 in Barachant,
# Alexandre, and Marco Congedo. "A Plug & Play P300 BCI Using Information 
# Geometry." arXiv preprint arXiv:1409.0107 (2014).
def sq_IG_distance(cov_1, cov_2) :
    
    cov_1_pow = la.fractional_matrix_power(cov_1, -0.5)
    return la.norm(la.logm(np.linalg.multi_dot([cov_1_pow, cov_2, cov_1_pow])), ord='fro') ** 2
        
# timecourse data connectivity matrices
timecourse_connectivity_data = np.genfromtxt(timecourse_dir + 'sparse_inverse_covariance_data.csv', delimiter=',')    
connectivity_data = np.genfromtxt(timecourse_dir + 'sparse_inverse_covariance_data.csv', delimiter=',')
timecourse_files = pd.read_csv(timecourse_dir + 'sparse_inverse_covariance_files.csv').T.index.values
timecourse_IDs = map(lambda x: int(x.split('/')[-1].split('_')[1][0:-4]), timecourse_files) 
labels = np.array([utils.load_labels(data_dir), ])[0]
connectivity_files = glob.glob(data_dir + '*.txt')
connectivity_IDs = map(lambda x: int(x.split('/')[-1][0:3]), connectivity_files)
connectivity_IDs.sort()
connectivity_in_timecourse = np.array([True if ID in timecourse_IDs else False for ID in connectivity_IDs])
timecourse_in_connectivity = np.array([True if ID in connectivity_IDs else False for ID in timecourse_IDs])
labels = labels[np.array(connectivity_in_timecourse)]
#labels = np.expand_dims(labels, axis=1)
timecourse_connectivity_data = timecourse_connectivity_data[timecourse_in_connectivity, :]
timecourse_connectivity_matrices = np.reshape(timecourse_connectivity_data, (100, 90, 90))

# original connectivity matrices
connectivity_data, connectivity_files = utils.load_connectivity_data('/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/')
connectivity_data = connectivity_data[connectivity_in_timecourse, :]
connectivity_matrices = np.reshape(connectivity_data, (100, 90, 90))

n_subjects = len(labels)


print connectivity_matrices[0, :5 , :5]
print timecourse_connectivity_matrices[0, :5 , :5]

# create kernel matrices
#sq_IG = np.zeros((n_subjects, n_subjects))
#sq_IG_timecourse = np.zeros((n_subjects, n_subjects))
#for i in range(n_subjects) :
#    print i
#    for j in range(i) :
#        
#        
#        sq_IG_dist = sq_IG_distance(connectivity_matrices[i, :, :], connectivity_matrices[j, :, :])
#        sq_IG[i, j] = sq_IG_dist
#        sq_IG[j, i] = sq_IG_dist
#        sq_IG_dist = sq_IG_distance(timecourse_connectivity_matrices[i, :, :], timecourse_connectivity_matrices[j, :, :])
#        sq_IG_timecourse[i, j] = sq_IG_dist
#        sq_IG_timecourse[j, i] = sq_IG_dist
#
#clf = svm.SVC(kernel='precomputed')
#for i in range (-10, 30):
#    
#    sigma = 2.0 ** (i - 10)
#    K_sigma = np.exp(-sq_IG / (sigma ** 2))
#    cv = cross_validation.StratifiedShuffleSplit(labels, n_iter=100, test_size=0.1)
#    scores = cross_validation.cross_val_score(clf, K_sigma, labels, cv=cv)
#    print scores
#    print np.mean(scores)

# save for transfer to Matlab
labels = np.expand_dims(labels, 1)
print np.shape(labels)
M_connectivity_data = np.hstack((labels, connectivity_data))
M_timecourse_connectivity_data = np.hstack((labels, timecourse_connectivity_data))

np.savetxt('/home/jonyoung/IoP_data/Data/connectivity_data/M_connectivity_data.csv', M_connectivity_data, delimiter=',')
np.savetxt('/home/jonyoung/IoP_data/Data/connectivity_data/M_timecourse_connectivity_data.csv', M_timecourse_connectivity_data, delimiter=',')