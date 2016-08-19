# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:07:12 2016

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

# define Matusita measure between two covariance matrices, as given in eq 1 of
# Estimation of similarity measure for multivariate normal distributions, 
# Minami, M. & Shimizu, K. Environmental and Ecological Statistics (1999).
# assume the two covariance matrices represent multivariate Gaussians with
# equal means, hence ignore the R part of the equation and calculate only the
# Q part which represents the information contributed by the difference
# between the covariance matrices
def Matusita_kernel(cov_1, cov_2):
    
    p = np.shape(cov_1)[0]    
    
    det_1 = la.det(cov_1)
    det_2 = la.det(cov_2)
    det_sum = la.det(cov_1 + cov_2)
    return ((2 ** (p/2.0)) * (det_1 ** 0.25) * (det_2 ** 0.25))/(det_sum ** 0.5)
    
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
#connectivity_data, connectivity_files = utils.load_connectivity_data('/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/')
#connectivity_data = connectivity_data[connectivity_in_timecourse, :]
#connectivity_matrices = np.reshape(connectivity_data, (100, 90, 90))

n_subjects = len(labels)

# create kernel matrices
#K_Matusita = np.zeros((n_subjects, n_subjects))
K_Matusita_timecourse = np.zeros((n_subjects, n_subjects))
for i in range(n_subjects) :
    for j in range(i + 1) :
        
#        k = Matusita_kernel(connectivity_matrices[i, :, :], connectivity_matrices[j, :, :])
#        K_Matusita[i, j] = k
#        K_Matusita[j, i] = k
        k_timecourse = Matusita_kernel(timecourse_connectivity_matrices[i, :, :], timecourse_connectivity_matrices[j, :, :])
        K_Matusita_timecourse[i, j] = k_timecourse
        K_Matusita_timecourse[j, i] = k_timecourse
print np.shape(labels)
print np.shape(K_Matusita_timecourse)
clf = svm.SVC(kernel='precomputed')
cv = cross_validation.StratifiedShuffleSplit(labels, n_iter=100, test_size=0.1)
scores = cross_validation.cross_val_score(clf, K_Matusita_timecourse, labels, cv=cv)
print scores
print np.mean(scores)
