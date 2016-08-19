# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:07:40 2016

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

# indices of lower triangle of a 90 x 90 matrix
lotril_ind = np.ravel_multi_index(np.tril_indices(90, k=-1), (90, 90))

# for sparse inverse covariance matrices generated from raw timecourses
timecourse_connectivity_data = np.genfromtxt(timecourse_dir + 'sparse_inverse_covariance_data.csv', delimiter=',')
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

# take matrix logs of timecourse_connectivity_data
timecourse_log_data = np.squeeze(np.array(map(lambda x: np.reshape(la.logm(np.reshape(x, (90, 90))), (1, 8100)), timecourse_connectivity_data)))

# pull out lower triangle
timecourse_log_data_lotril = timecourse_log_data[:, lotril_ind]
timecourse_connectivity_data_lotril = timecourse_connectivity_data[:, lotril_ind]

# make kernels
K_lin_timecourse = np.dot(timecourse_connectivity_data_lotril, np.transpose(timecourse_connectivity_data_lotril))
K_lin_timecourse_log = np.dot(timecourse_log_data_lotril, np.transpose(timecourse_log_data_lotril))

# pull in old kernel for comparison
K_lin_connectivity = np.genfromtxt(kernel_dir + 'K_edge.csv', delimiter = ',')[:, 1:]
K_lin_connectivity = K_lin_connectivity[:, connectivity_in_timecourse][connectivity_in_timecourse, :]

kf = cross_validation.StratifiedKFold(labels, 10, shuffle=True)
sss = cross_validation.StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.1)
clf = svm.SVC(kernel='precomputed')
#preds = np.zeros((np.shape(timecourse_connectivity_data)[0],))
preds_rep = np.zeros((0,))
labels_rep = np.zeros((0,))

# find geometric mean
# construct base covariance matrix by repeated averaging in tangent space
timecourse_connectivity_matrices = np.reshape(timecourse_connectivity_data, (len(timecourse_connectivity_data), 90, 90))
#base_cov_matrix = np.mean(timecourse_connectivity_matrices, axis=0)
#for i in range(20) :
#    
#    print i
#            
#    # project all matrices into the tangent space
#    tangent_matrices = np.zeros_like(timecourse_connectivity_matrices)
#    for j in range(len(timecourse_connectivity_matrices)) :
#            
#        tangent_matrices[j, :, :] = np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, 0.5), la.logm(np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, -0.5), timecourse_connectivity_matrices[j, :, :],la.fractional_matrix_power(base_cov_matrix, -0.5)])), la.fractional_matrix_power(base_cov_matrix, 0.5)])
#        
#    # calculate the tangent space mean
#    tangent_space_base_cov_matrix = np.mean(tangent_matrices, axis=0)
#    norm = np.linalg.norm(tangent_space_base_cov_matrix, ord='fro')
#
#        
#    # project new tangent mean back to the manifold
#    base_cov_matrix = np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, 0.5), la.expm(np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, -0.5), tangent_space_base_cov_matrix ,la.fractional_matrix_power(base_cov_matrix, -0.5)])), la.fractional_matrix_power(base_cov_matrix, 0.5)])
#
#
##for train_index, test_index in kf :
#for train_index, test_index in sss :
#    
#    # transform the data
#    # construct base covariance matrix by averaging training data
##    base_cov_data = np.mean(timecourse_connectivity_data[train_index, :], axis = 0)
##    base_cov_matrix = np.reshape(base_cov_data, (90, 90))
#    
#
#            
#            
#            
#    # apply whitening transport and projection for training AND testing data
#    transformed_connectivity_data = np.zeros_like(timecourse_connectivity_data)
#    for i in range(len(timecourse_connectivity_data)) :
#        
#        connectivity_matrix = np.reshape(timecourse_connectivity_data[i, :], (90, 90))
#        base_cov_matrix_pow = la.fractional_matrix_power(base_cov_matrix, -0.5)
#        transformed_connectivity_data[i, :] = np.reshape(la.logm(np.linalg.multi_dot([base_cov_matrix_pow, connectivity_matrix, base_cov_matrix_pow])), (1, 8100))
#
#    # pull out lower triangle
#    transformed_connectivity_data = transformed_connectivity_data[:, lotril_ind]
#    
#    # split data and labels into train and test
#    training_data = transformed_connectivity_data[train_index, :]
#    testing_data = transformed_connectivity_data[test_index, :]
#    
#    training_connectivity_data = timecourse_connectivity_data[train_index, :]
#    testing_connectivity_data = timecourse_connectivity_data[test_index, :]
#    
#    # apply whitening transport and projection for training data
#    base_cov_data_train = np.mean(training_connectivity_data, axis = 0)
#    base_cov_matrix_train = np.reshape(base_cov_data_train, (90, 90))
#    transformed_connectivity_data_train = np.zeros_like(training_connectivity_data)
#    for i in range(len(training_connectivity_data)) :
#        
#        connectivity_matrix = np.reshape(training_connectivity_data[i, :], (90, 90))
#        base_cov_matrix_pow = la.fractional_matrix_power(base_cov_matrix_train, -0.5)
#        transformed_connectivity_data_train[i, :] = np.reshape(la.logm(np.linalg.multi_dot([base_cov_matrix_pow, connectivity_matrix, base_cov_matrix_pow])), (1, 8100))    
#   
#    # apply whitening transport and projection for testing data
#    base_cov_data_test = np.mean(testing_connectivity_data, axis = 0)
#    base_cov_matrix_test = np.reshape(base_cov_data_test, (90, 90))
#    transformed_connectivity_data_test = np.zeros_like(testing_connectivity_data)
#    for i in range(len(testing_connectivity_data)) :
#        
#        connectivity_matrix = np.reshape(testing_connectivity_data[i, :], (90, 90))
#        base_cov_matrix_pow = la.fractional_matrix_power(base_cov_matrix_test, -0.5)
#        transformed_connectivity_data_test[i, :] = np.reshape(la.logm(np.linalg.multi_dot([base_cov_matrix_pow, connectivity_matrix, base_cov_matrix_pow])), (1, 8100))
#    
#    # pull out lower triangle
#    training_data = transformed_connectivity_data_train[:, lotril_ind]
#    testing_data = transformed_connectivity_data_test[:, lotril_ind]     
#    
#    training_labels = labels[train_index]
#    testing_labels = labels[test_index]
##    
##    # make kernels
#    K_train = np.dot(training_data, np.transpose(training_data))
#    K_test = np.dot(testing_data, np.transpose(training_data))
##    K_train = K_lin_timecourse[train_index, :][:, train_index]
##    K_test = K_lin_timecourse[test_index, :][:, train_index]
#    
#    
#    # train classifier and predict
#    clf.fit(K_train, training_labels)
#    preds_rep = np.append(preds_rep, clf.predict(K_test))
#    labels_rep = np.append(labels_rep, testing_labels)
    
#print np.transpose(np.vstack((np.transpose(labels), np.transpose(preds_rep))))
print metrics.accuracy_score(labels_rep, preds_rep)

# try pyriemann's tangent space classifier
# adapted from http://pythonhosted.org/pyriemann/auto_examples/motor-imagery/plot_single.html#sphx-glr-auto-examples-motor-imagery-plot-single-py
# reshape data for TSclassifier
timecourse_connectivity_data_TSclassifier = np.reshape(timecourse_connectivity_data, (100, 90, 90))
clf = TSclassifier()
clf = MDM()
#cv = cross_validation.KFold(len(labels), 10, shuffle=True, random_state=42)
cv = cross_validation.StratifiedShuffleSplit(labels, n_iter=100, test_size=0.1)
scores = cross_validation.cross_val_score(clf, timecourse_connectivity_data_TSclassifier, labels, cv=cv, n_jobs=1)
print scores
print np.mean(scores)

    
    
