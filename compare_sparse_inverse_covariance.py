# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:12:12 2016

@author: jonyoung
"""

# import the packages we need
import numpy as np
import glob
import csv
import re
import connectivity_utils as utils
from sklearn import svm, cross_validation, metrics

# define natural sort key so we sort files into correct (natural) order
# taken from http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort?lq=1
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
        for text in re.split(_nsre, s)]

# set directories
sparse_inverse_cov_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC_Unsmooth_TimeCourse/'
correlation_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/'

# read in and process the list of files for which we have correlation data
correlation_files = glob.glob(correlation_dir + 'matrix_unsmooth/*.txt')
n_correlation_files = len(correlation_files)
    
# sort into numerical order
correlation_files.sort(key=natural_sort_key)

# convert to list of subject numbers
correlation_subject_ids = map(lambda string: int(string.split('/')[-1][0:3]), correlation_files)
    
# sort into numerical order
correlation_files.sort(key=natural_sort_key)

# convert to list of subject numbers
correlation_subject_ids = map(lambda string: int(string.split('/')[-1][0:3]), correlation_files)

# read in and process the list of files for which we have sparse covariance data
with open(sparse_inverse_cov_dir + 'sparse_inverse_covariance_files.csv', 'rb') as f:
    reader = csv.reader(f)
    covariance_files = list(reader)
    
# convert to list of subject numbers
covariance_subject_ids = map(lambda string: int(string.split('_')[-1][0:-4]), covariance_files[0])

# find indices of common elements in BOTH lists
common_correlation_indices = [i for i, item in enumerate(correlation_subject_ids) if item in set(covariance_subject_ids)]
common_covariance_indices = [i for i, item in enumerate(covariance_subject_ids) if item in set(correlation_subject_ids)]

# read in the labels 
labels = np.array([utils.load_labels(correlation_dir + 'matrix_unsmooth/' ), ])

# take only labels for subjects common between covariance and correlation sets.
# initially labels are for the same subjects as correlation data so use common_correlation_indices
labels = labels[0, common_correlation_indices]
print len(labels)

# read in the sparse inverse covariance data and cut it down to only include common subjects
sparse_inverse_cov_data = np.genfromtxt(sparse_inverse_cov_dir + 'OAS_data.csv', delimiter=',')[common_covariance_indices,:]

# map lower triangles of connectivities to an array
sparse_cov_edge_data = np.apply_along_axis(lambda x: x[np.ravel_multi_index(np.tril_indices(90, k=-1), (90, 90))], 1, sparse_inverse_cov_data)

# optional - remove negative correlations
#sparse_cov_edge_data = np.apply_along_axis(lambda x: [0 if element < 0 else element for element in x], 1, sparse_cov_edge_data)


# calcalculate a linear kernel for the spare inverse covariance data
K_sparse_cov_lin = np.dot(sparse_cov_edge_data, np.transpose(sparse_cov_edge_data))

# read in the linear kernel for the correlation, remove labels from first column and cut it down to only include the common subjects
K_corr_lin  = np.genfromtxt(correlation_dir + 'kernels/K_edge.csv', delimiter=',')
K_corr_lin = K_corr_lin[:, 1:]
K_corr_lin = K_corr_lin[common_correlation_indices,:][:, common_correlation_indices]

# initialise the classifier
clf = svm.SVC(kernel='precomputed')

preds_sparse_inverse_cov = np.zeros((len(labels),))
preds_corr = np.zeros((len(labels),))

i = 0

# shuffle labels and kernels together
perm = np.random.permutation(len(labels))
labels = labels[perm]
K_corr_lin = K_corr_lin[perm, :][:, perm]
K_sparse_cov_lin = K_sparse_cov_lin[perm, :][:, perm]


# 20-fold stratified cross validation
folds = cross_validation.StratifiedKFold(labels, n_folds=20)
for train_index, test_index in folds :
    
    print i
    labels_train = labels[train_index]
    labels_test = labels[test_index]
    K_corr_lin_train = K_corr_lin[train_index, :][:, train_index]
    K_corr_lin_test = K_corr_lin[test_index, :][:, train_index]
    K_sparse_cov_lin_train = K_sparse_cov_lin[train_index, :][:, train_index]
    K_sparse_cov_lin_test = K_sparse_cov_lin[test_index, :][:, train_index]   
    clf.fit(K_corr_lin_train, labels_train)
    preds_corr[test_index] = clf.predict(K_corr_lin_test)
    clf.fit(K_sparse_cov_lin_train, labels_train)
    preds_sparse_inverse_cov[test_index] = clf.predict(K_sparse_cov_lin_test)
    i = i+1
    
print metrics.roc_auc_score(labels, preds_sparse_inverse_cov)
print metrics.accuracy_score(labels, preds_sparse_inverse_cov)
print metrics.roc_auc_score(labels, preds_corr)
print metrics.accuracy_score(labels, preds_corr)






