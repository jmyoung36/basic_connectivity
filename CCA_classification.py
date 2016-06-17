# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 15:08:29 2016

@author: jonyoung
"""

import numpy as np
import connectivity_utils as utils
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn import svm, cross_validation, metrics, decomposition

# set directories
data_dir_1 = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/'
data_dir_2 = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC2/matrix_unsmooth/'
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/kernels/'

# include negatively weighted edges or not
include_negative_weights = True

# standardise data with a z-transform
standardise_data = False

# read in connectivity data and labels
connectivity_data_1 = utils.load_connectivity_data(data_dir_1, standardise_data)
labels_1 = np.array([utils.load_labels(data_dir_1), ])
connectivity_data_2 = utils.load_connectivity_data(data_dir_2, standardise_data)
labels_2 = np.array([utils.load_labels(data_dir_2), ])
labels = np.squeeze(np.hstack((labels_1, labels_2)))

n_subjects = len(labels)
n_folds = 10

connectivity_data = np.vstack((connectivity_data_1, connectivity_data_2))

# map lower triangles of connectivities to an array
edge_data = np.apply_along_axis(lambda x: x[np.ravel_multi_index(np.tril_indices(90, k=-1), (90, 90))], 1, connectivity_data)

if not include_negative_weights :

    # set negative connectivities to 0
    edge_data = np.apply_along_axis(lambda x: [0 if element < 0 else element for element in x], 1, edge_data)
    
# re-split data (3 ways) for CCA
X1_train = edge_data[:140, :]
X2_train = edge_data[140:280, :]
X2_remain = edge_data[280:, :]
#cca = CCA(n_components =2)
#cca.fit(X1_train, X2_train)
cca = PLSCanonical(n_components = 100)
cca.fit(X1_train, X2_train)
block_1_transformed, block_2_transformed = cca.transform(X1_train, X2_train, copy=False)
block_3_transformed = np.dot(X2_remain, cca.y_rotations_)

edge_data_transformed = np.vstack((block_1_transformed, block_2_transformed, block_3_transformed))
# initialise the classifier

clf = svm.SVC(kernel='precomputed')



# optional shuffle
perm = np.random.permutation(n_subjects)
#print perm
#print n_subjects
labels = labels[perm]
edge_data_transformed = edge_data_transformed[perm, :]
#labels = labels[:140]
#edge_data_transformed = edge_data_transformed[140:, :]
#n_subjects = 140


accs = np.zeros((n_folds, 1))
senss = np.zeros((n_folds, 1))
specs = np.zeros((n_folds, 1))
preds = np.zeros((n_subjects,))

K = np.dot(edge_data_transformed, np.transpose(edge_data_transformed))

# n-fold cross validation
#folds = cross_validation.StratifiedKFold(labels, n_folds = n_folds)
folds = cross_validation.KFold(n_subjects, n_folds = n_folds)
for i, fold in zip(range(n_folds), folds) :
    
    train_index = fold[0]
    test_index = fold[1]

    K_train = K[train_index,:][:, train_index]
    K_test = K[test_index, :][:, train_index]
    labels_train = labels[train_index]
    labels_test = labels[test_index]
    clf.fit(K_train, labels_train)
    fold_preds = clf.predict(K_test)
    preds[test_index] = fold_preds
    accs[i] = metrics.accuracy_score(labels_test, fold_preds)
    senss[i] = float(sum(fold_preds[labels_test == 1] == 1))/sum(labels_test == 1)
    specs[i] = float(sum(fold_preds[labels_test == 0] == 0))/sum(labels_test == 0)
    
print metrics.roc_auc_score(labels, preds)
print metrics.accuracy_score(labels, preds)
print float(sum(preds[labels == 1] == 1))/sum(labels == 1)
print float(sum(preds[labels == 0] == 0))/sum(labels == 0)
print np.std(accs)
print np.std(senss)
print np.std(specs)