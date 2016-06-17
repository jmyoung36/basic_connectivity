# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 14:33:53 2016

@author: jonyoung
"""

# import the stuff we need
import numpy as np
from sklearn import svm, cross_validation, metrics

# set directory
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/kernels/'

# number of folds for cross-validation
n_folds = 10

# number of subjects
n_subjects = 333

# containers for per-fold accuracy, sensitivity and specificity, all predictions
accs = np.zeros((n_folds, 1))
senss = np.zeros((n_folds, 1))
specs = np.zeros((n_folds, 1))
preds = np.zeros((n_subjects,))

# load connectivity data
connectivity_data = np.genfromtxt(kernel_dir + 'adjusted_connectivity_data.csv', delimiter=',')

# load labels
label_data = np.genfromtxt(kernel_dir + 'K_edge.csv', delimiter=',')
labels = label_data[:,0]

# calculate kernel
K = np.dot(connectivity_data, np.transpose(connectivity_data))

# initialise the classifier
clf = svm.SVC(kernel='precomputed')

# optional shuffle
perm = np.random.permutation(n_subjects)
labels = labels[perm]
K = K[perm, :][:, perm]

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

