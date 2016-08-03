# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:47:59 2016

@author: jonyoung
"""

# import the stuff we need
import numpy as np
from sklearn import svm, cross_validation, metrics
import sys

def mccv(data, labels, classifier, n_iter, test_size, metric):
    
    n = len(data)
    accs = []    
    splits = cross_validation.ShuffleSplit(n, n_iter, test_size) 
    
    for train_index, test_index in splits :
                
        K_train = data[train_index,:][:, train_index]
        K_test = data[test_index, :][:, train_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]
        classifier.fit(K_train, labels_train)
        accs.append(classifier.score(K_test, labels_test))
                
    return accs
    
def mccv_roc(data, labels, classifier, n_iter, test_size, metric):
    
    n = len(data)
    accs = []    
    splits = cross_validation.StratifiedShuffleSplit(labels, n_iter, test_size) 
    
    for train_index, test_index in splits :
                
        K_train = data[train_index,:][:, train_index]
        K_test = data[test_index, :][:, train_index]
        labels_train = labels[train_index]
        labels_test = labels[test_index]
        classifier.fit(K_train, labels_train)
        dv = classifier.decision_function(K_test)
        accs.append(metrics.roc_auc_score(labels_test, dv))
                
    return accs
    
def repeated_k_cv(data, labels, classifier, repeats, k, metric) :
    
    n = len(data)
    accs = np.zeros((repeats,))
    for i in range(repeats) :histogram_level
        
        #print i        
        
        # generate a permutation of the subjects        
        perm = np.random.permutation(range(n))
        
        # shuffle the labels and kernel matrix according to the permutation
        rep_labels = labels[perm]
        rep_K = data[perm, :][:, perm]
        score = cross_validation.cross_val_score(classifier, rep_K, rep_labels - 1, cv=k, scoring=metric)
        accs[i] = np.mean(score)
    
    return accs
        
# directories
kernel_dir = '/home/k1511004/Data/connectivity_data/KCL_SC1/kernels/'
data_dir = '/home/k1511004/Data/connectivity_data/KCL_SC1/'

# parameters for nested CV loop
n_folds = 10
n_reps = 100
n_subsamplings = 100

# read in all kernels
kernel_dict = {}

# loop through gamma values
for i in range(20) :
    
    # loop through sigma values
    # restrict sigma range for speed
    for j in range(10, 19):
        
        # generate unique key for dictionary
        key = (2 ** i) * (3 ** j)
             
        # read in the kernel file
        K = np.genfromtxt(kernel_dir + 'K_log_euclidean_' + str(i-10) + '_' + str(j-10) + '.csv', delimiter=',')

        # pull labels
        if i == 0 and j == 0 :
            
            labels = K[:,0]
            
        # pull out kernel matrix and correct diagonal            
        K = K[:,1:]
        np.fill_diagonal(K, 1)
        #print K[-5:,-5:]
        
        # store it in the dictionary
        kernel_dict.update({key: K})

# set up results storage: 6 numbers of subjects by 100 repreats
results = np.zeros((6, n_subsamplings))

# get labels
labels = np.genfromtxt(kernel_dir + 'K_log_euclidean_' + str(i-10) + '_' + str(j-10) + '.csv', delimiter=',')[:, 0]
     
# loop through different number of subjects: 40, 60, 80, 100, 120, 140
# number of subjects comes from the command line args
n_subjects = int(sys.argv[1]) * 20     

print 'number of subjects = ' + str(n_subjects)
    
# vector of results for this number of subjects
n_subjects_results = np.zeros((n_subsamplings,))
    
# generate n_subsamplings subsamplings
for j in range(n_subsamplings) :
        
    print 'subsampling number = ' + str(j)
        
    # permute to generate subsampling
    perm = np.random.permutation(140)
    subsampling_indices = perm[:n_subjects]
        
    # subsample the labels
    labels_subsample = labels[subsampling_indices]
        
    # loop to get results
    parameter_results = np.zeros((20, 9, 12))
        
    kf = cross_validation.StratifiedKFold(labels_subsample, n_folds, shuffle=True) 
    ind = 0
        
    subsample_results = np.zeros((n_subjects,))
        
    fold_ind = 0
        
    for train_index, test_index in kf:
    
        print 'ind = ' + str(fold_ind)
    
        labels_fold_train = labels_subsample[train_index]
        labels_fold_test = labels_subsample[test_index]
        
                
        for k in range(20) :
        
            for l in range(10, 19) :
            
                # load the kernel
                key = (2 ** k) * (3 ** l)
                K_fold = kernel_dict[key]
                
                # apply subsample to kernel and labels
                K_subsample = K_fold[subsampling_indices, :][:, subsampling_indices]
            
                # loop through values of SVM C parameter
                for m in range(12) :
                
                    c_val = 2 ** m
                    clf = svm.SVC(kernel='precomputed', C=c_val)

                    #print 'gamma = ' + str(2 ** (i - 10))
                    #print 'sigma = ' + str(2 ** (j - 10))          
            
                    # get MCCV results from training data/labels            
                    K_fold_train = K_fold[train_index, :][:, train_index]
                    parameter_results[k, l-10, m] = np.mean(mccv(K_fold_train, labels_fold_train, clf, n_reps, 0.1, 'accuracy_score'))
            
        # find most accurate set of parameters
        best_parameters = np.unravel_index(parameter_results.argmax(), parameter_results.shape)
            
        # train on training data with the best parameters and test on the left over data
        key = (2 ** best_parameters[0]) * (3 ** (best_parameters[1] + 10))
        c_val = 2 ** best_parameters[2]
        K = kernel_dict[key]
        K_subsample = K[subsampling_indices, :][:, subsampling_indices]
        K_train = K_subsample[train_index, :][:, train_index]
        K_test = K_subsample[test_index, :][:, train_index]            
        clf = svm.SVC(kernel='precomputed', C=c_val)
        clf.fit(K_train, labels_fold_train)
        subsample_results[test_index] = clf.predict(K_test)
            
        fold_ind = fold_ind + 1
            
    n_subjects_results[j] = metrics.accuracy_score(subsample_results, labels_subsample)
    
# save the results for this number of subejcts
np.savetxt(output_dir + 'compare_' + str(n_subjects) + '_subjects_LE.csv', n_subjects_results, delimiter=',')
