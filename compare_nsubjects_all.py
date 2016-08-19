# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 14:26:47 2016

@author: jonyoung
"""

# import the stuff we need
import numpy as np
from sklearn import svm, cross_validation, metrics
import matplotlib.pyplot as plt


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
    for i in range(repeats) :
        
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
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/kernels/'

# parameters for nested CV loop
n_folds = 10
n_subsamplings = 100

# read in the kernel
K = np.genfromtxt(kernel_dir + 'K_edge.csv', delimiter=',')

# set up results storage: 6 numbers of subjects by 100 repreats
results_linear = np.zeros((6, n_subsamplings))
results_LE = np.zeros((6, n_subsamplings))

# split kernel data and labels
labels = K[:, 0]
K = K[:,1:]

# define classifier
clf = svm.SVC(kernel='precomputed')
     
# loop through different number of subjects: 40, 60, 80, 100, 120, 140
for i in range(2, 8) :

    n_subjects = i * 20
    
    print 'number of subjects = ' + str(n_subjects)
    
    #################################################
    ######## do linear kernel calculations ##########
    #################################################
   
    # vector of results for this number of subjects
    n_subjects_results = np.zeros((n_subsamplings,))
    
    # generate n_subsamplings subsamplings
    for j in range(n_subsamplings) :
        
        print 'subsampling number = ' + str(j)
        
        # permute to generate subsampling
        perm = np.random.permutation(140)
        subsampling_indices = perm[:n_subjects]
        
        # subsample the labels and kernel
        labels_subsample = labels[subsampling_indices]
        K_subsample= K[:, subsampling_indices][subsampling_indices, :]
        
        
        # 10-fold x validation
        kf = cross_validation.StratifiedKFold(labels_subsample, n_folds, shuffle=True) 
        ind = 0
        
        subsample_results = np.zeros((n_subjects,))
        
        fold_ind = 0
        
        for train_index, test_index in kf:
    
            # generate 
            labels_fold_train = labels_subsample[train_index]
            labels_fold_test = labels_subsample[test_index]
            K_train = K_subsample[train_index, :][:, train_index]
            K_test = K_subsample[test_index, :][:, train_index]          
            
            clf.fit(K_train, labels_fold_train)
            subsample_results[test_index] = clf.predict(K_test)
            
            fold_ind = fold_ind + 1
            
        n_subjects_results[j] = metrics.accuracy_score(subsample_results, labels_subsample)
        
    results_linear[i-2, :]= n_subjects_results
    
    #################################################
    ######## read in precomputed LE results #########
    #################################################
    
    results_LE[i-2, :] = np.genfromtxt('/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/compare_' + str(n_subjects) + '_subjects_LE.csv', delimiter = ',')
        
print np.mean(results_linear, axis=1)
print np.std(results_linear, axis=1)
    
x = np.arange(40, 160, 20)   
plt.scatter(x, np.mean(results_linear, axis=1), color='red')
plt.errorbar(x, np.mean(results_linear, axis=1), yerr=np.std(results_linear, axis=1), color='red')
plt.scatter(x, np.mean(results_LE, axis=1), color='blue')
plt.errorbar(x, np.mean(results_LE, axis=1), yerr=np.std(results_LE, axis=1), color='blue')
plt.show()    