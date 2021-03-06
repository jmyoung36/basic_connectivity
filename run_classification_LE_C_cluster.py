# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:08:31 2016

@author: jonyoung
"""

# import the stuff we need
import numpy as np
from sklearn import svm, cross_validation, metrics
import sys

#def mccv(data, labels, classifier, n_iter, test_size):
#    
#    n = len(data)
#    accs = []
#    
#    i = 0
#    
#    splits = cross_validation.ShuffleSplit(n, n_iter, test_size)
#    for train_index, test_index in splits :
#        
#        #print i
#        
#        K_train = data[train_index,:][:, train_index]
#        K_test = data[test_index, :][:, train_index]
#        labels_train = labels[train_index]
#        labels_test = labels[test_index]
#        classifier.fit(K_train, labels_train)
#        accs.append(classifier.score(K_test, labels_test))
#        
#        i+=1
#        
#    return accs

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
kernel_dir = '/home/k1511004/Data/connectivity_data/KCL_SC1/kernels/'
#kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/kernels/'
results_dir = '/home/k1511004/Data/connectivity_data/'
#results_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/'
       

# parameters for nested CV loop
fold = int(sys.argv[1])
n_reps = 250

# read in all kernels
kernel_dict = {}

# loop through gamma values
for i in range(20) :
    
    # loop through sigma values
    for j in range(20):
        
        # generate unique key for dictionary
        key = (2 ** i) * (3 ** j)
             
        # read in the kernel file
        K = np.genfromtxt(kernel_dir + 'K_log_euclidean_sparse_inverse_cov_' + str(i-10) + '_' + str(j-10) + '.csv', delimiter=',')

        # pull labels and size
        if i == 0 and j == 0 :
            
            labels = K[:,0]
            n_subjects = np.shape(K)[0]
            
        # pull out kernel matrix and correct diagonal            
        K = K[:,1:]
        np.fill_diagonal(K, 1)
        #print K[-5:,-5:]
        
        # store it in the dictionary
        kernel_dict.update({key: K})
  
# create a data structure to hold grid search results, selected parameters, fold results and predictions
grid_search_results = np.zeros((20, 20, 12))
preds = np.zeros((len(labels),))


clf = svm.SVC(kernel='precomputed') 

# one fold of nested grid 10-fold cv with inner loop to select parameters
step_size = int(np.ceil(n_subjects/10.0))
start_ind = (fold-1) * step_size
stop_ind = np.min((start_ind + step_size, n_subjects))
train_index = np.arange(n_subjects)
train_index = np.delete(train_index, range(start_ind, stop_ind))
test_index = np.arange(n_subjects)[start_ind:stop_ind]

print 'fold = ' + str(fold)
    
labels_fold_train = labels[train_index]
labels_fold_test = labels[test_index]
    
# do a grid search on the training group
    
# loop through possible parameters    
parameter_results = np.zeros((20, 20, 12))
    
for i in range(20) :
    
    print i
        
    for j in range(20) :
        
        print j
            
        # load the kernel
        key = (2 ** i) * (3 ** j)
        K_fold = kernel_dict[key]
            
            
        # loop through values of SVM C parameter
        for k in range(12) :
                
            c_val = 2 ** k
            clf = svm.SVC(kernel='precomputed', C=c_val)

            #print 'gamma = ' + str(2 ** (i - 10))
            #print 'sigma = ' + str(2 ** (j - 10))
            
            # get MCCV results from training data/labels            
            K_fold_train = K_fold[train_index, :][:, train_index]
            parameter_results[i, j, k] = np.mean(mccv(K_fold_train, labels_fold_train, clf, n_reps, 0.1, 'accuracy_score'))
                    
#print 'parameter results:'
print parameter_results
           
# find most accurate set of parameters
best_parameters = np.unravel_index(parameter_results.argmax(), parameter_results.shape)
                           
# train on training data with the best parameters and test on the left over data
key = (2 ** best_parameters[0]) * (3 ** best_parameters[1])
c_val = 2 ** best_parameters[2]
print 'best_parameters = ' + str(best_parameters)
print 'best value of gamma is ' + str(2 ** (best_parameters[0] - 10))    
print 'best value of sigma is ' + str(2 ** (best_parameters[1] - 10))  
print 'best value of c is ' + str(c_val)     
K = kernel_dict[key]
K_train = K[train_index, :][:, train_index]
K_test = K[test_index, :][:, train_index]
clf = svm.SVC(kernel='precomputed', C=c_val)
clf.fit(K_train, labels_fold_train)
fold_preds = clf.predict(K_test)
fold_acc = metrics.accuracy_score(labels_fold_test, fold_preds)
print 'Accuracy for this fold = ' + str(fold_acc)

# save prediction results and labels for this fold
fold_results = np.transpose(np.vstack((labels_fold_test, fold_preds))) 
np.savetxt(results_dir + 'LE_C_cluster_results_fold_' + str(fold) + '.csv', fold_results, delimiter=',')

# save parameter results for this fold
np.save(results_dir + 'LE_C_cluster_parameters_fold_' + str(fold) + '.npy', parameter_results)
