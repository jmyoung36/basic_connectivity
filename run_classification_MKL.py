# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 08:18:14 2016

@author: jonyoung
"""

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

# import the stuff we need
import numpy as np
from sklearn import svm, cross_validation, metrics
from scipy.spatial.distance import pdist, squareform

# set directory
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/kernels/'

# number of folds for cross-validation
n_folds = 10

# number of subjects
n_subjects = 333

# number of repeats for mccv in inner CV loop
n_reps = 200

# containers for per-fold accuracy, sensitivity and specificity, all predictions
accs = np.zeros((n_folds, 1))
senss = np.zeros((n_folds, 1))
specs = np.zeros((n_folds, 1))
preds = np.zeros((n_subjects,))

# load kernel data
Kernel_data = np.genfromtxt(kernel_dir + 'K_edge.csv', delimiter=',')

# split kernel data into groups (first column) labels (second column) and kernel matrix (everything else)
groups = Kernel_data[:,0]
labels = Kernel_data[:,1]
K_connectivity = Kernel_data[:, 2:]

# create the group kernel
K_group = 1 - squareform(pdist(np.matrix(groups).T, 'Euclidean'))

# initialise the classifier
clf = svm.SVC(kernel='precomputed')

# optional shuffle
perm = np.random.permutation(n_subjects)
labels = labels[perm]
groups = groups[perm]
K_connectivity = K_connectivity[perm, :][:, perm]
K_group = K_group[perm, :][:, perm]

# create a data structure to hold grid search results, selected parameters, fold results and predictions
grid_search_results = np.zeros((20, 1))
fold_results = np.zeros((n_folds,))
preds = np.zeros((len(labels),))
selected_parameters = np.zeros((n_folds, 1))

clf = svm.SVC(kernel='precomputed') 

# proper nested grid 10-fold cv with inner loop to select parameters
kf = cross_validation.StratifiedKFold(labels, n_folds, shuffle=True) 
ind = 0 
for train_index, test_index in kf:
    
    print 'ind = ' + str(ind)
    
    labels_fold_train = labels[train_index]
    labels_fold_test = labels[test_index]
    
    # do a grid search on the training group
    
    # loop through possible parameters    
    parameter_results = np.zeros((20, 1))
    
    for i in range(20) :
        
        connectivity_weight = i * 0.05
        group_weight = 1 - connectivity_weight
        
        K_fold = (connectivity_weight * K_connectivity) + (group_weight * K_group)
        K_fold_train = K_fold[train_index, :][:, train_index]
        parameter_results[i] = np.mean(mccv(K_fold_train, labels_fold_train, clf, n_reps, 0.1, 'accuracy_score'))
                    
    print 'parameter results:'
    print parameter_results
           
    # find most accurate set of parameters
    best_parameters = np.unravel_index(parameter_results.argmax(), parameter_results.shape)
    print best_parameters
                        
#    # store the scores across all parameters for the fold
    grid_search_results = grid_search_results + parameter_results
#    
#    # train on training data with the best parameters and test on the left over data
    connectivity_weight = best_parameters[0] * 0.05
    group_weight = 1 - connectivity_weight
    print 'best value of connectivity weight is ' + str(connectivity_weight)     
    K_fold = (connectivity_weight * K_connectivity) + (group_weight * K_group)
    K_fold_train = K_fold[train_index, :][:, train_index]
    K_fold_test = K_fold[test_index, :][:, train_index]
    clf.fit(K_fold_train, labels_fold_train)
    fold_preds = clf.predict(K_fold_test)
    fold_result = metrics.accuracy_score(labels_fold_test, fold_preds)
    fold_results[ind] = fold_result
    preds[test_index] = fold_preds
    selected_parameters[ind, :] =  np.array([connectivity_weight])
    print 'Accuracy for this fold = ' + str(fold_result)     
    ind += 1

#print 'selected_parameters:'
#print selected_parameters
#print 'LE results:'
#print fold_results                
print np.mean(fold_results)
#print metrics.accuracy_score(labels, preds)
#print np.sqrt(metrics.mean_squared_error(IXI_metadata['AGE'].tolist(), SPMK_results))
#
#print 'SPMK grid search results:'
#print mean_intersection_results/n_folds
#print mean_bhattacharyya_results/n_folds
#
#
## save individual score for plotting
#plot_results_SPMK = np.zeros((2, len(IXI_metadata)))
#plot_results_SPMK[0, :] = IXI_metadata['AGE']
#plot_results_SPMK[1, :] = SPMK_results
#
#plot_results_lin = np.zeros((2, len(IXI_metadata)))
#plot_results_lin[0, :] = IXI_metadata['AGE']
#plot_results_lin[1, :] = lin_results
#
#np.savetxt(metadata_dir + 'plot_results_rescaled_SPMK.csv', plot_results_SPMK, delimiter=',')
#np.savetxt(metadata_dir + 'plot_results_rescaled_lin.csv', plot_results_lin, delimiter=',')
