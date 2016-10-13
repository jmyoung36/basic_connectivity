# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:27:36 2016

@author: jonyoung
"""

# import the stuff we need
import numpy as np
from sklearn import svm, cross_validation, metrics
import pandas as pd
import os
from scipy import linalg as la

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
data_dir = '/home/jonyoung/IoP_data/Data/HCP_PTN820/QUIC_connectivity/'
metadata_dir = '/home/jonyoung/IoP_data/Data/HCP_PTN820/'

# number of regions to use
n_regions = 300

# number of element in lower triangle
n_lower_elements = (n_regions * (n_regions - 1)) / 2

# construct the connectivity directory
connectivity_data_dir = data_dir + str(n_regions) + '_regions/'

# get a list of available connectivity files
connectivity_files = os.listdir(connectivity_data_dir)

# convert it to a DF of connectivity subjects
connectivity_subjects = map(lambda x: int(x[0:6]), connectivity_files)
connectivity_subjects = pd.DataFrame(connectivity_subjects, columns=['subject ID'])
connectivity_subjects['connectivity file'] = pd.Series(connectivity_files)

# read in the HCP metadata to a DF
HCP_metadata = pd.read_csv(metadata_dir + 'HCP_metadata_9_12_2016_8_23_54.csv')

# join on subject id so we only have metadata for the subjects we have a connectivity file for
HCP_metadata = pd.merge(HCP_metadata, connectivity_subjects, how='inner', left_on='Subject', right_on='subject ID')

# allocate memory for connectivity data
connectivity_data = np.zeros((len(HCP_metadata), n_regions, n_regions))

# get the subject list and labels (gender)
connectivity_files = HCP_metadata['connectivity file'].tolist()
labels = HCP_metadata['Gender'].tolist()
labels = np.array(labels)
labels[labels == 'F'] = 1
labels[labels == 'M'] = 0
labels = np.array(labels)

# roll through the files
for connectivity_file, i in zip(connectivity_files, range(len(connectivity_files))) :
    
    # read in the file
    connectivity_data[i, :, :] = np.loadtxt(data_dir + str(n_regions) + '_regions/' + connectivity_file, delimiter = ',')
    
# initialise the classifier
clf = svm.SVC(kernel='precomputed')

# number of iterations for cross validation
n_iter = 1000

# create structure to store results
accs = np.zeros((n_iter,))

# generate splits for cross validation
splits = cross_validation.StratifiedShuffleSplit(labels, n_iter, 0.1) 

i = 0
    
for train_index, test_index in splits :
    
    #print i
                
    training_data = connectivity_data[train_index, :, :]
    testing_data = connectivity_data[test_index, :, :]
    
    # find geometric mean
    # construct base covariance matrix by repeated averaging in tangent space
    base_cov_matrix = np.mean(training_data, axis=0)
    
#    for j in range(20) :
#        
#        print j
#    
#        # project all matrices into the tangent space
#        tangent_matrices = np.zeros_like(training_data)
#        for k in range(len(training_data)) :
#        
#            tangent_matrices[k, :, :] = np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, 0.5), la.logm(np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, -0.5), training_data[k, :, :],la.fractional_matrix_power(base_cov_matrix, -0.5)])), la.fractional_matrix_power(base_cov_matrix, 0.5)])
#    
#        # calculate the tangent space mean
#        tangent_space_base_cov_matrix = np.mean(tangent_matrices, axis=0)
#        
#        # project new tangent mean back to the manifold
#        base_cov_matrix = np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, 0.5), la.expm(np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, -0.5), tangent_space_base_cov_matrix ,la.fractional_matrix_power(base_cov_matrix, -0.5)])), la.fractional_matrix_power(base_cov_matrix, 0.5)])

    # apply whitening transport and projection for training AND testing data
    transformed_training_data = np.zeros_like(training_data)
    transformed_testing_data = np.zeros_like(testing_data)
    
    base_cov_matrix_pow = la.fractional_matrix_power(base_cov_matrix, -0.5)
    
    for j in range(len(training_data)) :
        
        transformed_training_data[j, :, :] = la.logm(np.linalg.multi_dot([base_cov_matrix_pow, training_data[j, :, :], base_cov_matrix_pow]))
        #print j
        
    for j in range(len(testing_data)) :
        
        transformed_testing_data[j, :, :] = la.logm(np.linalg.multi_dot([base_cov_matrix_pow, testing_data[j, :, :], base_cov_matrix_pow]))
        #print j

    # extract lower triangles from the transformed data
    transformed_training_data = transformed_training_data[:, np.tril_indices(n_regions, k=-1)[0], np.tril_indices(n_regions, k=-1)[1]]
    transformed_testing_data = transformed_testing_data[:, np.tril_indices(n_regions, k=-1)[0], np.tril_indices(n_regions, k=-1)[1]]
    
    #print np.tril_indices(n_regions, k=-1)
    #print np.shape(transformed_testing_data)

    # make kernels from transformed data
    K_train = np.dot(transformed_training_data, np.transpose(transformed_training_data))
    K_test = np.dot(transformed_testing_data, np.transpose(transformed_training_data))
    labels_train = labels[train_index]
    labels_test = labels[test_index]
    clf.fit(K_train, labels_train)
    accs[i] = clf.score(K_test, labels_test)
    
    print accs[i]
    if i % 50 == 0 :
        
        print i
    
    i = i + 1
    
    
print accs
print np.mean(accs)
    





