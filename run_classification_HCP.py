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
from scipy.linalg import logm

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
connectivity_data = np.zeros((len(HCP_metadata), n_lower_elements))

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
    connectivity_file_data = np.loadtxt(data_dir + str(n_regions) + '_regions/' + connectivity_file, delimiter = ',')
    
    # take the matrix log
    connectivity_file_data = logm(connectivity_file_data)
    connectivity_data[i, :] = connectivity_file_data[np.tril_indices(n_regions, k=-1)]


# compute kernel matrix
K = np.dot(connectivity_data, np.transpose(connectivity_data))

# initialise the classifier
clf = svm.SVC(kernel='precomputed')

accs = mccv(K, labels, clf, 1000, 0.1, 'acc')
print accs
print np.mean(accs)









