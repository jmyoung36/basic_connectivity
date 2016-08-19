# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:58:05 2016

@author: jonyoung
"""

# import the stuff we need
import numpy as np
from sklearn import metrics

# set directories
results_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/'

# loop through files
for i in range(10) :
    
    # read in results and parameters
    fold_results = np.genfromtxt(results_dir + 'LE_C_cluster_results_fold_' + str(i + 1) + '.csv', delimiter=',')      
    fold_parameters = np.load(results_dir + 'LE_C_cluster_parameters_fold_' + str(i + 1) + '.npy')
    
    # store appropriately
    if i == 0 :
        
        results = fold_results
        parameters = fold_parameters
        
    else :
        
        results = np.vstack((results, fold_results))
        parameters = parameters + fold_parameters
    
acc = metrics.accuracy_score(results[:,0], results[:,1])
sens = sum(results[:,1][results[:,0] == 1] == 1) / float(sum(results[:,0] == 1))    
spec = sum(results[:,1][results[:,0] == 0] == 0) / float(sum(results[:,0] == 0))   
        
print acc
print sens
print spec
