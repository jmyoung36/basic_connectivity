# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:21:20 2016

@author: jonyoung
"""

import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV, ledoit_wolf, GraphLasso, OAS
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from connectivity_utils import load_timecourse_data
import csv
from scipy.linalg import logm


# Generate the data
timecourse_data, timecourse_files = load_timecourse_data('/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC_Unsmooth_TimeCourse/')

# generate structure to hold data
OAS_matrices = np.zeros((len(timecourse_data), 8100))

print timecourse_files

# roll through the subjects
print np.shape(timecourse_data)[0]
for i in range(np.shape(timecourse_data)[0]) :
#for i in range(10) :

    print i
    
    # extract the timecourses for this subejct
    subject_timecourses = timecourse_data[i, : ,:]
    #print np.shape(subject_timecourses)
    
    # calculate Pearson covariance
    X = scale(subject_timecourses, axis=1)
    cov = np.dot(X, np.transpose(X)) / np.shape(X)[1]
    print cov[:5, :5]
    print logm(cov)[:5, :5]
    
    # calculate sparse inverse covariance (precision) matrix
    model = OAS(store_precision=False, assume_centered=True)
    model.fit(np.transpose(X))
    cov = model.covariance_
    OAS_matrices[i, :] = np.reshape(cov, (1, 8100))
    #print cov[:5, :5]
    foo = logm(cov)
    #print logm(cov[:5, :5])
    
    
## save the data
np.savetxt('/home/jonyoung/IoP_data/Data/connectivity_data/OAS_data.csv', OAS_matrices, delimiter=',')