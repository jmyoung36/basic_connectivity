# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:12:04 2016

@author: jonyoung
"""

import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV, ledoit_wolf, GraphLasso
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from connectivity_utils import load_timecourse_data
import csv

# directories
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/'
timecourse_dir = data_dir + 'KCL3_timecourse/'
output_dir = timecourse_dir


# Generate the data
timecourse_data, timecourse_files = load_timecourse_data(timecourse_dir)

# generate structure to hold data
sparse_inverse_covariance_matrices = np.zeros((len(timecourse_data), 8100))

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
    #cov = np.dot(X, np.transpose(X)) / np.shape(X)[1]
    #print cov[:5, :5]
    #print linalg.cholesky(cov)
    
    # calculate sparse inverse covariance (precision) matrix
    model = GraphLassoCV(max_iter=1500, assume_centered=True)
    model.fit(np.transpose(X))
    cov = model.covariance_
    sparse_inverse_covariance_matrices[i, :] = np.reshape(cov, (1, 8100))
    print model.cv_alphas_
    print model.alpha_
    print cov[:5, :5]
    print linalg.logm(cov)[:5, :5]
    
# save the data
np.savetxt(output_dir + 'sparse_inverse_covariance_data.txt', sparse_inverse_covariance_matrices, delimiter=',')

with open(output_dir + 'sparse_inverse_covariance_files.csv', 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(timecourse_files)