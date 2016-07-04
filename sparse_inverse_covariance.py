# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:12:04 2016

@author: jonyoung
"""

import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphLassoCV, ledoit_wolf
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from connectivity_utils import load_timecourse_data


# Generate the data
timecourse_data = load_timecourse_data('/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC_Unsmooth_TimeCourse/')

# roll through the subjects
print np.shape(timecourse_data)[0]
#for i in range(np.shape(timecourse_data)[0]) :
for i in range(10) :
    
    # extract the timecourses for this subejct
    subject_timecourses = timecourse_data[i, : ,:]
    print np.shape(subject_timecourses)
    
    # calculate Pearson covariance
    X = scale(subject_timecourses, axis=1)
    cov = np.dot(X, np.transpose(X)) / np.shape(X)[1]
    print cov[:5, :5]
    
    # calculate sparse inverse covariance (precision) matrix
    model = GraphLassoCV()
    model.fit(X)
    cov = model.covariance_
    model = GraphLassoCV()
    print cov[:5, :5]
