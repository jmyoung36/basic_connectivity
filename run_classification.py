# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:14:10 2016

@author: jonyoung
"""

# import the stuff we need
import numpy as np
from sklearn import svm, cross_validation, metrics

# set directory
kernel_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/kernels/'

# load kernel data
Kernel_data = np.genfromtxt(kernel_dir + 'K_edge.csv', delimiter=',')

# split kernel data into labels (first column) and kernel matrix (everything else)
labels = Kernel_data[:,0]
K = Kernel_data[:,1:]