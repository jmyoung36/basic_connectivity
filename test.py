# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:35:54 2016

@author: jonyoung
"""

# import what we need
import numpy as np
import utils

# set directories
data_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/matrix_unsmooth/'

# read in some data
with open(data_dir + '00.txt') as infile:
    data = infile.readlines()
    
# split by tab separation
data = map(lambda x: x.split('\t'), data)

# remove returns/newlines and convert to float
data = map(lambda x: [float(element) for element in x if not element == '\r\n'], data)
print data
print len(data)
print len(data[0])

connectivity_matrix = np.array(data)
#print connectivity_matrix
#print np.size(connectivity_matrix)
#print connectivity_matrix[:6, :6]

M = np.array([[0, 1, 2, 3], [1, 0, 2, 0], [2, 2, 0, 3], [3, 0, 3, 0]])
print M
print np.sum(M, axis=1)
I = np.eye(4)
print np.diag(np.sum(M, axis=1))