# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 17:35:54 2016

@author: jonyoung
"""

# import what we need
import numpy as np

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
print connectivity_matrix
print np.size(connectivity_matrix)
print connectivity_matrix[:6, :6]