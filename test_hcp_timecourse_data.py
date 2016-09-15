# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:16:16 2016

@author: jonyoung
"""

import connectivity_utils as utils
import numpy as np
import scipy.linalg as la
from sklearn.covariance import GraphLassoCV, ledoit_wolf, GraphLasso
from sklearn.preprocessing import scale


connectivity_data = utils.load_hcp_matrix('/home/jonyoung/IoP_data/Data/HCP_PTN820/node_timeseries/3T_HCP820_MSMAll_d15_ts2/715950.txt');

print connectivity_data
print np.shape(connectivity_data)
print np.std(connectivity_data, axis = 1)
connectivity_data = connectivity_data[:, :250]
X = scale(connectivity_data, axis=1)
model = GraphLassoCV(max_iter=1500, assume_centered=True)
model.fit(np.transpose(X))