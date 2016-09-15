# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 16:28:01 2016

@author: jonyoung
"""

import connectivity_utils as utils
import numpy as np
import scipy.linalg as la
from sklearn.covariance import GraphLassoCV, GraphLasso
from sklearn.preprocessing import StandardScaler, scale
from sklearn.cross_validation import KFold
import py_quic
from scipy.stats import multivariate_normal
import os

def QUICCV(data) :
    
    # values of regularisation parameter lambda to try
    # 0, 0.05, 0.01,...,0.95
    lambdas = np.arange(20) * 0.05
    
    # dimensions of data
    n_timepoints, n_parcelations = np.shape(data)   
    data = np.transpose(data)
        
    # store log-likelihood values - 10 folds x 20 lambda values
    log_liks = np.zeros((20,))
       
    # 10-fold cv loop
    kf = KFold(n_timepoints, 10)
    for train_index, test_index in kf:
        
        # split the data
        training_data = data[:, train_index]
        testing_data = data[:, test_index]
        
        # normalise data with training means and stds
        training_means = np.mean(training_data, axis=1)
        training_stds = np.std(training_data, axis=1)
        training_data_norm = scale(training_data, axis=1)
        testing_data_norm = (testing_data - training_means[:, None]) / training_stds[:, None]      
        
        # calculate sample covariance matrix to initialise QUIC        
        training_sample_cov = np.dot(training_data_norm, np.transpose(training_data_norm))
        training_sample_cov = training_sample_cov / np.shape(training_data)[1]        
        
        # loop through lambda values
        for lambda_val, i in zip(lambdas, range(20)) :

            lambda_val = float(lambda_val)
            
            # do the sparse inverse covariance matrix estimation            
            X, W, opt, cputime, iters, dGap = py_quic.quic(S=training_sample_cov, L=lambda_val, max_iter=100, msg=2)
            
            # rescale the estimated sparse inverse cov matrix W
            cov = W / (1 + lambda_val)
            
            # create a multivariate normal distribution with zero mean and the estimated covariance
            mvn = multivariate_normal(cov=cov)
            
            # calculate log-likelihood of the test data under this mvn
            log_lik = np.apply_along_axis(mvn.logpdf, 0, testing_data)
            log_liks[i] = log_liks[i] + np.sum(log_lik)
            
    # find lambda value giving best CV log likelihood
    best_lambda_val = float(np.argmax(log_liks) * 0.05)

    # do QUIC on whole data with the best lambda
    data_norm = scale(data, axis=1)
    data_sample_cov = np.dot(data_norm, np.transpose(data_norm)) / n_timepoints
    X, W, opt, cputime, iters, dGap = py_quic.quic(S=data_sample_cov, L=best_lambda_val, max_iter=100, msg=2)
    W = W / (1 + best_lambda_val)
    return best_lambda_val, W
    
# directories
data_dir = '/home/jonyoung/IoP_data/Data/HCP_PTN820/node_timeseries/3T_HCP820_MSMAll_d100_ts2/'
output_dir = '/home/jonyoung/IoP_data/Data/HCP_PTN820/QUIC_connectivity/100_regions/'

# list files in directory
timecourse_files = os.listdir(data_dir)

# roll through files
for timecourse_file in timecourse_files :
    
    # read in the file
    timecourse_data = utils.load_hcp_timecourse(data_dir + timecourse_file)
    
    # extract the id number
    timecourse_id = timecourse_file[:-4]
    
    # get the QUIC covariance matrix and estimated lambda
    best_lambda_val, cov = QUICCV(timecourse_data)
    
    # construct output filename - subject id, number of regions, estimated lambda
    cov_filename = output_dir + timecourse_id + '_15_' + str(best_lambda_val) + '_cov.csv'
    
    # save the covariance as a csv
    np.savetxt(cov_filename, cov, delimiter = ',')
    
    print timecourse_file
