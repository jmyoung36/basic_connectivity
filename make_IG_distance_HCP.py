# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:40:51 2016

@author: jonyoung
"""

# import the stuff we need
import numpy as np
from sklearn import svm, cross_validation, metrics
import pandas as pd
import os
from scipy.linalg import logm, fractional_matrix_power, norm

# SQUARED DISTANCE between two covariance matrices according to eq'n 2 in Barachant,
# Alexandre, and Marco Congedo. "A Plug & Play P300 BCI Using Information 
# Geometry." arXiv preprint arXiv:1409.0107 (2014).
def sq_IG_distance(cov_1, cov_2) :
    
    cov_1_pow = fractional_matrix_power(cov_1, -0.5)
    return norm(logm(np.linalg.multi_dot([cov_1_pow, cov_2, cov_1_pow])), ord='fro') ** 2

# directories
data_dir = '/home/jonyoung/IoP_data/Data/HCP_PTN820/QUIC_connectivity/'
metadata_dir = '/home/jonyoung/IoP_data/Data/HCP_PTN820/'

# number of regions to use
n_regions = 25

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
connectivity_matrices = np.zeros((len(HCP_metadata), n_regions, n_regions))

# get the subject list and labels (gender)
connectivity_files = HCP_metadata['connectivity file'].tolist()
labels = HCP_metadata['Gender'].tolist()
labels = np.array(labels)
labels[labels == 'F'] = 1
labels[labels == 'M'] = 0
labels = np.array(labels)

# roll through the files
for connectivity_file, i in zip(connectivity_files, range(len(connectivity_files))) :
    
    # read in the file and store it
    connectivity_matrices[i, :, :] = np.loadtxt(data_dir + str(n_regions) + '_regions/' + connectivity_file, delimiter = ',')
    
# create kernel matrices
n_subjects = len(connectivity_files)
sq_IG_dists = np.zeros((n_subjects, n_subjects))
for i in range(n_subjects) :
    print i
    for j in range(i) :
        sq_IG_dist = sq_IG_distance(connectivity_matrices[i, :, :], connectivity_matrices[j, :, :])
        sq_IG_dists[i, j] = sq_IG_dist
        sq_IG_dists[j, i] = sq_IG_dist

labels = np.expand_dims(labels, axis=1).astype(float)

# attach the labels
sq_IG_dists = np.hstack((labels, sq_IG_dists))

# save the distance matrix
np.savetxt(data_dir + 'sq_IG_dists_' + str(n_regions) + '_regions.csv', sq_IG_dists, delimiter=',')