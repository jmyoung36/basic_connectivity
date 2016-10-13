# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:27:36 2016

@author: jonyoung
"""

# import the stuff we need
import numpy as np
from sklearn import svm, cross_validation, metrics
import pandas as pd
import os
from scipy.linalg import logm
       
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
connectivity_data = np.zeros((len(HCP_metadata), n_lower_elements))

# get the subject list and labels (gender)
connectivity_files = HCP_metadata['connectivity file'].tolist()
labels = HCP_metadata['Gender'].tolist()
labels = np.array(labels)
labels[labels == 'F'] = 1
labels[labels == 'M'] = 0
labels = np.array(labels)

# roll through the files
for connectivity_file, i in zip(connectivity_files, range(len(connectivity_files))) :
    
    # read in the file
    connectivity_file_data = np.loadtxt(data_dir + str(n_regions) + '_regions/' + connectivity_file, delimiter = ',')
    
    # take the matrix log
    connectivity_file_data = logm(connectivity_file_data)
    connectivity_data[i, :] = connectivity_file_data[np.tril_indices(n_regions, k=-1)]

labels = np.expand_dims(labels, 1)
print labels.dtype

print np.shape(labels)
print np.shape(connectivity_data)

# join the labels and data
labeled_data = np.hstack((labels, connectivity_data))
labeled_data = labeled_data.astype(float)

print labeled_data

# save the data
np.savetxt(data_dir + 'labeled_log_connectivity_data_' + str(n_regions) + '.csv', labeled_data, delimiter=',')








