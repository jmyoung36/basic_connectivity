# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:18:19 2016

@author: jonyoung
"""

# utilities used by all kernel generation algorithms

# import the packages we need
import numpy as np
import glob
import re

# define natural sort key so we sort files into correct (natural) order
# taken from http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort?lq=1
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

# take a directory containing .txt files containing connectivity data and a 
# .csv file with the per-subject labels. Load the connectvity data into a numpy
# matrix with one row per subject and return this along with the set of labels
def load_connectivity_data(data_dir):
    
    # list of data files
    connectivity_files = glob.glob(data_dir + '*.txt')
    n_files = len(connectivity_files)
    
    # sort into numerical order
    connectivity_files.sort(key=natural_sort_key)    
    
    # generate numpy arrays for data & labels
    # each subject is 90 x 90 matrix = 8100 elements
    connectivity_data = np.zeros((n_files, 8100))
    
    # roll through files in order
    for connectivity_file, i in zip(connectivity_files, range(len(connectivity_files))) :
        
        with open(connectivity_file) as infile:
            connectivity_file_data = infile.readlines()
        connectivity_file_data = map(lambda x: x.split('\t'), connectivity_file_data)

        
        # tidy up - split into indidual elements, remove returns/newlines, convert 
        # to float, convert to np array, set diagonals to zero, and reshape
        connectivity_file_data = np.array(map(lambda x: [float(element) for element in x if not element == '\r\n'], connectivity_file_data))
        connectivity_file_data[np.diag_indices(90)] = 0
        connectivity_data[i, :] = np.reshape(connectivity_file_data, (1, 8100))
        
    return connectivity_data
    
def load_labels(data_dir) :
            
    # read in label data and convert to np array
    with open(data_dir + 'Labels.csv') as infile:
        labels = infile.readlines()
    labels = np.array(map(lambda label: int(label[0]), labels))
    return labels
        
    
    

    
    
    