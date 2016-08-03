# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:15:18 2016

@author: jonyoung
"""
# import the packages we need
import numpy as np
import glob
import re
from sklearn.preprocessing import scale
from scipy.linalg import logm, det, cholesky

# set directories
timecourse_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC_Unsmooth_TimeCourse/'
connectivity_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/'

# define natural sort key so we sort files into correct (natural) order
# taken from http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort?lq=1
def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

# get a list of files - merge timecourse data listing and connectivity data
# listings as there are some subejcts in only one of each list,
timecourse_files = glob.glob(timecourse_dir + '*.txt')
connectivity_files = glob.glob(connectivity_dir + '*.txt')

# extract integer serial numbers
timecourse_serials = map(lambda x: int(x.split('_')[-1][0:-4]), timecourse_files)
connectivity_serials = map(lambda x: int(x.split('/')[-1][0:3]), connectivity_files)
serials = list(set(timecourse_serials).intersection(set(connectivity_serials)))
n_files = len(serials)
 
# sort into numerical order
#timecourse_files.sort(key=natural_sort_key)

# loop through files
for serial in serials :
    
    # print the serial number
    print 'Subject number = ' + str(serial)
    
    # read in the timecourse and calculate a covariance matrix
    timecourse_file = timecourse_dir + 'TimeCourse_' + str(serial).zfill(2) + '.txt'
    timecourse_data = np.transpose(np.genfromtxt(timecourse_file, delimiter='\t')[:, :90])
    print timecourse_data
    print np.shape(timecourse_data)
    #X = scale(timecourse_data, axis=1)
    X = timecourse_data
    cov = np.dot(X, np.transpose(X)) / np.shape(X)[1]
    print np.shape(cov)
    print 'Timecourse covariance matrix:'
    print cov[:5, :5]
    print 'log of Timecourse covariance matrix:'
    print logm(cov)[:5, :5]
    print cholesky(cov)
    # Sylvester's criterion
#    for i in range(1, 91) :
#        
#        M = cov[:i, :i]
#        d = det(M)
#        print str(i), d

    
    # read in the connectivity file     
    serial_str =  str(serial).zfill(3)        
    connectivity_file = connectivity_dir + serial_str + '.txt'
    with open(connectivity_file) as infile:
        connectivity_file_data = infile.readlines()
    connectivity_file_data = map(lambda x: x.split('\t'), connectivity_file_data)
        
    # tidy up - split into indidual elements, remove returns/newlines, convert 
    # to float, convert to np array and reshape
    #connectivity_file_data = np.array(map(lambda x: [float(element) for element in x if not element == '\r\n'], connectivity_file_data))
    #print 'Connectivity matrix:'
    #print connectivity_file_data[:5, :5]
    #print 'log of connecitivty matrix:'
    #print logm(connectivity_file_data)

