# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 09:47:15 2016

@author: jonyoung
"""

import pandas as pd

# read in the data
metadata = pd.read_csv('/home/jonyoung/IoP_data/Data/Optimise/Info_Data_20160321_corrected_3.csv')

# create new file name column
metadata['File name'] = metadata['Grey matter images (segment+DARTEL) – Original location'].apply(lambda x: x.split('/')[-1])

# sort by file name
metadata = metadata.sort_values(by='File name')

# save the sorted DF
metadata.to_csv('/home/jonyoung/IoP_data/Data/Optimise/Info_Data_20160321_corrected_sorted.csv')