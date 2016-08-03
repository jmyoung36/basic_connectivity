# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:18:50 2016

@author: jonyoung
"""

import pandas as pd

metadata_dir = '/home/jonyoung/IoP_data/Data/Optimise/'

# read in the metadata with correct labels
correct_metadata = pd.read_csv(metadata_dir + 'Info_Data_20160321_corrected_sorted.csv')

# use it to create a new DF to merge
to_merge = pd.DataFrame(columns=[['ID','corrected Responder/non-responder']])
to_merge['ID'] = correct_metadata['ID']
to_merge['corrected Responder/non-responder'] = correct_metadata['Responder/non-responder']

# read in the metadata to correct
bad_metadata = pd.read_csv(metadata_dir + 'Info_Data_20160719_pt_hi_stringency_merged.csv')

# merge them to form a new DF
corrected_bad_metadata = pd.merge(to_merge, bad_metadata, on = 'ID')

# save the corrected metadata
corrected_bad_metadata.to_csv(metadata_dir + 'Info_Data_20160719_pt_hi_stringency_merged_correct_labels.csv')
