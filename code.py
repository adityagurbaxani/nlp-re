#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:52:50 2023

@author: agurbaxani
"""

import pandas as pd
import numpy as np
import urllib

#target_url = 'https://raw.githubusercontent.com/adityagurbaxani/nlp-re/main/Data/nfr.txt'

#for line in urllib.request.urlopen(target_url):
    #print(type(line.decode('utf-8')))
    #lines.append(line.decode('utf_16'))

#Reading the text data from file
lines = []
with open('./Data/nfr.txt') as f:
    lines = f.readlines()


for i,l in enumerate(lines):
    print(i,l)
    
#Create a DataFrame to process the text
df = pd.DataFrame()
df['Raw'] = lines


#Split each requirement into req-label & req
req_labels = []
req_texts =[]

for i in range(df.shape[0]):
    req = df.iloc[i,0]
    sep_index = req.find(":")
    req_labels.append(req[:sep_index])
    req_texts.append(req[sep_index+1:])

df['req-label'] = req_labels
df['req'] = req_texts