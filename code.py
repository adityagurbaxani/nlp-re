#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 22:52:50 2023

@author: agurbaxani
"""

import pandas as pd
import numpy as np

#Create a DataFrame to process the text

filepath = 'https://raw.githubusercontent.com/adityagurbaxani/nlp-re/2483a5a3f99c211bd770c73c72376d07c45d7dcf/Data/nfr.txt'
with open(filepath,'r') as f:
    lines = f.readlines()
    print(lines)