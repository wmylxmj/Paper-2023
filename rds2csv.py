# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 19:08:14 2022

@author: wmy
"""

import pyreadr
import pandas as pd
import os

print("Reading: COVID-19_Unified-Dataset-master/COVID-19.rds")
# also works for RData
result = pyreadr.read_r('COVID-19_Unified-Dataset-master/COVID-19.rds')

# done! 
# result is a dictionary where keys are the name of objects and the values python
# objects. In the case of Rds there is only one object with None as key

# extract the pandas data frame 
df = result[None]

df.to_csv("dataset/COVID-19.csv")

print("Reading: COVID-19_Unified-Dataset-master/Policy.rds")
# also works for RData
result = pyreadr.read_r('COVID-19_Unified-Dataset-master/Policy.rds')

# done! 
# result is a dictionary where keys are the name of objects and the values python
# objects. In the case of Rds there is only one object with None as key

# extract the pandas data frame 
df = result[None]

df.to_csv("dataset/Policy.csv")