# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:36:10 2020

@author: Christian
"""

import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile


df = pd.read_excel('F-16 Data Digitization.xlsx', sheet_name=list(range(1, 49)))
#data = np.zeros((3, 20, 19))
#j = 0
#for i in range(39, 42):
#    data[j, :, :] = df[i].values[1:, 1:]
#    j += 1

#data = np.zeros((20, 19))
#data[:, :] = df[45].values[1:, 1:]

#data = np.zeros(14)
#data[:] = df[47].values[:14, 2]

data = np.zeros((3, 5, 6))
data[0, :, :] = df[48].values[2:7, 9:15]
data[1, :, :] = df[48].values[9:14, 9:15]
data[2, :, :] = df[48].values[16:21, 9:15]

location = './Python Data/Thrust_En(th,H).npy'
np.save(location, data)
datacheck = np.load(location)
