# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/nickvalverde/Research/trajectoryfile', sep=' ')
#because of formatting, pandas will read in multiple unnamed columns. The following
#commanda drops the unnamed data columns
#data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


fig, ax = plt.subplots(figsize = (10,9))

