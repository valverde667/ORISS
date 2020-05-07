#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 18:39:52 2020

@author: nickvalverde
"""

import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
import pandas as pd
import seaborn as sns

#--Subplot to grid takes in two tuples the gridsize (m,n) and the grid location (i,j)
#  For example, a 3x2 grid has 3 rows, 2 columns. Let's make one grid span the entire column o
#  on the right which and have three plots in the first column.
#  I will also turn off all the tick labels for the large plot. This will act as a data table. 
fig = plt.figure(figsize = (8,8))

ax1 = plt.subplot2grid((3,2), (0,0))
ax1.set_title("One")

ax2 = plt.subplot2grid((3,2), (1,0))
ax2.set_title("Two")

ax3 = plt.subplot2grid((3,2), (2,0), colspan = 2)
ax3.set_title("Three")

#Here I make the large plot and turn off the tick marks. An excellent example of tick
#mark manipulation is here https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html
ax4 = plt.subplot2grid((3,2), (0,1), rowspan = 2)
ax4.set_title("Data Sheet")
ax4.yaxis.set_major_locator(plt.NullLocator())
ax4.xaxis.set_major_locator(plt.NullLocator())
#Here I place text on data sheet. 
#Good resource is here https://jakevdp.github.io/mpl_tutorial/tutorial_pages/tut4.html
ax4.text(.30, .95, 'hello world: $\int_0^\infty e^x dx$', size=10, ha='center', va='center')
ax4.text(.30, .85, 'hello world: $\int_0^\infty e^x dx$', size=10, ha='center', va='center')
ax4.text(.30, .75, 'hello world: $\int_0^\infty e^x dx$', size=10, ha='center', va='center')
ax4.text(.30, .65, 'hello world: $\int_0^\infty e^x dx$', size=10, ha='center', va='center')
ax4.text(.30, .55, 'hello world: $\int_0^\infty e^x dx$', size=10, ha='center', va='center')
ax4.text(.30, .45, 'hello world: $\int_0^\infty e^x dx$', size=10, ha='center', va='center')


plt.tight_layout()
plt.show()


