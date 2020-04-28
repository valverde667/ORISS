#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:29:48 2020

@author: nickvalverde
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.animation as animation




#import re
mm = 1e-3
ms = 1e-3
time = 1e-7
kV = 1000.

variable_list = ["particle_energy", "mass", "time_step", "Np_initial", "zcentr8", "zcentr7", \
                    "zcentr6", "zcentr5", "zcentr4", "zcentr3", "zcentr2", "zcentr1"]

f = open("parameters.txt", "r")
for value, variable in zip(f,variable_list):
  vars()[variable] = eval(value)
f.close()

#Read in the data into a dataframe called 'data'.
column_names = ['Particle', "Iter", "zp[i]", "uzp[i]", "xp[i]", "uxp[i]"]
data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/trajectoryfile.txt', names = column_names)
tracked_data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/tracked_particle.txt', names = column_names[1:])
                                                 #where each line is given by a line break '\n'
copy = data.copy() #Create copy of data frame
copy['time'] = copy['Iter']*time_step
tracked_data['time'] = tracked_data['Iter']*time_step

# for i in range(len(copy['Particle'].unique())):
#     if (len(copy[copy['Particle'] == i]) != len(tracked_data)):
#         copy.drop(copy[copy['Particle'] == i].index, inplace = True)
#     else:
#         pass
    
Np = len(copy['Particle'].unique())
#testdata = copy[copy['Particle'] == 2]
testdata = copy.copy()
#--Create Movie
xmin = testdata['xp[i]'].min()
xmax = testdata['xp[i]'].max()
zmin = testdata['zp[i]'].min()
zmax = testdata['zp[i]'].max()


fig, ax = plt.subplots(figsize=(7,7))
plt.xlim(zmin/mm, zmax/mm)
plt.ylim(xmin/mm, xmax/mm)
plt.xlabel("z [mm]",fontsize=20)
plt.ylabel("x [mm]",fontsize=20)
plt.title('Point to Parallel')

#--Routine for creating plots to stitch together


color_count = 0
trace_color = 0
sign_list = np.sign(tracked_data['uzp[i]'])
color_list = ['k', 'm', 'c', 'g', 'y', 'b']
trace_color_list = ['r', 'k']
def animate(i):
    global color_count
    global sign_list
    global color_list
    global trace_color
    global trace_color_list
    
    plot_color = color_list[color_count]
    if sign_list[i] != sign_list[i+1]:
        if color_count == 6:
            color_count = 0
        else:
            pass
        color_count += 1 
        trace_color = 1
    else:
        trace_color = 0
        pass
        
    data = testdata[testdata['Iter'] == i]
    tracked_particle = tracked_data.iloc[i]

    #data = testdata.iloc[i*Np: (i+1)*Np] #select data range
    #p = sns.scatterplot(x=data['zp[i]']/mm, y=data['xp[i]']/mm, data=data, ax=ax)
    ax.scatter(x=data['zp[i]']/mm, y=data['xp[i]']/mm, s=.1, color = plot_color)
    ax.scatter(x = tracked_particle['zp[i]']/mm, y = tracked_particle['xp[i]']/mm, \
               s = .2, color = trace_color_list[trace_color] )
    print(i)
    print(color_count)
    #p.tick_params(labelsize=17)
    #plt.setp(p.lines,linewidth=7)


num_of_frames = len(testdata['Iter'].unique())
Writer = animation.writers['ffmpeg']
writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames-1, interval=10000, repeat=False)
plt.show()
ani.save('/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Plots/Testmovie.mp4', writer=writer)
