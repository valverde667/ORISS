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

for i in range(len(copy['Particle'].unique())):
    if (len(copy[copy['Particle'] == i]) != len(tracked_data)):
        copy.drop(copy[copy['Particle'] == i].index, inplace = True)
    else:
        pass
    
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
def movie(dataset):
    
    testdata = dataset.copy()

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
    #testdata = copy[copy['Particle'] == 2] 
    testdata = dataset.copy()
   
    #--Create Movie 
    for i in range(len(dataset['Iter'].unique())):
        data = dataset[dataset['Iter'] == i]
        #p = sns.scatterplot(x=data['zp[i]']/mm, y=data['xp[i]']/mm, data=data, ax=ax)
        ax.scatter(x=data['zp[i]']/mm, y=data['xp[i]']/mm, s=.1, color = 'k')
        
        appendage = "%g-th_plot.png" % i
        path = "/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Plots/" + appendage
        plt.savefig(path)
        #p.tick_params(labelsize=17)
        #plt.setp(p.lines,linewidth=7)

def animate(i):
    data = testdata[testdata['Iter'] == i]
    #data = testdata.iloc[i*Np: (i+1)*Np] #select data range
    #p = sns.scatterplot(x=data['zp[i]']/mm, y=data['xp[i]']/mm, data=data, ax=ax)
    ax.scatter(x=data['zp[i]']/mm, y=data['xp[i]']/mm, s=.1, color = 'k')
    print(i)
    #p.tick_params(labelsize=17)
    #plt.setp(p.lines,linewidth=7)
    

num_of_frames = len(testdata['Iter'].unique())
Writer = animation.writers['ffmpeg']
writer = Writer(fps=250, metadata=dict(artist='Me'), bitrate=1800)  
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames, interval = 25, repeat=False)
plt.show()
ani.save('/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Plots/Testmovie.mp4', writer=writer)


    
    
    
    
    
    
    
    