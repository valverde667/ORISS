#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:41:21 2020

@author: nickvalverde
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:03:47 2020

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

scat = ax.scatter([], [], s = .5, c = 'k')
ideal_scat = ax.scatter([], [], s = 3, c = 'r')



def init():
    scat.set_offsets([])
    ideal_scat.set_offsets([])
    return scat, ideal_scat

def animate(i):
   
    print(i)
    
    ideal_coords = tracked_data[tracked_data['Iter'] == i]
    ideal_x = ideal_coords['xp[i]']/mm
    ideal_z = ideal_coords['zp[i]']/mm
    
    coords = testdata[testdata['Iter'] == i]
    xpoints = coords['xp[i]']/mm
    zpoints = coords['zp[i]']/mm
    
    plot_points = np.hstack((zpoints[:, np.newaxis], xpoints[:, np.newaxis]))
    ideal_points = np.hstack((ideal_z[:, np.newaxis], ideal_x[:, np.newaxis]))
    
    
    scat.set_offsets(plot_points)
    ideal_scat.set_offsets(ideal_points)
    
    
    return scat,ideal_scat
    
    
    
    


num_of_frames = len(testdata['Iter'].unique())
Writer = animation.writers['ffmpeg']
writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames-1, interval=100, repeat=True, blit = True)
plt.show()
ani.save('/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Plots/scatter_blit.mp4', writer=writer)
