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


#Some useful defintions.
mm = 1e-3
ms = 1e-3
time = 1e-7
kV = 1000.

#--Read in variables and assign values from parameters.txt. 
#Variables that are in the parameters.txt file
variable_list = ["particle_energy", "mass", "time_step", "Np_initial", "zcentr8", "zcentr7", \
                    "zcentr6", "zcentr5", "zcentr4", "zcentr3", "zcentr2", "zcentr1"]

#Read in values and assign to variables.
f = open("parameters.txt", "r")
for value, variable in zip(f,variable_list):
  vars()[variable] = eval(value)
f.close()

#--Create two dataframes. One that will be for the ideal particle (tracked_data)
#  and one that will be for all other particles (data)
#Create dataframes
column_names = ['Particle', "Iter", "zp[i]", "uzp[i]", "xp[i]", "uxp[i]"]
data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/trajectoryfile.txt', names = column_names)
tracked_data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/tracked_particle.txt', names = column_names[1:])

#Add time column to dataframes. Note that time_step is read in from parameters.txt
copy = data.copy() #Create copy of data
copy['time'] = copy['Iter']*time_step
tracked_data['time'] = tracked_data['Iter']*time_step

#--Cleans dataframe of lost particles if needed. 
# for i in range(len(copy['Particle'].unique())):
#     if (len(copy[copy['Particle'] == i]) != len(tracked_data)):
#         copy.drop(copy[copy['Particle'] == i].index, inplace = True)
#     else:
#         pass

Np = len(copy['Particle'].unique()) #number of particles including lost particles. 


#--Create figures for plotting
xmin = copy['xp[i]'].min()
xmax = copy['xp[i]'].max()
zmin = copy['zp[i]'].min()
zmax = copy['zp[i]'].max()


fig, ax = plt.subplots(figsize=(7,7))
plt.xlim(zmin/mm, zmax/mm)
plt.ylim(xmin/mm, xmax/mm)
plt.xlabel("z [mm]",fontsize=20)
plt.ylabel("x [mm]",fontsize=20)
plt.title('Point to Parallel')


#--Movie Creation
#Initialize empty scattery plot to be updated for the tracker particle and all others.
ideal_scat = ax.scatter([], [], s = 3, c = 'r') #tracker particle scatter plot
scat = ax.scatter([], [], s = .5, c = 'k') #all other particles


#This function allows the use of Blit in animation
def init():
    scat.set_offsets([])
    ideal_scat.set_offsets([])
    return scat, ideal_scat

#Animation function. This will update the scatterplot coordinates for the ith frame.
def animate(i):

    print(i) #useful to see how fast the program is running and when it will finish.

    ideal_coords = tracked_data[tracked_data['Iter'] == i] #Creates data frame for ith iteration
    ideal_x = ideal_coords['xp[i]']/mm #x-coordinate on tracker particle for ith iteration
    ideal_z = ideal_coords['zp[i]']/mm #z-coordinate on tracker particle for ith iteration

    coords = copy[copy['Iter'] == i] #Creates a data frame for all other particles for the ith iteration
    xpoints = coords['xp[i]']/mm     #x-coordinates for all other particles for ith iteration
    zpoints = coords['zp[i]']/mm     #z-coordinates on tracker particle for ith iteration

    #Create arrays to feed into scatter plots using scat.set_offsets. 
    #It is important to note tha set_offsets takes in (N,2) arrays. This is the reasoning for 
    #adding np.newaxis the command. These arrays are then ((x,y), newaxis). 
    plot_points = np.hstack((zpoints[:, np.newaxis], xpoints[:, np.newaxis])) 
    ideal_points = np.hstack((ideal_z[:, np.newaxis], ideal_x[:, np.newaxis]))

    #Eneter in new plot points. 
    scat.set_offsets(plot_points)
    ideal_scat.set_offsets(ideal_points)

    #Update plots
    return scat,ideal_scat





#--Animating and saving
num_of_frames = len(copy['Iter'].unique()) #number of times animate() will be called
Writer = animation.writers['ffmpeg'] #for saving purposes
writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800) #Some video settings. 
#Ani is the actual animations. interval is interval between frames in microsecons. Blit is what speeds up the animation process
#by only plotting the changes rather then replotting each frame. 
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames-1, interval=100, repeat=True, blit = True) #The actual animator
plt.show()
ani.save('/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Plots/scatter_blit.mp4', writer=writer)
