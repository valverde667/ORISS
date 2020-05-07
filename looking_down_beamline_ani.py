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
column_names = ['Particle', "Iter", "zp[i]", "uzp[i]", "xp[i]", "uxp[i]", "yp[i]", "uyp[i]"]
tracked_columns = ["Iter", "zp[i]", "uzp[i]", "xp[i]", "uxp[i]", "Nplost"]

data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/trajectoryfile.txt', names = column_names)
tracked_data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/tracked_particle.txt', names = tracked_columns)

#Add time column to dataframes. Note that time_step is read in from parameters.txt
copy = data.copy() #Create copy of data
copy['time'] = copy['Iter']*time_step
tracked_data['time'] = tracked_data['Iter']*time_step


#Add an r = sqrt(x_i**2 + y_i**2) column
copy['r'] = np.sqrt(copy['xp[i]']**2 + copy['yp[i]']**2)


#--Cleans dataframe of lost particles if needed.
# for i in range(len(copy['Particle'].unique())):
#     if (len(copy[copy['Particle'] == i]) != len(tracked_data)):
#         copy.drop(copy[copy['Particle'] == i].index, inplace = True)
#     else:
#         pass

Np = len(copy['Particle'].unique()) #number of particles including lost particles.


#--Create standard deviations data

stdr_data = []
stdz_data = []

for i in range(len(copy['Iter'].unique())):
    #Get position data for x and z for ith iteration
    r_sample = copy[copy['Iter'] == i]['r']
    z_sample = copy[copy['Iter'] == i]['zp[i]']

    #Calculate standard deviations and append them to lists
    stdr_data.append(r_sample.std())
    stdz_data.append(z_sample.std())

#Set min and max values for plots



#--Create figures for plotting

#Set min and max values for plots
#std plots
stdr_min = min(stdr_data)
stdr_max = max(stdr_data)
stdz_min = min(stdz_data)
stdz_max = max(stdz_data)
#position plots
xmin = copy['xp[i]'].min()
xmax = copy['xp[i]'].max()
ymin = copy['yp[i]'].min()
ymax = copy['yp[i]'].max()

rmin = copy['r'].min()
rmax = copy['r'].max()
zmin = copy['zp[i]'].min()
zmax = copy['zp[i]'].max()

#Will do x-statstics first
fig, ax = plt.subplots(figsize=(7,7))
ax.set_xlim(xmin/mm, xmax/mm)
ax.set_ylim(ymin/mm, ymax/mm)

#-------------------------------------------------------------------------------
# #--Old routine didn't alow specified axis sharing
# fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize=(7,7), sharex = True)
# xax = ax[0]
# stdxax = ax[1]
#-------------------------------------------------------------------------------

ax.axhline(y = 0, lw =.5, c = 'k')
ax.axvline(x = 0, lw =.5, c = 'k')
ax.set_xlabel('x [mm]', fontsize=14)
ax.set_ylabel( "y [mm]",fontsize=14)
ax.set_title('Working Title', fontsize = 18)

def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y

cx, cy = make_circle(25*mm)
plt.plot(cx/mm, cy/mm, lw = .5, c = 'm', label = 'Aperture')
plt.legend() 
plt.tight_layout()
#--Movie Creation

#Initialize empty scattery for trajectory and line plot for RMS visualization
scat = ax.scatter([], [], s = .5, c = 'k') #Create empty scatter plot to be updated
com_scat = ax.scatter([], [], s = 2, c = 'r')
# ideal_scat = rax.scatter([], [], s=1.5, c = 'r')
# line, = stdrax.plot([], [], lw = 1, c = 'k')#create empty line plot to be updated


#This function allows the use of Blit in animation
def init():
    scat.set_offsets([])
    com_scat.set_offsets([])
    #ideal_scat.set_offsets([])
    #line.set_data([], [])
    return scat, com_scat

#Animation function. This will update the scatterplot coordinates for the ith frame.
def animate(i):

    print(i) #useful to see how fast the program is running and when it will finish.
    coords = copy[copy['Iter'] == i]
    xpoints = coords['xp[i]']/mm
    ypoints = coords['yp[i]']/mm
    
    com_xpoint = coords['xp[i]'].mean()/mm
    com_ypoint = coords['yp[i]'].mean()/mm

    # #particle data points
    # coords = copy[copy['Iter'] == i] #Creates a data frame for all other particles for the ith iteration
    # rpoints = coords['r']/mm     #x-coordinates for all other particles for ith iteration
    # zpoints = coords['zp[i]']/mm     #z-coordinates on tracker particle for ith iteration
    #
    # #ideal particle data points
    # ideal_xpoint = tracked_data[tracked_data['Iter'] == i]['xp[i]']/mm #x-coordinate of ideal particle
    # ideal_zpoint = tracked_data[tracked_data['Iter'] == i]['zp[i]']/mm #z-coordinate of ideal particle
    #
    # #x-rms data points
    # stdr_zpoint = tracked_data['zp[i]'][0:i]/mm #z-coordinate from 0 to ith iteration
    # stdr_point = np.array(stdr_data[0:i])/mm #std in x from 0 to ith iteration



    #Create arrays to feed into scatter plots using scat.set_offsets.
    #It is important to note tha set_offsets takes in (N,2) arrays. This is the reasoning for
    #adding np.newaxis the command. These arrays are then ((x,y), newaxis).
    scat_plot_points = np.hstack((xpoints[:, np.newaxis], ypoints[:, np.newaxis]))
    com_scat_plot_points =  np.hstack(((com_xpoint,),(com_ypoint,)))

    #ideal_scat_point = np.hstack((ideal_zpoint[:, np.newaxis], ideal_xpoint[:, np.newaxis]))
    #Enter in new plot points.
    scat.set_offsets(scat_plot_points)
    com_scat.set_offsets(com_scat_plot_points)
    # ideal_scat.set_offsets(ideal_scat_point)
    # line.set_data(stdr_zpoint, stdr_point)
    #Update plots
    return scat, com_scat





#--Animating and saving
num_of_frames = len(copy['Iter'].unique()) #number of times animate() will be called
Writer = animation.writers['ffmpeg'] #for saving purposes
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800) #Some video settings.
#Ani is the actual animations. interval is interval between frames in microseconds. Blit is what speeds up the animation process
#by only plotting the changes rather then replotting each frame.
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames-1, interval=200, repeat=True, blit = True) #The actual animator
plt.show()
ani.save('/Users/nickvalverde/Dropbox/Research/ORISS/Working/Animated_Plots/looking_down_beamline.mp4', writer=writer)
