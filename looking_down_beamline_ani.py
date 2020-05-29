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
variable_list = ["particle_energy", "mass", "Np" \
                 "time_step", "tperp", "tz", \
                 "sigmax", "sigmay", "sigmaz", \
                 "time_step", "initial_bunchlength"]

#Read in values and assign to variables.
f = open("parameters.txt", "r")
for value, variable in zip(f,variable_list):
  vars()[variable] = eval(value)
f.close()

#--Create two dataframes. One that will be for the ideal particle (tracked_data)
#  and one that will be for all other particles (data)
#Create dataframes
column_names = ['Particle', "Iter", "zp[i]", "uzp[i]", "xp[i]", "uxp[i]", "yp[i]", "uyp[i]"]
tracked_columns = ["Iter", "zp[i]", "uzp[i]", "xp[i]", "uxp[i]", "yp[i]", "uyp[i]", "Nplost"]

data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/nosctrajectoryfile.txt', names = column_names)
tracked_data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/nosctracked_particle.txt', names = tracked_columns)

scdata = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/sctrajectoryfile.txt', names = column_names)
sctracked_data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/sctracked_particle.txt', names = tracked_columns)
#Add time column to dataframes. Note that time_step is read in from parameters.txt
copy = data.copy() #Create copy of data
copy['time'] = copy['Iter']*time_step
tracked_data['time'] = tracked_data['Iter']*time_step

sccopy = scdata.copy() #Create copy of data
sccopy['time'] = sccopy['Iter']*time_step
sctracked_data['time'] = sctracked_data['Iter']*time_step


# #Add an r = sqrt(x_i**2 + y_i**2) column
# copy['r'] = np.sqrt(copy['xp[i]']**2 + copy['yp[i]']**2)
# sccopy['r'] = np.sqrt(sccopy['xp[i]']**2 + sccopy['yp[i]']**2)
#

#--Cleans dataframe of lost particles if needed.
# for i in range(len(copy['Particle'].unique())):
#     if (len(copy[copy['Particle'] == i]) != len(tracked_data)):
#         copy.drop(copy[copy['Particle'] == i].index, inplace = True)
#     else:
#         pass

Np = len(copy['Particle'].unique()) #number of particles including lost particles.


#--Create standard deviations data

#Set min and max values for plots



#--Create figures for plotting


#position plots
xmin = sccopy['xp[i]'].min()
xmax = sccopy['xp[i]'].max()
ymin = sccopy['yp[i]'].min()
ymax = sccopy['yp[i]'].max()

# rmin = sccopy['r'].min()
# rmax = sccopy['r'].max()
# zmin = sccopy['zp[i]'].min()
# zmax = sccopy['zp[i]'].max()

#Will do x-statstics first
fig,axes = plt.subplots(nrows=1, ncols=2,figsize=(14,7), sharey=True)
ax, scax = axes[0], axes[1]
ax.set_xlim(ymin/mm, ymax/mm)
ax.set_ylim(ymin/mm, ymax/mm)
scax.set_xlim(ymin/mm, ymax/mm)

ax.axhline(y = 0, lw =.5, c = 'k')
ax.axvline(x = 0, lw =.5, c = 'k')
ax.set_xlabel('x [mm]', fontsize=12)
ax.set_ylabel( "y [mm]",fontsize=12)
ax.set_title('x vs y (No Space Charge)', fontsize = 12)

scax.axhline(y = 0, lw =.5, c = 'k')
scax.axvline(x = 0, lw =.5, c = 'k')
scax.set_xlabel('x [mm]', fontsize=12)
scax.set_title('x vs y (Space Charge)', fontsize = 12)
plt.tight_layout()
plt.show()
time_text = ax.text(.55,.95, "", transform=ax.transAxes, fontsize = 8)
time_template = "Time = %f [ms]"
# def make_circle(r):
#     t = np.arange(0, np.pi * 2.0, 0.01)
#     t = t.reshape((len(t), 1))
#     x = r * np.cos(t)
#     y = r * np.sin(t)
#     time_text.set_text('')
#     return x, y

# cx, cy = make_circle(25*mm)
# plt.plot(cx/mm, cy/mm, lw = .5, c = 'm', label = 'Aperture')
# plt.legend()
# plt.tight_layout()
#--Movie Creation
#Initialize empty scattery for trajectory and line plot for RMS visualization
scat = ax.scatter([], [], s = .1, c = 'k') #Create empty scatter plot to be updated
com_scat = ax.scatter([], [], s = .1, c = 'r')

scscat = scax.scatter([], [], s = .1, c = 'k') #Create empty scatter plot to be updated
sccom_scat = scax.scatter([], [], s = .1, c = 'r')
# ideal_scat = rax.scatter([], [], s=1.5, c = 'r')
# line, = stdrax.plot([], [], lw = 1, c = 'k')#create empty line plot to be updated


#This function allows the use of Blit in animation
def init():
    scat.set_offsets([])
    com_scat.set_offsets([])
    time_text.set_text('')

    scscat.set_offsets([])
    sccom_scat.set_offsets([])
    #ideal_scat.set_offsets([])
    #line.set_data([], [])
    return scat, com_scat, scscat, sccom_scat, time_text

#Animation function. This will update the scatterplot coordinates for the ith frame.
def animate(i):

    print(i) #useful to see how fast the program is running and when it will finish.
    coords = copy[copy['Iter'] == i]
    xpoints = coords['xp[i]']/mm
    ypoints = coords['yp[i]']/mm

    com_xpoint = coords['xp[i]'].mean()/mm
    com_ypoint = coords['yp[i]'].mean()/mm

    time_point = tracked_data['time'][i]/ms

    sccoords = sccopy[sccopy['Iter'] == i]
    scxpoints = sccoords['xp[i]']/mm
    scypoints = sccoords['yp[i]']/mm

    sccom_xpoint = sccoords['xp[i]'].mean()/mm
    sccom_ypoint = sccoords['yp[i]'].mean()/mm



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

    scscat_plot_points = np.hstack((scxpoints[:, np.newaxis], scypoints[:, np.newaxis]))
    sccom_scat_plot_points =  np.hstack(((sccom_xpoint,),(sccom_ypoint,)))

    #ideal_scat_point = np.hstack((ideal_zpoint[:, np.newaxis], ideal_xpoint[:, np.newaxis]))
    #Enter in new plot points.
    scat.set_offsets(scat_plot_points)
    com_scat.set_offsets(com_scat_plot_points)
    time_text.set_text(time_template % time_point)

    scscat.set_offsets(scscat_plot_points)
    sccom_scat.set_offsets(sccom_scat_plot_points)

    # ideal_scat.set_offsets(ideal_scat_point)
    # line.set_data(stdr_zpoint, stdr_point)
    #Update plots
    return scat, com_scat, scscat, sccom_scat, time_text





#--Animating and saving
num_of_frames = len(copy['Iter'].unique()) #number of times animate() will be called
Writer = animation.writers['ffmpeg'] #for saving purposes
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800) #Some video settings.
#Ani is the actual animations. interval is interval between frames in microseconds. Blit is what speeds up the animation process
#by only plotting the changes rather then replotting each frame.
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames-1, interval=200, repeat=True, blit = True) #The actual animator
ani.save('/Users/nickvalverde/Dropbox/Research/ORISS/Working/Animated_Plots/looking_down_beamline.mp4', writer=writer)
plt.show()
