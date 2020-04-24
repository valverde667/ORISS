#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:38:47 2020

@author: nickvalverde
"""

#Script will first clean trajctoryfile so that it can be read into a pandas DataFrame.
#The script will then be able to plot using matplotlib or pandas plotting capabilities.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import re
mm = 1e-3
ms = 1e-3
time = 1e-7
# =============================================================================
# #Open Trajectory file and create a new file named cleaned that will be a comma seperated file.
# trajectory = open('/Users/nickvalverde/Research/ORISS/trajectoryfile.txt')
# cleaned = open('/Users/nickvalverde/Research/ORISS/cleaned', 'w')
# cleaned_excel = open('/Users/nickvalverde/Research/ORISS/cleaned_excel.txt', 'w')
# #Replaces the spaces in the trajectory file with ','s using a regex. The command
# # re.sub(r' +', r',', line) first matches any pattern with a space, replacs it with a comma, and does this for each
# #line in the file.
# # =============================================================================
# # for line in trajectory:
# #     cleaned.write(re.sub(r' +', r',', line))
# #     cleaned_excel.write(re.sub(r' +', r',', line))
# # #Close files.
# # trajectory.close()
# # cleaned.close()
# # cleaned_excel.close()
# #
# # =============================================================================
# =============================================================================
#Read in the data into a dataframe called 'data'.
column_names = ['Particle', "Iter", "zp[i]", "uzp[i]", "xp[i]", "uxp[i]"]
data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/trajectoryfile.txt', names = column_names)
tracked_data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/tracked_particle.txt', names = column_names[1:])
                                                 #where each line is given by a line break '\n'
copy = data.copy() #Create copy of data frame
copy['time'] = copy['Iter']*1e-7
tracked_data['time'] = tracked_data['Iter']*1e-7






for i in range(len(copy['Particle'].unique())):
    if (len(copy[copy['Particle'] == i]) != len(tracked_data)):
        copy.drop(copy[copy['Particle'] == i].index, inplace = True)
    else:
        pass










## --Point to Point
fig, ax = plt.subplots(figsize = (7,7))
ax.scatter(copy['time']/ms, copy['zp[i]']/mm, c = 'k', s = 0.5)
ax.plot(tracked_data['time']/ms, tracked_data['zp[i]']/mm, c = 'r', lw = 0.9) #Plot Tracer

ax.set_xlabel(r'Time [milliseconds]')
ax.set_title(r' Particle Energy = %g [kV]' %(2.77))
ax.set_ylabel('z [mm]')
ax.axhline(y=0, alpha=0.7, c = 'k', lw = 0.5)

plt.tight_layout()
plt.savefig('/Users/nickvalverde/Dropbox/Research/ORISS/Plot_Reproductions/z-trajectory.png', dpi=300)



##--Parallel to point
rmin = 60. #mm
fig, ax = plt.subplots(figsize = (7,7))
ax.scatter(copy['zp[i]']/mm, copy['xp[i]']/mm, s = 0.5, c = 'k')
ax.plot(tracked_data['zp[i]']/mm, tracked_data['xp[i]']/mm, lw = 0.9, c = 'r')

ax.set_xlabel('z [mm]')
ax.set_title('Parallel to Point')
ax.set_ylabel('x [mm]')
ax.axhline(y=0, alpha=0.7, c = 'k', lw = 0.5)
ax.axvline(x=0, alpha=0.7, c = 'k', lw = 0.5)

plt.tight_layout()
plt.savefig('/Users/nickvalverde/Dropbox/Research/ORISS/Plot_Reproductions/x-trajectory.png', dpi=300)



#--x Moment Plots
fig, ax = plt.subplots(figsize = (7,7))

iteration_list = []
stdx_list = []
neg_vertical_extent = []
for i in range(len(copy['Iter'].unique())):
    sample = copy[copy['Iter'] == i]
    stdx_list.append(sample['xp[i]'].std())
    iteration_list.append(i)

iteration_list = np.array(iteration_list)*time
stdx_list = np.array(stdx_list)


ax.plot(iteration_list/ms, stdx_list/mm)
ax.set_xlabel(r'Time [milliseconds]')
ax.set_title('Transverse RMS')
ax.set_ylabel('x [mm]')
ax.axhline(y=0, alpha=0.7, c = 'k', lw = 0.5)

plt.savefig('/Users/nickvalverde/Dropbox/Research/ORISS/Plot_Reproductions/rms_x.png', dpi=300)

#--z Moment Plots
fig, ax = plt.subplots(figsize = (7,7))

iteration_list = []
stdz_list = []
neg_vertical_extent = []
for i in range(len(copy['Iter'].unique())):
    sample = copy[copy['Iter'] == i]
    stdz_list.append(sample['zp[i]'].std())
    iteration_list.append(i)

iteration_list = np.array(iteration_list)*time
stdz_list = np.array(stdz_list)



ax.plot(iteration_list/ms, stdz_list/mm)
ax.set_xlabel(r'Time [milliseconds]')
ax.set_title('Transverse RMS')
ax.set_ylabel('z [mm]')
ax.axhline(y=0, alpha=0.7, c = 'k', lw = 0.5)

plt.savefig('/Users/nickvalverde/Dropbox/Research/ORISS/Plot_Reproductions/rms_z.png', dpi=300)















#Further additions:

##Done
#--split dataframe and plot each particle. This can be done using the .unique() on the DF.
#For example, to get particle 0: df_particle0 = data[data['Particle'] == 0]. This will
#create a new data frame with the particle 0 elements. From here, plotting is nothing but selecting columns.
#A for loop can plot all simultaneously:
#for part in data['Particle'].unique():
#   df_particlePart = data[data['Particle'] == 0]
#   plot wanted elements.


#--Use plotting file to update itself after a certain amount of time.
#This will be useful for watching the trajectory. For efficiency, the trajectory file should be updated
#started with the last known elements rather than recreating the entire file.
#Should also allow user to pick the time intervals for plotting; for example, plot every 1, 2, 10, 20, etc. seconds.
#Allow for figure to adjust size/resolution.
#Subplots for longitudinal and transverse trajectories.

#--Clean up Oriss formatting for trajectory file so that it is more standard. I pretty much
#eyeballed the formatting while also trying to quantify the spacing. Right now it is a hybrid of
#quantifiable spacing and eye-balling.
