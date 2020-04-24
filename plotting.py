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
kV = 1000.

variable_list = ["particle_energy", "mass", "time_step", "Np_initial", "zcentr8", "zcentr7", \
                    "zcentr6", "zcentr5", "zcentr4", "zcentr3", "zcentr2", "zcentr1"]
    
f = open("parameters.txt", "r")
for value, variable in zip(f,variable_list):
  vars()[variable] = eval(value)
f.close()





def geometry(axis):
    ring_centers = [zcentr8, zcentr7, zcentr6, zcentr5, zcentr4, zcentr3, \
                          zcentr2, zcentr1, -zcentr8, -zcentr7, -zcentr6, -zcentr5, \
                          -zcentr4, -zcentr3, -zcentr2, -zcentr1]
    max_vertical = ax.get_ylim()[1]
    for center in ring_centers:
            axis.plot(center/mm, max_vertical, marker=11, c ='b')
    
    return True


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
particles_lost = 100*(Np_initial - (len(copy['Particle'].unique())-1))/Np_initial

## --Point to Point
fig, ax = plt.subplots(figsize = (7,7))
ax.scatter(copy['time']/ms, copy['zp[i]']/mm, c = 'k', s = 0.3)
ax.plot(tracked_data['time']/ms, tracked_data['zp[i]']/mm, c = 'r', lw = 0.9) #Plot Tracer
ax.set_xlabel(r'Time [milliseconds]')
ax.set_title(r' Particle Energy = {:.2f} [kV], {:.2f}% Particles Lost'.format(particle_energy/kV, particles_lost))
ax.set_ylabel('z [mm]')
ax.axhline(y=0, alpha=0.7, c = 'k', lw = 0.5)

plt.tight_layout()
plt.savefig('/Users/nickvalverde/Dropbox/Research/ORISS/Plot_Reproductions/z-trajectory.png', dpi=300)



##--Parallel to point
fig, ax = plt.subplots(figsize = (7,7))
ax.scatter(copy['zp[i]']/mm, copy['xp[i]']/mm, s = 0.3, c = 'k')
ax.plot(tracked_data['zp[i]']/mm, tracked_data['xp[i]']/mm, lw = 0.9, c = 'r')
geometry(ax)


ax.set_xlabel('z [mm]')
ax.set_title('Parallel to Point, {:.2f}% particles lost'.format(particles_lost))
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
ax.set_title('Transverse RMS, {:.2f}% Particles Lost'.format(particles_lost))
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
ax.set_title('Transverse RMS, {:.2f}% Particles Lost'.format(particles_lost))
ax.set_ylabel('z [mm]')
ax.axhline(y=0, alpha=0.7, c = 'k', lw = 0.5)

plt.savefig('/Users/nickvalverde/Dropbox/Research/ORISS/Plot_Reproductions/rms_z.png', dpi=300)








