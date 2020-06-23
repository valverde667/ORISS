#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 14:52:09 2020

@author: nickvalverde
"""
#Animate statistics for hand made distributions.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.animation as animation

import warp as wp

file_save = '/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/Animations/Multispecies/'

#Some useful defintions.
mm = 1e-3
ms = 1e-3
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

data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/trajectoryfile.txt', names = column_names)
tracked_data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/tracked_particle.txt', names = tracked_columns)

ndata = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/ntrajectoryfile.txt', names = column_names)
ntracked_data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/ntracked_particle.txt', names = tracked_columns)


copy = data.copy() #Create copy of data
ncopy = ndata.copy()

copy['time'] = copy['Iter']*time_step
ncopy['time'] = ncopy['Iter']*time_step

tracked_data['time'] = tracked_data['Iter']*time_step
ntracked_data['time'] = ntracked_data['Iter']*time_step


#Add an r = sqrt(x_i**2 + y_i**2) column
copy['r'] = np.sqrt(copy['xp[i]']**2 + copy['yp[i]']**2)
ncopy['r'] = np.sqrt(ncopy['xp[i]']**2 + ncopy['yp[i]']**2)

#--Cleans dataframe of lost particles if needed.
# for i in range(len(copy['Particle'].unique())):
#     if (len(copy[copy['Particle'] == i]) != len(tracked_data)):
#         copy.drop(copy[copy['Particle'] == i].index, inplace = True)
#     else:
#         pass

#
# fig,ax = plt.subplots(figsize = (7,7))
# ax.scatter(copy['zp[i]']/mm, copy['xp[i]']/mm, s=.1)
# ax.set_xlabel('z[mm]')
# ax.set_ylabel('x[mm]')
# ax.axhline(y=0,lw=.1,c='k')
# ax.axvline(x=0, lw=.1, c='k')
# plt.tight_layout()
# plt.savefig(file_save+'E{:.6f}transverse.png'.format(particle_energy/kV),dpi=300)
# plt.show()
# raise Exception()

Np = len(copy['Particle'].unique()) #number of particles including lost particles.


#--Create standard deviations data


#--Create Center of Mass data
zcom_coords = []
rcom_coords = []
for i in range(len(copy['Iter'].unique())):
    #Get position data for r and z for ith iteration
    r_sample = copy[copy['Iter'] == i]['r']
    z_sample = copy[copy['Iter'] == i]['zp[i]']

    #Calculate standard deviations and append them to lists
    zcom_coords.append(z_sample.mean())
    rcom_coords.append(r_sample.mean())

zcom_coords = zcom_coords
rcom_coords = rcom_coords

nzcom_coords = []
nrcom_coords = []
for i in range(len(copy['Iter'].unique())):
    #Get position data for r and z for ith iteration
    r_sample = copy[copy['Iter'] == i]['r']
    z_sample = copy[copy['Iter'] == i]['zp[i]']

    #Calculate standard deviations and append them to lists
    nzcom_coords.append(z_sample.mean())
    nrcom_coords.append(r_sample.mean())

nzcom_coords = nzcom_coords
nrcom_coords = nrcom_coords


# #--Create figures for plotting
# fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(7,7))
# uirz = axes[0][0]
# nirz = axes[1][0]
# idens = axes[0][1]
# fdens = axes[1][1]
#
# #-initial data
# uinitial = copy[copy['time']==0]
# ninitial = ncopy[ncopy['time']==0]
#
# #-final data
# ufinal = copy[copy['time']/ms == .379]
# nfinal = ncopy[ncopy['time']/ms == .379]
#
#
# #-initial distribution
# uirz.scatter(x=uinitial['zp[i]']/mm, y=uinitial['r']/mm,\
#              s = .1, c = 'b')
# uirz.set_title(r"Initial Distribution $_{238}U^+$", fontsize=16)
# uirz.set_xlabel("z [mm]",fontsize=14)
# uirz.set_ylabel("r [mm]",fontsize=14)
#
# nirz.scatter(x=ninitial['zp[i]']/mm, y=ninitial['r']/mm,\
#              s = .1, c = 'r')
# nirz.set_title(r"Initial Distribution $_{237}Np^+$", fontsize=16)
# nirz.set_xlabel("z [mm]", fontsize=14)
# nirz.set_ylabel("r [mm]", fontsize=14)
#
# #--Initial Density Histogram
# sns.distplot(uinitial['zp[i]']/mm, hist=True, color='b', label=r'$_{238}U^+$', ax=idens)
# sns.distplot(ninitial['zp[i]']/mm, hist=True, color='r', label=r'$_{237}Np^+$', ax=idens)
# idens.set_title("Initial Distribution of Particles",fontsize=16)
# idens.set_xlabel('z [mm]', fontsize=14)
# idens.set_ylabel('Number of Particles',fontsize=14)
# idens.legend()
#
#
# #--Final distributions
# sns.distplot(ufinal['zp[i]']/mm, hist=True, color='b', label=r'$_{238}U^+$', ax=fdens)
# sns.distplot(nfinal['zp[i]']/mm, hist=True, color='r', label=r'$_{237}Np^+$', ax=fdens)
# fdens.set_title("Final Distribution of Particles",fontsize=16)
# fdens.set_xlabel('z [mm]',fontsize=14)
# fdens.set_ylabel('Number of Particles',fontsize=14)
# fdens.legend()
# plt.tight_layout()
# plt.savefig(filesave + 'distplots.png',dpi=400)
# plt.show()


#position plots
rmin = (1+.05)*copy['r'].min()
rmax = (1+.05)*copy['r'].max()
zmin = (1+.05)*copy['zp[i]'].min()
zmax = (1+.05)*copy['zp[i]'].max()

fig,rax = plt.subplots(figsize=(7,7))
rax.set_xlim(zmin/mm, zmax/mm)
rax.set_ylim(rmin/mm, rmax/mm)
rax.axhline(y = 0, lw = .5, c = 'k')
rax.axvline(x = 0, lw = .5, c = 'k')
rax.set_ylabel( "r [mm]",fontsize=14)
rax.set_xlabel("z [mm]", fontsize=14)
rax.set_title(' r vs z', fontsize = 18)
time_text = rax.text(.55,.95, "", transform = rax.transAxes)
time_template = "Time = %f [ms]"

#--Movie Creation
#Initialize empty scattery for trajectory and line plot for RMS visualization
scat = rax.scatter([], [], s = .1, c='r'  )       #Empty scatter plot for r values
com_scat = rax.scatter([], [], s = .1, c = 'k')      #Empty scatter plot for com values
nscat = rax.scatter([], [], s = .1, c='b' )       #Empty scatter plot for r values
ncom_scat = rax.scatter([], [], s = .1, c = 'k')      #Empty scatter plot for com values
plt.show()



#This function allows the use of Blit in animation
def init():
    scat.set_offsets([])
    com_scat.set_offsets([])
    nscat.set_offsets([])
    ncom_scat.set_offset([])
    time_text.set_text('')

    return scat, com_scat, nscat, ncom_scat, time_text

#Animation function. This will update the scatterplot coordinates for the ith frame.
def animate(i):

    print(i) #useful to see how fast the program is running and when it will finish.

    #Create a color switching routine


    #--Particle data points
    sample = copy[copy['Iter'] == i] #For particle only plotting
    #sample = copy[(copy['Iter'] >= 0) & (copy['Iter'] <= i)] #For ray Tracing
    rpoints = sample['r']/mm     #x-coordinates for all particles from 0 to ith iteration
    zpoints = sample['zp[i]']/mm     #z-coordinates for all particles from 0 to ith iteration

    nsample = ncopy[ncopy['Iter'] == i] #For particle only plotting
    #sample = copy[(copy['Iter'] >= 0) & (copy['Iter'] <= i)] #For ray Tracing
    nrpoints = nsample['r']/mm     #x-coordinates for all particles from 0 to ith iteration
    nzpoints = nsample['zp[i]']/mm     #z-coordinates for all particles from 0 to ith iteration


    #--Center of Mass Calculations
    com_rpoint = np.array([rcom_coords[i]])/mm   #x-COM-coordinates for particle plotting
    com_zpoint = np.array([zcom_coords[i]])/mm   #z-COM-coordinates for particle plotting
    #com_xpoint = np.array(xcom_coords[0:i])/mm #x-COM-coordinates for ray tracing
    com_zpoints = np.array(zcom_coords[0:i])/mm #z-COM-coordinates for ray tracing

    ncom_rpoint = np.array([nrcom_coords[i]])/mm   #x-COM-coordinates for particle plotting
    ncom_zpoint = np.array([nzcom_coords[i]])/mm   #z-COM-coordinates for particle plotting
    #com_xpoint = np.array(xcom_coords[0:i])/mm #x-COM-coordinates for ray tracing
    ncom_zpoints = np.array(nzcom_coords[0:i])/mm #z-COM-coordinates for ray tracing

    #-Time settings
    time_point = tracked_data['time'][i]/ms



    #Create arrays to feed into scatter plots using scat.set_offsets.
    #It is important to note tha set_offsets takes in (N,2) arrays. This is the reasoning for
    #adding np.newaxis the command. These arrays are then ((x,y), newaxis).
    scat_plot_points = np.hstack((zpoints[:, np.newaxis], rpoints[:, np.newaxis]))
    #com_scat_point = np.hstack((com_zpoint[:,np.newaxis], com_xpoint[:,np.newaxis])) #Particle plotting
    com_scat_point = np.hstack((com_zpoint[:, np.newaxis], com_rpoint[:, np.newaxis])) #ray tracing

    nscat_plot_points = np.hstack((nzpoints[:, np.newaxis], nrpoints[:, np.newaxis]))
    #com_scat_point = np.hstack((com_zpoint[:,np.newaxis], com_xpoint[:,np.newaxis])) #Particle plotting
    ncom_scat_point = np.hstack((ncom_zpoint[:, np.newaxis], ncom_rpoint[:, np.newaxis])) #ray tracing



    #Enter in new plot points.
    scat.set_offsets(scat_plot_points) #Scatter plot of x vs z
    com_scat.set_offsets(com_scat_point) # center of mass points

    nscat.set_offsets(nscat_plot_points) #Scatter plot of x vs z
    ncom_scat.set_offsets(ncom_scat_point) # center of mass points

    time_text.set_text(time_template % time_point)


    #Update plots
    return scat,com_scat, nscat, ncom_scat, time_text





#--Animating and saving
num_of_frames = len(copy['Iter'].unique()) #number of times animate() will be called
Writer = animation.writers['ffmpeg'] #for saving purposes
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800) #Some video settings.
#Ani is the actual animations. interval is interval between frames in microsecons. Blit is what speeds up the animation process
#by only plotting the changes rather then replotting each frame.
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames-1, interval=200, repeat=False, blit = True) #The actual animator


plt.tight_layout()
ani.save(file_save + 'rE{:.4f}run.mp4'.format(particle_energy/kV), writer=writer)
plt.show()

final_time = float(input("Enter time in ms:"))
final_iter = round(final_time*ms/time_step)
#--Create figures for plotting
fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(7,7))
uirz = axes[0][0]
nirz = axes[1][0]
idens = axes[0][1]
fdens = axes[1][1]

#-initial data
uinitial = copy[copy['Iter']==0]
ninitial = ncopy[ncopy['Iter']==0]

#-final data
ufinal = copy[copy['Iter'] == final_iter]
nfinal = ncopy[ncopy['Iter'] == final_iter]


#-initial distribution
uirz.scatter(x=uinitial['zp[i]']/mm, y=uinitial['r']/mm,\
             s = .1, c = 'b')
uirz.set_title(r"Initial Distribution $_{238}U^+$", fontsize=14)
uirz.set_xlabel("z [mm]",fontsize=12)
uirz.set_ylabel("r [mm]",fontsize=12)

nirz.scatter(x=ninitial['zp[i]']/mm, y=ninitial['r']/mm,\
             s = .1, c = 'r')
nirz.set_title(r"Initial Distribution $_{237}Np^+$", fontsize=14)
nirz.set_xlabel("z [mm]", fontsize=14)
nirz.set_ylabel("r [mm]", fontsize=12)

#--Initial Density Histogram
sns.distplot(uinitial['zp[i]']/mm, hist=True, color='b', label=r'$_{238}U^+$', \
             hist_kws=dict(edgecolor="k", linewidth=.5), ax=idens)
sns.distplot(ninitial['zp[i]']/mm, hist=True, color='r', label=r'$_{237}Np^+$',\
             hist_kws=dict(edgecolor="k", linewidth=.5), ax=idens)
idens.set_title("Initial Distribution of Particles",fontsize=14)
idens.set_xlabel('z [mm]', fontsize=12)
idens.set_ylabel('Particle Probability Density',fontsize=12)
idens.legend()


#--Final distributions
sns.distplot(ufinal['zp[i]']/mm, hist=True, color='b', label=r'$_{238}U^+$',\
             hist_kws=dict(edgecolor="k", linewidth=.5), ax=fdens)
sns.distplot(nfinal['zp[i]']/mm, hist=True, color='r', label=r'$_{237}Np^+$',\
             hist_kws=dict(edgecolor="k", linewidth=.5), ax=fdens)
fdens.set_title("Final Distribution of Particles",fontsize=14)
fdens.set_xlabel('z [mm]',fontsize=12)
fdens.set_ylabel('Particle Probability Density',fontsize=12)
fdens.legend()
plt.tight_layout()
plt.savefig(file_save + 'distplots.png',dpi=400)
plt.show()
