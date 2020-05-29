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

file_save = '/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/Animations/Real_Dist/'

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

data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/ntrajectoryfile.txt', names = column_names)
tracked_data = pd.read_csv('/Users/nickvalverde/Dropbox/Research/ORISS/ntracked_particle.txt', names = tracked_columns)

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

stdr_data = []
stdz_data = []

for i in range(len(copy['Iter'].unique())):
    #Get position data for x and z for ith iteration
    r_sample = copy[copy['Iter'] == i]['r']
    z_sample = copy[copy['Iter'] == i]['zp[i]']

    #Calculate standard deviations and append them to lists
    stdr_data.append(r_sample.std())
    stdz_data.append(z_sample.std())


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

#--Create figures for plotting

#Set min and max values for plots
#std plots
stdr_min = 0
stdr_max = (1+.05)*max(stdr_data)
stdz_min = 0
stdz_max = (1+.05)*max(stdz_data)
#position plots
rmin = (1+.05)*copy['r'].min()
rmax = (1+.05)*copy['r'].max()
zmin = (1+.05)*copy['zp[i]'].min()
zmax = (1+.05)*copy['zp[i]'].max()
#Will do x-statstics first
fig = plt.figure(figsize = (9,9))
rax = plt.subplot2grid((3,3), (0,0), colspan = 2)
stdrax = plt.subplot2grid((3,3), (1,0), colspan = 2)
stdrtax = plt.subplot2grid((3,3), (2,0), colspan = 3)

data_table = plt.subplot2grid((3,3),(0,2), rowspan = 2) #Display data during animation
data_table.set_title("Data Table")
data_table = plt.subplot2grid((3,3),(0,2), rowspan = 2) #Display data during animation
data_table.set_title("Data Table")
data_table.text(.05, .95, 'Position: Gaussian', fontsize=10)
data_table.text(.05, .90, 'Velocity: Gaussian', fontsize=10)
data_table.text(.05, .85, r'Particle Energy = {:.4f} $\pm$ {}% [kV]'\
                .format(particle_energy/1000, 2), fontsize=10)
data_table.text(.05, .80, 'Np = {}'.format(Np), fontsize=10)
data_table.text(.05, .75, r"Tperp, Tz = {:.2f} [K], {:.2f} [K]"\
                .format(wp.jperev*tperp/wp.boltzmann, wp.jperev*tz/wp.boltzmann),\
                 fontsize=10)
data_table.text(.05, .70, r'$\sigma_x$, $\sigma_y$, $\sigma_z$ = {:.2f}, {:.2f}, {:.2f} [mm]'\
                .format(sigmax/mm, sigmay/mm, sigmaz/mm), fontsize=10)


data_table.spines['right'].set_visible(False)         #Turn off right spine
data_table.yaxis.set_major_locator(plt.NullLocator()) #Turn off tick marks in y
data_table.xaxis.set_major_locator(plt.NullLocator()) #Turn off tick marks in x


rax.set_xlim(zmin/mm, zmax/mm)
rax.set_ylim(rmin/mm, rmax/mm)
stdrax.set_ylim(stdr_min/mm, stdr_max/mm)
stdrax.set_xlim(zmin/mm, zmax/mm)
stdrtax.set_xlim(0, tracked_data['time'].max()/ms)
stdrtax.set_ylim(stdr_min/mm, stdr_max/mm)

rax.axhline(y = 0, lw = .5, c = 'k')
rax.axvline(x = 0, lw = .5, c = 'k')
stdrax.axvline(x = 0, lw = .5, c = 'k')

stdrax.set_xlabel("z [mm]",fontsize=14)
rax.set_ylabel( "r [mm]",fontsize=14)
stdrax.set_ylabel(r'$\sigma_{r}$[mm]', fontsize = 14)
stdrtax.set_xlabel(r'Time [ms]', fontsize = 14)
stdrtax.set_ylabel(r'$\sigma_{r}$[mm]', fontsize = 14)



rax.set_title(' r vs z', fontsize = 18)
stdrax.set_title(r'$\sigma_{r}$ vs z', fontsize = 18)
stdrtax.set_title(r'$\sigma_{r}$ vs Time', fontsize = 18)

plt.tight_layout()

#--Movie Creation

#Initialize empty scattery for trajectory and line plot for RMS visualization
scat = rax.scatter([], [], s = .1, )       #Empty scatter plot for r values
com_scat = rax.scatter([], [], s = .2, c = 'r')      #Empty scatter plot for com values
line, = stdrax.plot([], [], lw = 1, c = 'k')      #Empty line plot for stdx vs z
std_time_line, = stdrtax.plot([], [], lw = 1, c = 'k')  #Empty line plot for stdx vs time

#Create text objects to fill data table
lstpart_text = data_table.text(0.05, 0.65, "", transform=data_table.transAxes)
time_text = data_table.text(0.05, 0.60, "", transform=data_table.transAxes)
#Create templates for data
lstpart_template = 'Particles Lost = {:.2f}% ({}/{})'
time_template = 'Time = %f [ms]'


#This function allows the use of Blit in animation
def init():
    scat.set_offsets([])
    com_scat.set_offsets([])
    line.set_data([], [])
    std_time_line.set_data([],[])
    lstpart_text.set_text('')
    time_text.set_text('')

    return scat, com_scat, line, std_time_line, lstpart_text, time_text

#Animation function. This will update the scatterplot coordinates for the ith frame.
sign_list = np.sign(tracked_data['uzp[i]']) #A sign list used later to switch colors of particle rays
sign_list = np.append(sign_list, 1) #append dummy value to end to prevent animate function error.
def animate(i):
    #Color switching wont work like this for ray tracing since the entire trajectory is replotted.
    #if sign_list[i] != sign_list[i+1]:
        #scat.set_color('m')
   # else:
        #scat.set_color('k')
    print(i) #useful to see how fast the program is running and when it will finish.

    #Create a color switching routine


    #--Particle data points
    sample = copy[copy['Iter'] == i] #For particle only plotting
    #sample = copy[(copy['Iter'] >= 0) & (copy['Iter'] <= i)] #For ray Tracing
    rpoints = sample['r']/mm     #x-coordinates for all particles from 0 to ith iteration
    zpoints = sample['zp[i]']/mm     #z-coordinates for all particles from 0 to ith iteration

    #--Center of Mass Calculations
    com_rpoint = np.array([rcom_coords[i]])/mm   #x-COM-coordinates for particle plotting
    com_zpoint = np.array([zcom_coords[i]])/mm   #z-COM-coordinates for particle plotting
    #com_xpoint = np.array(xcom_coords[0:i])/mm #x-COM-coordinates for ray tracing
    com_zpoints = np.array(zcom_coords[0:i])/mm #z-COM-coordinates for ray tracing

    #--Stdx_points for line plot
    stdr_zpoints = np.array(zcom_coords[0:i])/mm
    stdr_rpoints = np.array(stdr_data[0:i])/mm #x_rms from 0 to ith iteration

    #Lost Particle value
    Nplost = tracked_data['Nplost'][i]
    Nplost_point = Nplost/Np*100

    #time coordinates
    time_point = tracked_data['time'][i]/ms
    time_points = tracked_data['time'][0:i]/ms




    #Create arrays to feed into scatter plots using scat.set_offsets.
    #It is important to note tha set_offsets takes in (N,2) arrays. This is the reasoning for
    #adding np.newaxis the command. These arrays are then ((x,y), newaxis).
    scat_plot_points = np.hstack((zpoints[:, np.newaxis], rpoints[:, np.newaxis]))
    #com_scat_point = np.hstack((com_zpoint[:,np.newaxis], com_xpoint[:,np.newaxis])) #Particle plotting
    com_scat_point = np.hstack((com_zpoint[:, np.newaxis], com_rpoint[:, np.newaxis])) #ray tracing


    #Enter in new plot points.
    scat.set_offsets(scat_plot_points) #Scatter plot of x vs z
    com_scat.set_offsets(com_scat_point) # center of mass points

    line.set_data(com_zpoints, stdr_rpoints) #std_r line plot
    std_time_line.set_data(time_points, stdr_rpoints) #std_r time plot

    lstpart_text.set_text(lstpart_template.format(Nplost_point, Nplost, Np))
    time_text.set_text(time_template % time_point)

    #Update plots
    return scat,com_scat, line, std_time_line, lstpart_text, time_text





#--Animating and saving
num_of_frames = len(copy['Iter'].unique()) #number of times animate() will be called
Writer = animation.writers['ffmpeg'] #for saving purposes
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800) #Some video settings.
#Ani is the actual animations. interval is interval between frames in microsecons. Blit is what speeds up the animation process
#by only plotting the changes rather then replotting each frame.
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames-1, interval=200, repeat=False, blit = True) #The actual animator
plt.show()
ani.save(file_save + 'rE{:.4f}run.mp4'.format(particle_energy/kV), writer=writer)
