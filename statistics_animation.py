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

#stdr_data = np.array(stdr_data)
#stdz_data = np.array(stdz_data)

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

zcom_coords = np.array(zcom_coords)
rcom_coords = np.array(rcom_coords)




#--Create figures for plotting

#Set min and max values for plots
#std plots
stdr_min = min(stdr_data)
stdr_max = max(stdr_data)
stdz_min = min(stdz_data)
stdz_max = max(stdz_data)
#position plots
rmin = copy['r'].min()
rmax = copy['r'].max()
zmin = copy['zp[i]'].min()
zmax = copy['zp[i]'].max()

#Will do x-statstics first
fig = plt.figure(figsize = (9,9))
rax = plt.subplot2grid((3,3), (0,0), colspan = 2)
stdrax = plt.subplot2grid((3,3), (1,0), colspan = 2)
stdrtax = plt.subplot2grid((3,3), (2,0), colspan = 3)

data_table = plt.subplot2grid((3,3),(0,2), rowspan = 2) #Display data during animation
data_table.set_title("Data Table")
data_table.text(.05, .95, 'Particle Energy = {:.2f} [kV]'.format(particle_energy/1000))
data_table.text(.05, .90, 'Gaussian Position Distribution')
data_table.text(.05, .85, 'Gaussian Velocity Distribution')

data_table.yaxis.set_major_locator(plt.NullLocator()) #Turn off tick marks in y
data_table.xaxis.set_major_locator(plt.NullLocator()) #Turn off tick marks in x
#-------------------------------------------------------------------------------
# #--Old routine didn't alow specified axis sharing
# fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize=(7,7), sharex = True)
# xax = ax[0]
# stdxax = ax[1]
#-------------------------------------------------------------------------------
rax.set_xlim(zmin/mm, zmax/mm)
rax.set_ylim(rmin/mm, rmax/mm)
stdrax.set_ylim(stdr_min/mm, stdr_max/mm)
stdrax.set_xlim(zmin/mm, zmax/mm)
stdrtax.set_xlim(0, tracked_data['time'].max()/ms)
stdrtax.set_ylim(stdr_min/mm, stdr_max/mm)

rax.axhline(y = 25, lw = .5, c = 'm', label = 'Aperture Radius')
rax.axvline(x = 0, lw = .5, c = 'k')
rax.legend(loc = 'upper right')
stdrax.axvline(x = 0, lw = .5, c = 'k')

stdrax.set_xlabel("z [mm]",fontsize=14)
rax.set_ylabel( "r [mm]",fontsize=14)
stdrax.set_ylabel(r'$\sigma_{r}$[mm]', fontsize = 14)
stdrtax.set_xlabel(r'Time [ms]', fontsize = 14)
stdrtax.set_ylabel(r'$\sigma_{r}$[mm]', fontsize = 14)



rax.set_title(' r vs Longitudinal Position', fontsize = 18)
stdrax.set_title(r'$\sigma_{r}$ vs Longitudinal Position', fontsize = 18)
stdrtax.set_title(r'$\sigma_{r}$ vs Time', fontsize = 18)

plt.tight_layout()

#--Movie Creation

#Initialize empty scattery for trajectory and line plot for RMS visualization
scat = rax.scatter([], [], s = .1, c = 'k')       #Empty scatter plot for r values
com_scat = rax.scatter([], [], s=2, c = 'r')      #Empty scatter plot for com values
line, = stdrax.plot([], [], lw = 1, c = 'k')      #Empty line plot for stdr vs z
std_time_line, = stdrtax.plot([], [], lw = 1, c = 'k')  #Empty line plot for stdr vs time

#Create text objects to fill data table 
lstpart_text = data_table.text(0.05, 0.80, "", transform=data_table.transAxes)
time_text = data_table.text(0.05, 0.75, "", transform=data_table.transAxes)
#Create templates for data
lstpart_template = 'Particles Lost = {:.2f}%'
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
def animate(i):

    print(i) #useful to see how fast the program is running and when it will finish.

    #particle data points
    coords = copy[copy['Iter'] == i] #Creates a data frame for all other particles for the ith iteration
    rpoints = coords['r']/mm         #x-coordinates for all other particles for ith iteration
    zpoints = coords['zp[i]']/mm     #z-coordinates on tracker particle for ith iteration

    #Center of Mass Coordinates
    com_rpoint = rcom_coords[i]/mm #r-COM-coordinate
    com_zpoint = zcom_coords[i]/mm #z-COM-coordinate

    #Line plot for stdr vs z
    stdr_zpoints = tracked_data['zp[i]'][0:i]/mm #z-coordinate from 0 to ith iteration
    stdr_points = np.array(stdr_data[0:i])/mm #std in x from 0 to ith iteration
    
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
    com_scat_point = np.hstack(((com_zpoint,),(com_rpoint,)))

    
    #Enter in new plot points.
    scat.set_offsets(scat_plot_points) #Scatter plot of r vs z
    com_scat.set_offsets(com_scat_point) # center of mass points
    
    line.set_data(stdr_zpoints, stdr_points) #std_r line plot
    std_time_line.set_data(time_points, stdr_points) #std_r time plot
    
    lstpart_text.set_text(lstpart_template.format(Nplost_point))
    time_text.set_text(time_template % time_point)
    
    #Update plots
    return scat,com_scat, line, std_time_line, lstpart_text, time_text





#--Animating and saving
num_of_frames = len(copy['Iter'].unique()) #number of times animate() will be called
Writer = animation.writers['ffmpeg'] #for saving purposes
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800) #Some video settings.
#Ani is the actual animations. interval is interval between frames in microsecons. Blit is what speeds up the animation process
#by only plotting the changes rather then replotting each frame.
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames-1, interval=200, repeat=True, blit = True) #The actual animator
plt.show()
ani.save('/Users/nickvalverde/Dropbox/Research/ORISS/Working/Animated_Plots/toy_animated_moments.mp4', writer=writer)
