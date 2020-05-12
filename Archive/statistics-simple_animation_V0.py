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

file_save = '/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/Animations/Simple_Dist/On_E'

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

stdx_data = []
stdz_data = []

for i in range(len(copy['Iter'].unique())):
    #Get position data for x and z for ith iteration
    x_sample = copy[copy['Iter'] == i]['xp[i]']
    z_sample = copy[copy['Iter'] == i]['zp[i]']

    #Calculate standard deviations and append them to lists
    stdx_data.append(x_sample.std())
    stdz_data.append(z_sample.std())


#--Create Center of Mass data
zcom_coords = []
xcom_coords = []
for i in range(len(copy['Iter'].unique())):
    #Get position data for r and z for ith iteration
    x_sample = copy[copy['Iter'] == i]['xp[i]']
    z_sample = copy[copy['Iter'] == i]['zp[i]']

    #Calculate standard deviations and append them to lists
    zcom_coords.append(z_sample.mean())
    xcom_coords.append(x_sample.mean())

zcom_coords = zcom_coords
xcom_coords = xcom_coords




#--Create figures for plotting

#Set min and max values for plots
#std plots
stdx_min = min(stdx_data)
stdx_max = max(stdx_data)
stdz_min = min(stdz_data)
stdz_max = max(stdz_data)
#position plots
xmin = copy['xp[i]'].min()
xmax = copy['xp[i]'].max()
zmin = copy['zp[i]'].min()
zmax = copy['zp[i]'].max()

#Will do x-statstics first
fig = plt.figure(figsize = (9,9))
xax = plt.subplot2grid((3,3), (0,0), colspan = 2)
stdxax = plt.subplot2grid((3,3), (1,0), colspan = 2)
stdxtax = plt.subplot2grid((3,3), (2,0), colspan = 3)

data_table = plt.subplot2grid((3,3),(0,2), rowspan = 2) #Display data during animation
data_table.set_title("Data Table")
data_table.text(.05, .95, 'Position: Max 3mm', fontsize = 10)
data_table.text(.05, .90, 'Velocity: No Distribution', fontsize = 10)
data_table.text(.05, .85, 'Particle Energy = {:.2f} [kV]'.format(particle_energy/1000), fontsize = 10)

data_table.yaxis.set_major_locator(plt.NullLocator()) #Turn off tick marks in y
data_table.xaxis.set_major_locator(plt.NullLocator()) #Turn off tick marks in x


xax.set_xlim(zmin/mm, zmax/mm)
xax.set_ylim(xmin/mm, xmax/mm)
stdxax.set_ylim(stdx_min/mm, stdx_max/mm)
stdxax.set_xlim(zmin/mm, zmax/mm)
stdxtax.set_xlim(0, tracked_data['time'].max()/ms)
stdxtax.set_ylim(stdx_min/mm, stdx_max/mm)

xax.axhline(y = 0, lw = .5, c = 'k')
xax.axvline(x = 0, lw = .5, c = 'k')
stdxax.axvline(x = 0, lw = .5, c = 'k')

stdxax.set_xlabel("z [mm]",fontsize=14)
xax.set_ylabel( "x [mm]",fontsize=14)
stdxax.set_ylabel(r'$\sigma_{x}$[mm]', fontsize = 14)
stdxtax.set_xlabel(r'Time [ms]', fontsize = 14)
stdxtax.set_ylabel(r'$\sigma_{x}$[mm]', fontsize = 14)



xax.set_title(' x vs z', fontsize = 18)
stdxax.set_title(r'$\sigma_{x}$ vs z', fontsize = 18)
stdxtax.set_title(r'$\sigma_{x}$ vs Time', fontsize = 18)

plt.tight_layout()

#--Movie Creation

#Initialize empty scattery for trajectory and line plot for RMS visualization
scat = xax.scatter([], [], s = .1, c = 'k')       #Empty scatter plot for r values
com_scat = xax.scatter([], [], s=2, c = 'r')      #Empty scatter plot for com values
line, = stdxax.plot([], [], lw = 1, c = 'k')      #Empty line plot for stdx vs z
std_time_line, = stdxtax.plot([], [], lw = 1, c = 'k')  #Empty line plot for stdx vs time

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

    #--Particle data points
    #coords = copy[copy['Iter'] == i] #Creates a data frame for all other particles for the ith iteration
    xpoints = coords['xp[i]'][0:i]/mm     #x-coordinates for all particles from 0 to ith iteration
    zpoints = coords['zp[i]'][0:i]/mm     #z-coordinates for all particles from 0 to ith iteration

    #--Center of Mass Calculations
    com_xpoint = np.array(xcom_coords[0:i])/mm #x-COM-coordinates from 0 to ith iteration
    com_zpoint = np.array(zcom_coords[0:i])/mm #z-COM-coordinates from 0 to ither iteration

    #Line plot for stdx vs z
    stdx_zpoints = np.array(std_data[0:i])/mm #z_rms from 0 to ith iteration
    stdx_points = np.array(stdx_data[0:i])/mm #z_rms from 0 to ith iteration

    #Lost Particle value
    Nplost = tracked_data['Nplost'][i]
    Nplost_point = Nplost/Np*100

    #time coordinates
    time_point = tracked_data['time'][i]/ms
    time_points = tracked_data['time'][0:i]/ms




    #Create arrays to feed into scatter plots using scat.set_offsets.
    #It is important to note tha set_offsets takes in (N,2) arrays. This is the reasoning for
    #adding np.newaxis the command. These arrays are then ((x,y), newaxis).
    scat_plot_points = np.hstack((zpoints[:, np.newaxis], xpoints[:, np.newaxis]))
    com_scat_point = np.hstack((com_zpoint[:, np.newaxis], com_xpoint[:, np.newaxis]))


    #Enter in new plot points.
    scat.set_offsets(scat_plot_points) #Scatter plot of r vs z
    com_scat.set_offsets(com_scat_point) # center of mass points

    line.set_data(stdx_zpoints, stdx_points) #std_r line plot
    std_time_line.set_data(time_points, stdx_points) #std_r time plot

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
ani.save(file_save + 'run.mp4', writer=writer)
