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

file_save = '/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/Animations/Simple_Dist/'

#Some useful defintions.
mm = 1e-3
ms = 1e-3
time = 1e-7
kV = 1000.

#--Read in variables and assign values from parameters.txt.
#Variables that are in the parameters.txt file
variable_list = ["particle_energy", "mass", "time_step","zcentr8", "zcentr7", \
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

fig,ax = plt.subplots(nrows = 2, ncols=1,figsize = (7,7))
ax[0].scatter(copy['time']/ms, copy['zp[i]']/mm, s=.1)
ax[0].axhline(y=0,lw=.1,c='k')
ax[0].axvline(x=0,lw=.1,c='k')
ax[0].set_xlabel('z[mm]')
ax[0].set_ylabel('x[mm]')

ax[1].scatter(zcom_coords[:], stdz_data[:], s=.1)
ax[1].axhline(y=0,lw=.1,c='k')
ax[1].axvline(x=0,lw=.1,c='k')
ax[1].set_ylabel(r'$sigma_z$ [mm]')
ax[1].set_xlabel('z[mm]')

plt.tight_layout()
plt.savefig(file_save+'E{:.6f}longitudinal.png'.format(particle_energy/kV),dpi=300)
plt.show()
raise Exception()


#--Create figures for plotting

#Set min and max values for plots
#std plots
stdx_min = 0
stdx_max = (1+.05)*max(stdx_data)
stdz_min = 0
stdz_max = (1+.05)*max(stdz_data)

#CFind limites for plots
xmin = (1+.05)*copy['xp[i]'].min()
xmax = (1+.05)*copy['xp[i]'].max()
zmin = (1+.05)*copy['zp[i]'].min()
zmax = (1+.05)*copy['zp[i]'].max()
tmin = 0.
tmax = tracked_data['time'].max()

#Create figures
fig = plt.figure(figsize = (9,9))
zax = plt.subplot2grid((3,3), (0,0), colspan = 2)
stdzax = plt.subplot2grid((3,3), (1,0), colspan = 2)
stdztax = plt.subplot2grid((3,3), (2,0), colspan = 3)


#Create data Table
data_table = plt.subplot2grid((3,3),(0,2), rowspan = 2) #Display data during animation
data_table.set_title("Data Table")
data_table.text(.05, .95, 'Simple Tuning Distribution', fontsize=10)
data_table.text(.05, .90, 'Velocity: No Distribution', fontsize=10)
data_table.text(.05, .85, r'Particle Energy = {:.4f} $\pm$ {}% [kV]'.format(particle_energy/1000, 2), fontsize=10)
data_table.text(.05, .80, 'Initial Bunch Length: 0mm', fontsize=10)


data_table.spines['right'].set_visible(False)         #Turn off right spine
data_table.yaxis.set_major_locator(plt.NullLocator()) #Turn off tick marks in y
data_table.xaxis.set_major_locator(plt.NullLocator()) #Turn off tick marks in x


#figure limit setting
zax.set_xlim(tmin/ms, tmax/ms)
zax.set_ylim(zmin/mm, zmax/mm)

stdzax.set_xlim(zmin/mm, zmax/mm)
stdzax.set_ylim(stdz_min/mm, stdz_max/mm)

stdztax.set_xlim(tmin/ms, tmax/ms)
stdztax.set_ylim(stdz_min/mm, stdz_max/mm)


#zax.axhline(y = 0, lw = .5, c = 'k')
zax.axhline(y = 0, lw = .5, c = 'k')
stdzax.axvline(x = 0, lw = .5, c = 'k')


#axis labeling
zax.set_xlabel('Time [ms]',fontsize=14)
zax.set_ylabel( "z [mm]",fontsize=14)

stdzax.set_xlabel("z [mm]",fontsize=14)
stdzax.set_ylabel(r'$\sigma_{z}$[mm]', fontsize = 14)

stdztax.set_xlabel(r'Time [ms]', fontsize = 14)
stdztax.set_ylabel(r'$\sigma_{z}$[mm]', fontsize = 14)


#titling
zax.set_title('z vs Time', fontsize = 18)
stdzax.set_title(r'$\sigma_{z}$ vs z', fontsize = 18)
stdztax.set_title(r'$\sigma_{z}$ vs Time', fontsize = 18)

plt.tight_layout()
plt.show()


#--Movie Creation

#Initialize empty scattery for trajectory and line plot for RMS visualization
scat = zax.scatter([], [], s = .1, c = 'b' )       #Empty scatter plot for r values
com_scat = zax.scatter([], [], s = .1, c = 'b')      #Empty scatter plot for com values
line, = stdzax.plot([], [], lw = 1, c = 'k')      #Empty line plot for stdz vs z
std_time_line, = stdztax.plot([], [], lw = 1, c = 'k')  #Empty line plot for stdz vs time

#Create text objects to fill data table
bunchlength_text = data_table.text(0.05, 0.75, "", transform=data_table.transAxes)
lstpart_text = data_table.text(0.05, 0.70, "", transform=data_table.transAxes)
time_text = data_table.text(0.05, 0.65, "", transform=data_table.transAxes)
#Create templates for data
bunchlength_template = r'$\Delta$(Bunch Length) = {:.4f} [mm]'
lstpart_template = 'Particles Lost = {:.2f}% ({}/{})'
time_template = 'Time = %f [ms]'



#This function allows the use of Blit in animation
def init():
    scat.set_offsets([])
    com_scat.set_offsets([])
    line.set_data([], [])
    std_time_line.set_data([],[])
    bunchlength_text.set_text('')
    lstpart_text.set_text('')
    time_text.set_text('')

    return scat, com_scat, line, std_time_line, bunchlength_template, lstpart_text, time_text

#Animation function. This will update the scatterplot coordinates for the ith frame.
def animate(i):
    #Color switching wont work like this for ray tracing since the entire trajectory is replotted.
    #if sign_list[i] != sign_list[i+1]:
        #scat.set_color('m')
   # else:
        #scat.set_color('k')
    print(i) #useful to see how fast the program is running and when it will finish.

    #Create a color switching routine


    #--Particle data points
    #sample = copy[copy['Iter'] == i] #For particle only plotting
    sample = copy[(copy['Iter'] >= 0) & (copy['Iter'] <= i)] #For ray Tracing
    #xpoints = sample['xp[i]']/mm     #x-coordinates for all particles from 0 to ith iteration
    zpoints = sample['zp[i]']/mm     #z-coordinates for all particles from 0 to ith iteration
    time = sample['time']/ms

    #--Center of Mass Calculations
    #com_xpoint = np.array([xcom_coords[i]])/mm   #x-COM-coordinates for particle plotting
    #com_zpoint = np.array([zcom_coords[i]])/mm   #z-COM-coordinates for particle plotting
    com_xpoint = np.array(xcom_coords[0:i])/mm #x-COM-coordinates for ray tracing
    com_zpoint = np.array(zcom_coords[0:i])/mm #z-COM-coordinates for ray tracing

    #--stdz_points for line plot
    stdz_zpoints = np.array(stdz_data[0:i])/mm
    stdx_xpoints = np.array(stdx_data[0:i])/mm #x_rms from 0 to ith iteration

    #Update bunch Length

    bunchlength = abs(sample[sample['Iter'] == i]['zp[i]'].max()
                      - sample[sample['Iter'] == i]['zp[i]'].min())/mm

    #Lost Particle value
    Nplost = tracked_data['Nplost'][i]
    Nplost_point = Nplost/Np*100

    #time coordinates
    time_point = tracked_data['time'][i]/ms
    time_points = tracked_data['time'][0:i]/ms #ray tracing




    #Create arrays to feed into scatter plots using scat.set_offsets.
    #It is important to note tha set_offsets takes in (N,2) arrays. This is the reasoning for
    #adding np.newaxis the command. These arrays are then ((x,y), newaxis).
    scat_plot_points = np.hstack((time[:, np.newaxis], zpoints[:, np.newaxis]))
    #com_scat_point = np.hstack((time_point[:,np.newaxis], com_zpoints[:,np.newaxis])) #Particle plotting
    com_scat_point = np.hstack((time_points[:, np.newaxis], com_zpoint[:, np.newaxis])) #ray tracing


    #Enter in new plot points.
    scat.set_offsets(scat_plot_points) #Scatter plot of x vs z
    com_scat.set_offsets(com_scat_point) # center of mass points

    line.set_data(com_zpoint, stdz_zpoints) #std_z line plot
    std_time_line.set_data(time_points, stdz_zpoints) #std_z time plot


    bunchlength_text.set_text(bunchlength_template.format(bunchlength))
    lstpart_text.set_text(lstpart_template.format(Nplost_point, Nplost, Np))
    time_text.set_text(time_template % time_point)

    #Update plots
    return scat, com_scat, line, std_time_line, bunchlength_text, lstpart_text, time_text





#--Animating and saving
num_of_frames = len(copy['Iter'].unique()) #number of times animate() will be called
Writer = animation.writers['ffmpeg'] #for saving purposes
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800) #Some video settings.
#Ani is the actual animations. interval is interval between frames in microsecons. Blit is what speeds up the animation process
#by only plotting the changes rather then replotting each frame.
ani = animation.FuncAnimation(fig, animate, frames=num_of_frames-1, interval=200, repeat=False, blit = True) #The actual animator
plt.show()
ani.save(file_save + 'E{:.4f}run.mp4'.format(particle_energy/kV), writer=writer)
