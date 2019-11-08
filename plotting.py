#Script will first clean trajctoryfile so that it can be read into a pandas DataFrame.
#The script will then be able to plot using matplotlib or pandas plotting capabilities.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

#Open Trajectory file and create a new file named cleaned that will be a comma seperated file.
trajectory = open('/Users/nickvalverde/Research/ORISS/trajectoryfile.txt')
cleaned = open('/Users/nickvalverde/Research/ORISS/cleaned', 'w')

#Replaces the spaces in the trajectory file with ','s using a regex. The command
# re.sub(r' +', r',', line) first matches any pattern with a space, replacs it with a comma, and does this for each
#line in the file.
for line in trajectory:
    cleaned.write(re.sub(r' +', r',', line))
#Close files.
trajectory.close()
cleaned.close()

#Read in the data into a dataframe called 'data'.
data = pd.read_csv('/Users/nickvalverde/Research/ORISS/cleaned')
data.drop(data.columns[-1], axis=1, inplace=True) #Drops last column which is Nan values. This is due to the formatting of the oriss file
                                                  #where each line is given by a line break '\n'


#Plotting Particles
##z-trajectory
fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (7,7), sharex = True)
for i in range(len(data['Particle'].unique())):
    df_zp = data[data['Particle'] == i] #sets up a new dataframe for each particle
    ax[0].plot(df_zp['Iter'], df_zp['zp[i]'])

    df_xp = data[data['Particle'] == i]
    ax[1].plot(df_xp['Iter'], df_xp['xp[i]'])


plt.tight_layout()
ax[1].set_xlabel('Iteration')
ax[0].set_ylabel('zp')
ax[1].set_ylabel('xp')


plt.savefig('trajectory.png', dpi=300)














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
