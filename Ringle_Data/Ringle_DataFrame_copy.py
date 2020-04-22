#/Users/nickvalverde/Research/ORISS/Particle_Class.py#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:20:33 2019

@author: nickvalverde
"""

from warp import *
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('darkgrid')
#import sys


#stdoutOrigin=sys.stdout #writes output in terminal to a file.
#sys.stdout = open("ringle_data.txt", "w")


potassium_beam = Species(type=Potassium,charge_state=+0,name="Beam species",weight=0) #weight = 0 no spacecharge, 1 = spacecharge. Both go through Poisson sovler.
mass = potassium_beam.mass
mm = 1e-3
kV = 1e3

print("\n")
print("Potassium Beam (A = 39) used")
print("\n")
#--Ion Data
column_list = ["t(s)", "x(m)", "vx(m/s)", "y(m)", "vy(m/s)", "z(m)", "vz(m/s)"]
ion_data = pd.read_excel('/Users/nickvalverde/Dropbox/Research/ORISS/Ringle_Data/ion_data.xlsx', names = column_list)


def preparation(dataframe):
    #fix z-coordinate
    copy_data = dataframe.copy()
    min_time = copy_data['t(s)'].min()
    copy_data['z(m)'] = copy_data['z(m)']*(copy_data['t(s)'] - min_time)
    #Create deviation columns dvx, dvy, dvz
    copy_data['dvx'] = copy_data['vx(m/s)'] - copy_data['vx(m/s)'].mean()
    copy_data['dvy'] = copy_data['vy(m/s)'] - copy_data['vy(m/s)'].mean()
    copy_data['dvz'] = copy_data['vz(m/s)'] - copy_data['vz(m/s)'].mean()
    #Create Kinetic energy columns Kx, Ky, Kz
    copy_data['Kx(eV)'] = 0.5*mass*(copy_data['vx(m/s)']**2)/jperev
    copy_data['Ky(eV)'] = 0.5*mass*(copy_data['vy(m/s)']**2)/jperev
    copy_data['Kz(eV)'] = 0.5*mass*(copy_data['vz(m/s)']**2)/jperev
    #Create Temperature Columns
    copy_data['Tx(K)'] = mass*copy_data['dvx']**2/boltzmann
    copy_data['Ty(K)'] = mass*copy_data['dvy']**2/boltzmann
    copy_data['Tz(K)'] = mass*copy_data['dvz']**2/boltzmann
    return copy_data

def position_averages(dataframe):

    average_x = dataframe['x(m)'].mean()
    average_y = dataframe['y(m)'].mean()
    average_z = dataframe['z(m)'].mean()

    return_string = "x-average = {}[mm]".format(average_x/mm) + "\n" + \
                    "y-average = {}[mm]".format(average_y/mm) + "\n" + \
                    "z-average = {}[mm]".format(average_z/mm) + "\n"

    print(return_string)
    return print("Success")

def velocity_averages(dataframe):
    average_vx = dataframe['vx(m/s)'].mean()
    average_vy = dataframe['vy(m/s)'].mean()
    average_vz = dataframe['vz(m/s)'].mean()

    return_string = "vx-average = {}[m/s]".format(average_vx) + "\n" + \
                    "vy-average = {}[m/s]".format(average_vy) + "\n" + \
                    "vz-average = {}[m/s]".format(average_vz) + "\n"

    print(return_string)
    return("Success")

def position_stdev(dataframe):
    #Create Deviations
    dx = dataframe['x(m)'] - dataframe['x(m)'].mean()
    dy = dataframe['y(m)'] - dataframe['y(m)'].mean()
    dz = dataframe['z(m)'] - dataframe['z(m)'].mean()

    sigma_x = np.sqrt(  (dx**2).mean()   )
    sigma_y = np.sqrt(  (dy**2).mean()   )
    sigma_z = np.sqrt(  (dz**2).mean()   )

    return_string = "sigma_x = {}[mm]".format(sigma_x/mm) + "\n" + \
                    "sigma_y = {}[mm]".format(sigma_y/mm) + "\n" + \
                    "sigma_z = {}[mm]".format(sigma_z/mm) + "\n"

    print(return_string)
    return print("Success")

def velocity_stdev(dataframe):
    #Create Deviations
    dvx = dataframe['vx(m/s)'] - dataframe['vx(m/s)'].mean()
    dvy = dataframe['vy(m/s)'] - dataframe['vy(m/s)'].mean()
    dvz = dataframe['vz(m/s)'] - dataframe['vz(m/s)'].mean()

    N = len(dvx) #Number of particles

    sigma_vx = np.sqrt(  sum(dvx**2)/N  )
    sigma_vy = np.sqrt(  sum(dvy**2)/N  )
    sigma_vz = np.sqrt(  sum(dvz**2)/N  )

    return_string = "sigma_vx = {}[m/s]".format(sigma_vx) + "\n" + \
                    "sigma_vy = {}[m/s]".format(sigma_vy) + "\n" + \
                    "sigma_vz = {}[m/s]".format(sigma_vz) + "\n"

    print(return_string)
    return print("Success")

def longitudinal_energy(dataframe):
    #Velocity Deviations
    dvx = dataframe['vx(m/s)'] - dataframe['vx(m/s)'].mean()
    dvy = dataframe['vy(m/s)'] - dataframe['vy(m/s)'].mean()
    dvz = dataframe['vz(m/s)'] - dataframe['vz(m/s)'].mean()

    Ez = (0.5*mass*dataframe['vz(m/s)']**2/jperev).mean() #[eV]

    temp_x = 0.5*mass*(dvx**2).mean()/boltzmann #[K]
    temp_y = 0.5*mass*(dvy**2).mean()/boltzmann #[K]
    temp_z = 0.5*mass*(dvz**2).mean()/boltzmann #[K]


    return_string = "x-temperature = {}[K]".format(temp_x) + "\n" + \
                    "y-temperature = {}[K]".format(temp_y) + "\n" + \
                    "z-temperature = {}[K]".format(temp_z) + "\n"

    print("Longitudinal Energy (Ez) = {}[KeV]".format(Ez/kV) + "\n")
    print(return_string)

    return print("Success")


# Create a copy of data to manipulate later.
data = preparation(ion_data)
Np = len(data) #number of particles
#Create distribution plotsof  vx, vy, and vz
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,7), sharex = True, sharey = True )
vx_plot = axes[0]
vy_plot = axes[1]
sns.distplot(a=data['vx(m/s)'], kde = False, ax = vx_plot)
sns.distplot(a=data['vy(m/s)'], kde = False, ax = vy_plot)
#Axis labels
vx_plot.set_xlabel(r"$v_x$[m/s]", fontsize = 14)
vx_plot.set_ylabel("Counts", fontsize = 14)
vy_plot.set_xlabel(r"$v_y$[m/s]", fontsize = 14)
vx_plot.set_title("Distribution of x-velocity", fontsize = 16)
vy_plot.set_title("Distribution of y-velocity", fontsize = 16)
plt.tight_layout()


#                      Creating energy distribution plots
# # =============================================================================
# #Energy Histograms
#
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,7), sharex = True, sharey = False)
fig.add_subplot(111, frame_on = False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('Energy [eV]')
kinetic_x = axes[0]
kinetic_y = axes[1]
sns.distplot(a = data['Kx(eV)'], kde=False, ax = kinetic_x)
sns.distplot(a = data['Ky(eV)'], kde=False, ax = kinetic_y)
#axis labels
kinetic_x.set_title(r'$K_x$', fontsize = 16)
kinetic_y.set_title(r'$K_y$', fontsize = 16)
kinetic_x.set_ylabel('Frequency', fontsize = 14)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (10,7))
sns.distplot(a=data['Kz(eV)']/kV, kde = False, ax = ax)
ax.set_xlabel("Energy [KeV]", fontsize = 14)
ax.set_ylabel("Frequency", fontsize = 14)
ax.set_title(r'$K_z$', fontsize = 16)
plt.tight_layout()
plt.show()

# =============================================================================


#                   Creating tempearatue distribution plots
# # =============================================================================


fig, ax = plt.subplots(nrows=1, ncols = 2,figsize=(10,7), sharey = True)
fig.add_subplot(111, frame_on = False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('Temperature [K]', fontsize = 14)
Tx_plot = ax[0]
Ty_plot = ax[1]
sns.distplot(a=data['Tx(K)'], kde = False, ax = Tx_plot)
sns.distplot(a=data['Ty(K)'], kde = False, ax = Ty_plot)
Tx_plot.set_ylabel('Counts', fontsize = 14)
Tx_plot.set_title(r"$T_x$", fontsize = 16)
Ty_plot.set_title(r"$T_y$", fontsize = 16)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize = (10,7))
sns.distplot(data['Tz(K)'], kde = False, ax = ax)
ax.set_xlabel("Temperature [K]")
ax.set_ylabel("Counter")
ax.set_title(r"$T_z$", fontsize = 16)
plt.tight_layout()
plt.show()
# # =============================================================================



#                 Data Scraping
# # =============================================================================

#--Using Tx to start, 90% of the data lies below 1500K. This will be the new data
#and I will create a histogram showing the percentage of data that lies from 0-1500K in increments
#of 50K
#--Make histogram that gives the percentage of data points lying within a dvx^2 range
#Put this data into a histogram of 10 bins with  y-axis representing frequency = % of data
bins = [i*25 for i in range(61)]
scraped_data = data['Tx(K)']
scraped_data = scraped_data[scraped_data < 1500]
Npp = len(scraped_data) #Number of Particles
fig, ax = plt.subplots(figsize=(10,7))
ax.hist(scraped_data,bins = bins, weights = np.ones(Npp)/Npp*100)
hist_data = plt.hist(scraped_data,bins = bins, weights = np.ones(Npp)/Npp*100)
ax.set_xlabel(r"$T_x$[K], Bin size of 10", fontsize = 14)
ax.set_ylabel("Percent", fontsize = 14)
ax.set_title("Percent of x-Temperatures between 0 and 1500K", fontsize = 16)
plt.tight_layout()
plt.show()
## 30% of the data is over 100K while only 32% is between 0&10K
## Mean of scraped data is 237K
# # =============================================================================
