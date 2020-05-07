#/Users/nickvalverde/Research/ORISS/Particle_Class.py#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 17:20:33 2019

@author: nickvalverde
"""

from warp import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set_style('darkgrid')
#import sys 


#stdoutOrigin=sys.stdout #writes output in terminal to a file. 
#sys.stdout = open("ringle_data.txt", "w")


yttrium_beam = Species(type=Uranium,charge_state=+0,name="Beam species",weight=0) #weight = 0 no spacecharge, 1 = spacecharge. Both go through Poisson sovler.
mass = yttrium_beam.mass

print("\n")
print("Yttrium Beam used")
print("\n")
#--Ion Data
column_list = ["t(s)", "x(m)", "vx(m/s)", "y(m)", "vy(m/s)", "z(m)", "vz(m/s)"]
ion_data = pd.read_excel('/Users/nickvalverde/Dropbox/Research/ORISS/Ringle_Data_Plots/ion_data.xlsx', names = column_list)
print("Ion Data")
print(40*'_')


print("---Position Averages---")
#fix z-coordinates
min_time = ion_data['t(s)'].min()
ion_data['zpos'] = ion_data['vz(m/s)']*(ion_data['t(s)'] - min_time)
average_x = ion_data['x(m)'].mean()
average_y = ion_data['y(m)'].mean()
average_z = ion_data['zpos'].mean()
print("x average = ", average_x/mm, "[mm]")
print("y average = ", average_y/mm, "[mm]")
print("z average = ", average_z/mm, "[mm]")
print(40*'-')

print("---Velocity Averages---")
average_vx = ion_data['vx(m/s)'].mean()
average_vy = ion_data['vy(m/s)'].mean()
average_vz = ion_data['vz(m/s)'].mean()

print("vx average = ", average_vx, "[m/s]")
print("vy average = ", average_vy, "[m/s]")
print("vz average = ", average_vz, "[m/s]")
print(40*'-')

print("---Position Standard Deviations---")
dx = ion_data['x(m)'] - average_x 
sigma_x = np.sqrt((dx**2).mean())
dy = ion_data['y(m)'] - average_y
sigma_y = np.sqrt((dy**2).mean())
dz = ion_data['zpos'] - average_z
sigma_z = np.sqrt((dz**2).mean())
print("x-sigma = ", sigma_x/mm, "[mm]")
print("y-sigma = ", sigma_y/mm, "[mm]")
print("z-sigma = ", sigma_z/mm, "[mm]")
print(40*'-')

print("---Velocity Standard Deviations---")
dvx = ion_data['vx(m/s)'] - average_vx 
sigma_vx = np.sqrt((dvx**2).mean())
dvy = ion_data['vy(m/s)'] - average_vy
sigma_vy = np.sqrt((dvy**2).mean())
dvz = ion_data['vz(m/s)'] - average_vz
sigma_vz = np.sqrt((dvz**2).mean())
print("vx-sigma = ", sigma_vx, "[m/s]")
print("vy-sigma = ", sigma_vy, "[m/s]")
print("vz-sigma = ", sigma_vz, "[m/s]")
print(40*'-')

print("---Longitudinal Energy and Temperatures---")
longitudinal_energy = mass*average_vz**2/2/jperev
print("Longitudinal Energy = ",longitudinal_energy/kV,"[KeV]")
#temperature calculations 
temp_x = mass*((dvx**2).mean())/boltzmann
print("x-temperature = ", temp_x*boltzmann/jperev,"[eV]",',', temp_x, "[K]")
temp_y = mass*((dvy**2).mean())/boltzmann
print("y-temperature = ", temp_y*boltzmann/jperev, "[eV]",',', temp_y, "[K]")
temp_z = mass*((dvz**2).mean())/boltzmann
print("z-temperature = ", temp_z*boltzmann/jperev, "[eV]",',', temp_z, "[K]")
print(40*"-")

print("---Emittances Calculated From Temperature---")
emittance_x = sigma_x*np.sqrt(temp_x/longitudinal_energy)/np.sqrt(2)
emittance_y = sigma_y*np.sqrt(temp_y/longitudinal_energy)/np.sqrt(2)
emittance_z = sigma_z*np.sqrt(temp_z/longitudinal_energy)/np.sqrt(2)
print("x-emittance = ", emittance_x/(mm*mm), "[mm-mrad]")
print("y-emittance = ", emittance_y/(mm*mm), "[mm-mrad]")
print("z-emittance = ", emittance_z/(mm*mm), "[mm-mrad]")
#normalized emittance calculation
coeff = average_vz/clight
norm_emittance_x = coeff*sigma_x*np.sqrt(temp_x/longitudinal_energy)/np.sqrt(2)
norm_emittance_y = coeff*sigma_y*np.sqrt(temp_y/longitudinal_energy)/np.sqrt(2)
norm_emittance_z = coeff*sigma_z*np.sqrt(temp_z/longitudinal_energy)/np.sqrt(2)
print("normalized x-emittance = ", norm_emittance_x/(mm*mm), "[mm-mrad]")
print("normalized y-emittance = ", norm_emittance_y/(mm*mm), "[mm-mrad]")
print("normalized z-emittance = ", norm_emittance_z/(mm*mm), "[mm-mrad]")
print(40*"_")
print("---Emittances Directly Calculated---")
def emittance_calc(dx, dxprime):
    term1 = (dx**2).mean()*(dxprime**2).mean()
    term2 = ((dx*dxprime).mean())**2
    return (term1 - term2)

dxprime = (ion_data['vx(m/s)'] - average_vx)/average_vz
dyprime = (ion_data['vy(m/s)'] - average_vy)/average_vz
dzprime = (ion_data['vz(m/s)'] - average_vz)/average_vz 
xprime = ion_data['vx(m/s)']/ion_data['vz(m/s)']
#plt.scatter(ion_data['x(m/s)'], xprime, s=.1 )
#plt.show()
emittance_x = np.sqrt(emittance_calc(dx, dxprime))
emittance_y = np.sqrt(emittance_calc(dy, dyprime))
emittance_z = np.sqrt(emittance_calc(dz, dzprime))
print("x-emittance = ", emittance_x/(mm*mm), "[mm-mrad]")
print("y-emittance = ", emittance_y/(mm*mm), "[mm-mrad]")
print("z-emittance = ", emittance_z/(mm*mm), "[mm-mrad]")
#normalized emittance calculation
coeff = 1/clight**2
norm_emittance_x = np.sqrt(coeff*((dx**2).mean()*(dvx**2).mean() - ((dx*dvx).mean())**2))
norm_emittance_y = np.sqrt(coeff*((dy**2).mean()*(dvy**2).mean() - ((dy*dvy).mean())**2))
norm_emittance_z = np.sqrt(coeff*((dz**2).mean()*(dvz**2).mean() - ((dz*dvz).mean())**2))
print("normalized x-emittance = ", norm_emittance_x/(mm*mm), "[mm-mrad]")
print("normalized y-emittance = ", norm_emittance_y/(mm*mm), "[mm-mrad]")
print("normalized z-emittance = ", norm_emittance_z/(mm*mm), "[mm-mrad]")
print(40*"_")
print("\n")






#--No SC Data
no_Sc_data = pd.read_excel('/Users/nickvalverde/Dropbox/Research/ORISS/Ringle_Data_Plots/no_SC_data.xlsx', names = column_list)
print("No SC data")
print(40*'_')

print("---Position Averages---")
min_time = ion_data['t(s)'].min()
no_Sc_data['zpos'] = ion_data['vz(m/s)']*(ion_data['t(s)'] - min_time)

average_x = no_Sc_data['x(m)'].mean()
average_y = no_Sc_data['y(m)'].mean()
average_z = no_Sc_data['zpos'].mean()
print("x average = ", average_x/mm, "[mm]")
print("y average = ", average_y/mm, "[mm]")
print("z average = ", average_z/mm, "[mm]")
print(40*'-')

print("---Velocity Averages---")
average_vx = no_Sc_data['vx(m/s)'].mean()
average_vy = no_Sc_data['vy(m/s)'].mean()
average_vz = no_Sc_data['vz(m/s)'].mean()
print("vx average = ", average_vx, "[m/s]")
print("vy average = ", average_vy, "[m/s]")
print("vz average = ", average_vz, "[m/s]")
print(40*'-')

print("---Position Standard Deviations---")
dx = no_Sc_data['x(m)'] - average_x 
sigma_x = np.sqrt((dx**2).mean())
dy = no_Sc_data['y(m)'] - average_y
sigma_y = np.sqrt((dy**2).mean())
dz = no_Sc_data['zpos'] - average_z
sigma_z = np.sqrt((dz**2).mean())
print("x-sigma = ", sigma_x/mm, "[mm]")
print("y-sigma = ", sigma_y/mm, "[mm]")
print("z-sigma = ", sigma_z/mm, "[mm]")
print(40*'-')

print("---Velocity Standard Deviations---")
dvx = no_Sc_data['vx(m/s)'] - average_vx 
sigma_vx = np.sqrt((dvx**2).mean())
dvy = no_Sc_data['vy(m/s)'] - average_vy
sigma_vy = np.sqrt((dvy**2).mean())
dvz = no_Sc_data['vz(m/s)'] - average_vz
sigma_vz = np.sqrt((dvz**2).mean())
print("vx-sigma = ", sigma_vx, "[m/s]")
print("vy-sigma = ", sigma_vy, "[m/s]")
print("vz-sigma = ", sigma_vz, "[m/s]")
print(40*'-')

print("---Longitudinal Energy and Temperatures---")
longitudinal_energy = mass*average_vz**2/2/jperev
print("Longitudinal Energy = ",longitudinal_energy/kV,"[KeV]")
#temperature calculations 
temp_x = 0.5*mass*((dvx**2).mean())/boltzmann
print("x-temperature = ", temp_x*boltzmann/jperev,"[eV]",',', temp_x, "[K]")
temp_y = mass*((dvy**2).mean())/boltzmann
print("y-temperature = ", temp_y*boltzmann/jperev, "[eV]",',', temp_y, "[K]")
temp_z = mass*((dvz**2).mean())/boltzmann
print("z-temperature = ", temp_z*boltzmann/jperev, "[eV]",',', temp_z, "[K]")
print(40*"-")

print("---Emittances Calculated from Temperature---")
emittance_x = sigma_x*np.sqrt(temp_x/longitudinal_energy)/np.sqrt(2)
emittance_y = sigma_y*np.sqrt(temp_y/longitudinal_energy)/np.sqrt(2)
emittance_z = sigma_z*np.sqrt(temp_z/longitudinal_energy)/np.sqrt(2)
print("x-emittance = ", emittance_x/(mm*mm), "[mm-mrad]")
print("y-emittance = ", emittance_y/(mm*mm), "[mm-mrad]")
print("z-emittance = ", emittance_z/(mm*mm), "[mm-mrad]")
#normalized emittance calculation
coeff = average_vz/clight
norm_emittance_x = coeff*sigma_x*np.sqrt(temp_x/longitudinal_energy)/np.sqrt(2)
norm_emittance_y = coeff*sigma_y*np.sqrt(temp_y/longitudinal_energy)/np.sqrt(2)
norm_emittance_z = coeff*sigma_z*np.sqrt(temp_z/longitudinal_energy)/np.sqrt(2)
print("normalized x-emittance = ", norm_emittance_x/(mm*mm), "[mm-mrad]")
print("normalized y-emittance = ", norm_emittance_y/(mm*mm), "[mm-mrad]")
print("normalized z-emittance = ", norm_emittance_z/(mm*mm), "[mm-mrad]")
print(40*"_")
print("\n")
print("---Emittances Directly Calculated---")
def emittance_calc(dx, xdprime):
    term1 = (dx**2).mean()*(dxprime**2).mean()
    term2 = ((dx*dxprime).mean())**2
    return (term1 - term2)

dxprime = (no_Sc_data['vx(m/s)'] - average_vx)/average_vz
dyprime = (no_Sc_data['vy(m/s)'] - average_vy)/average_vz
dzprime = (no_Sc_data['vz(m/s)'] - average_vz)/average_vz 

emittance_x = np.sqrt(emittance_calc(dx, dxprime))
emittance_y = np.sqrt(emittance_calc(dy, dyprime))
emittance_z = np.sqrt(emittance_calc(dz, dzprime))

print("x-emittance = ", emittance_x/(mm*mm), "[mm-mrad]")
print("y-emittance = ", emittance_y/(mm*mm), "[mm-mrad]")
print("z-emittance = ", emittance_z/(mm*mm), "[mm-mrad]")
#normalized emittance calculation
coeff = 1/clight**2
norm_emittance_x = np.sqrt(coeff*((dx**2).mean()*(dvx**2).mean() - ((dx*dvx).mean())**2))
norm_emittance_y = np.sqrt(coeff*((dy**2).mean()*(dvy**2).mean() - ((dy*dvy).mean())**2))
norm_emittance_z = np.sqrt(coeff*((dz**2).mean()*(dvz**2).mean() - ((dz*dvz).mean())**2))
print("normalized x-emittance = ", norm_emittance_x/(mm*mm), "[mm-mrad]")
print("normalized y-emittance = ", norm_emittance_y/(mm*mm), "[mm-mrad]")
print("normalized z-emittance = ", norm_emittance_z/(mm*mm), "[mm-mrad]")
print(40*"_")
print("\n")

#sys.stdout.close()
#sys.stdout=stdoutOrigin


#plotting
#fig = sns.distplot(array of value, kde=False, bins = 40)
#fig.set_xlabel()
#fig = sns.jointplot(x = , y =, data = , s = .1)
#fig.set_axis_labels(x-axis, y-axis)
