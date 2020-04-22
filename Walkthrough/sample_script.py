#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:51:38 2020

@author: nickvalverde
"""
from warp import *
from Particle_Class import *
from fill_ellipse import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


kV = 1000. 
mm = 1e-3
#create beam
uranium_beam = uranium_beam = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0)
particle_energy = 10*kV
#Create instance 
p = MyParticle(particle_energy, uranium_beam)

def single_particle_example():
    load = p.loader('gaussian')
    print("This is the load array: ", load)
    load_entries = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    for i in range(len(load[0])):
        print("{}={}".format(load[0][i], load_entries[i]))
    print("\n")
    print("Notice that the load is a list of arrays which requires the double indexing.")
    print("load[i][j] is the ith particle array and j is always indexed 1-6.")
    print("\n")
    return True

def five_particle_transverse_width_example():
    #--In this example I'll created a load of 5 particles that will have some x-seperation
    #--To do this I create a uniform list of sepeartions. Then I will use the guassian routine
    #to create particles with these average coordinaes but set sigm=0.
    x_seperation = [-2, -1, 0, 1, 2]
    load_list = []
    for elem in x_seperation:
        load = p.loader('gaussian', avg_coordinates = (elem, 0, 0))
        load_list.append(load)
    
    #--Now the indexing is a little strange here. The loader itself returns
    #a list of arrays which I have just appended to a list. So there are three indexes. 
    print(load_list)
    for i in range(len(load_list)):
        particle_array = load_list[i]
        print("{}th particle array is {}".format(i, particle_array))
    print("\n")
    
    #--I will plot the x-coordinates for each particle
    fig,ax = plt.subplots(figsize=(7,7))
    for elem in load_list:
        xpos = elem[0][0]
        zpos = elem[0][2]
        ax.scatter(zpos, xpos, s = 4)
        ax.set_ylabel('x')
        ax.set_xlabel('z')
    
    return True


def particle_with_temperature_spread():
    #--Inputting a temperature will give a deviation for the guassian routine in the loader file. 
    #--I will input a temperature deviation and plot the x-vx phase-space for 1000 particles. 
    
    sigma_array = (1*mm, 0, 0) #sets sigmax = 1mm
    temperature_array = (0, 8.61e-05) #[1K in eV] array is (T_para, T_perp)
    load = p.loader('gaussian', num_of_particles = 1000, temperature = temperature_array, sigma = sigma_array)
    #--Notice the indexing is different when the num_of_particles is used. Now load[i][j] is the ith particle array and jth coordinate
    xarray = []
    vxarray = []
    for particle_array in load:
        xpos = particle_array[0]
        vxpos = particle_array[3]
        xarray.append(xpos)
        vxarray.append(vxpos)
    xarray, vxarray = np.array(xarray), np.array(vxarray)
    
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(xarray/mm, vxarray, s = .8)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel(r'$v_x$ [m/s]')
    
    
    return True
    
#--I will add uniform ellipsoid examples when I fix the fill_ellipse.py file. 
#It is pretty much the same concept though except you chose a different distriubtion. 