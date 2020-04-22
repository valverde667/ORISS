import numpy as np
import matplotlib.pyplot as plt
from warp import *
from Particle_Class_copy import *
from fill_ellipse_copy import *


kV = 1000
mm = 1e-3

uranium_beam = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0)
particle_energy = 12*kV
p = MyParticle(particle_energy, uranium_beam)
temperature_array = (8.61e-05, 8.61e-05) #[1K in eV], y-temp set to 0
sigma_array= (.01*mm, 0, .01*mm) #y-sigma set to 0
N = 1000 #number of particles
load_list = p.loader('uniform_ellipsoid', num_of_particles = N, temperature = temperature_array, sigma = sigma_array)






#Create phase-space plots
xbanana, vxbanana, zbanana, vzbanana = [], [], [], []
#***warning: apparently naming arrays xarray, xpos_array, etc throws warp off. Must be something in the
#library that gets imported in import * I dont know know about. Try fixing code to import necessary warp libraries.
for load in load_list:
    #Unpack coordinates and add to beam
    xpos, ypos, zpos = load[0], load[1], load[2]
    vx, vy, vz = load[3], load[4], load[5] #1.178 to math V4 code
    uranium_beam.addparticles(xpos, ypos, zpos, vx, vy, vz)
 
    #Populate phase-space arrays
    xbanana.append(xpos)
    vxbanana.append(vx)
    zbanana.append(zpos)
    vzbanana.append(vz)
 
xbanana, vxbanana, zbanana, vzbanana = np.array(xbanana), np.array(vxbanana), np.array(zbanana), np.array(vzbanana)
 
 # x-vx phase space plot
fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(x= xbanana/mm, y = vxbanana, s = .7)
ax.set_title(r'$x-v_x$ plot for %g particles' %N)
ax.set_xlabel(r'$x$ [mm]')
ax.set_ylabel(r'$v_x$[m/s]')
plt.tight_layout()
plt.savefig('/Users/nickvalverde/Research/ORISS/Runs_Plots/xvx_phase-space.png', dpi=300)

fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(x= zbanana/mm, y = vzbanana, s = .7)
ax.set_title(r'$z-v_z$ plot for %g particles' %N)
ax.set_xlabel(r'$z$ [mm]')
ax.set_ylabel(r'$v_z$[m/s]')
plt.tight_layout()
plt.savefig('/Users/nickvalverde/Research/ORISS/Runs_Plots/zvz_phase-space.png', dpi=300)