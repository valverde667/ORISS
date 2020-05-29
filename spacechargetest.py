#First version with actual mass spectrometry
#Using Uranium (238) and Neptunium (237)
#IMPORTANT DO NOT NAME ANYTHING LOAD, this interrupts with a declared functionion Species.py

#--Import Warp Packages and modules
from warp import *
from warp.particles.singleparticle import TraceParticle
from Forthon import *

#--Import user created files
from Particle_Class import *
from fill_ellipse import *

#--Import python packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import numpy as np
import time



####################################################################
# Simulation Mesh, create in w3d but run field solve in r-z
####################################################################




####################################################################
# Particle Moving and Species
####################################################################

top.dt     = 1e-7 # Timestep of particle advance



#Create Species of particles to advance
particle1 = Species(type=Uranium,charge_state=+100,name="Beam species",weight=1)
particle2 = Species(type=Uranium,charge_state=+100,name="Beam species",weight=1)



####################################################################
# Generate Code
####################################################################
w3d.xmax = 1.5e-3
w3d.xmin = -.5e-4
w3d.nx = 4000
generate()     # Initate code, this will also make an initial fieldsolve

solver.ldosolve = True #This sets up the field solver with the beam fields.
                     #particles are treated as interacting. False, turns off space charge

particle1.addparticles(0,0,0,0,0,0)
particle2.addparticles(0,0,.00001,0,0,0)
time = [0]
p1z = [0]
p2z = [.001]
timer = 0
while True:
    step(10)  # advance particle
    timer += 10
    p1z.append(particle1.getz()[0])
    p2z.append(particle2.getz()[0])
    time.append(timer)
    if timer > 80:
        break
    else:
        continue

p1z = np.array(p1z)/mm
p2z = np.array(p2z)/mm
com = (p2z - p1z)/2
time = np.array(time)/1e-9

plt.scatter(time, com, s = .5)
plt.xlabel('Time (ns)')
plt.ylabel('z [mm]')
plt.show()