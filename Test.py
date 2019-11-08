#First version with actual mass spectrometry
#Using Uranium (238) and Neptunium (237)
#

from warp import *
import matplotlib.pyplot as plt
import numpy as np

beam = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0) #weight = 0 no spacecharge, 1 = spacecharge. Both go through Poisson sovler.

particle_energy = 3*kV
#0.5*m*v**2 = particle_energy
velocity = np.sqrt(2.*particle_energy*echarge/beam.mass)
beam.addparticles(x=0., y=0., z=0., vx=0., vy=0., vz=velocity)

beam.ekin