#Loader Function

from warp import *
import numpy as np
import matplotlib.pyplot as plt


particle = Species(type = Uranium, charge_state = +1, name = "Single Particle", weight = 0)

#Energy convesion from mass    mass*clight**2/echarge will give electron-volts
#0.5mv^2 = energy
#the add beam takes in velocities so Input:Energy Output:velocity

initialx = 1
initialy = 2
initialz = 3
beam00 = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0) #weight = 0 no spacecharge, 1 = spacecharge. Both go through Poisson sovler.
#Do I need to specify particles and a beam? Or can I just specify the beam. Seems redundant doing both.
#No documentation on how to load particles with distribution. So it might
#be better to just work with the beam.
def loader(energy, rz, rperp, Tperp, Tz, number_of_particles):
    """This function creates particles and loads them onto the beam.
    For now, the function will put all energy into the longitudinal velocity"""
    energy_in_joules = energy*echarge

    velocity = np.sqrt(2*energy_in_joules/particle.mass)
    return True

loader(1, beam00, particle)

#input in loader sholud be energy rz, rperp, Tperp, Tz.
