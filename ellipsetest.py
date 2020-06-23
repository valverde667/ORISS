import numpy as np
import matplotlib.pyplot as plt

import warp as wp

kV = wp.kV
mm = wp.mm

def fill_velocity(mass, energy, num_of_particles,
              Vx, Vy, Vz,
              temp_para, temp_perp):

        #Unpack tuples
        eVtoK = 8.62e-5 #conversion from eV to Kelvin
        temp_para, temp_perp = temp_para*eVtoK, temp_perp*eVtoK #convert to K
        vperp = np.sqrt(5*wp.boltzmann*temp_perp/(2*mass))
        vpara = np.sqrt(5*wp.boltzmann*temp_para/mass)

        vx = Vx + vperp
        vy = Vy + vperp
        vz = Vz + vpara

        #Fill routine
        counter = 0
        velocity_array = np.zeros([num_of_particles, 3], dtype = float)
        while counter < num_of_particles:

            Ux = np.random.random()
            Uy = np.random.random()
            Uz = np.random.random()

            dvx = vx*(-1 + 2*Ux)
            dvy = vy*(-1 + 2*Uy)
            dvz = vz*(-1 + 2*Uz)

            term1 = (dvx**2 + dvy**2)/(vx**2 + vy**2)
            term2 = dvz**2/vz**2

            condition = term1 + term2

            if condition < 1:
                velocity_array[counter][0] = dvx
                velocity_array[counter][1] = dvy
                velocity_array[counter][2] = abs(vz)
                counter += 1
            else:
                pass

        return velocity_array


uranium_beam = wp.Species(type=wp.Uranium,charge_state=+1,name="Beam species",weight=0)
mass = uranium_beam.mass
energy = .3570*kV
Vx, Vy = 0, 0
Vz = np.sqrt(2*energy*wp.jperev/mass)
print("Velecity = %f" %Vz)
sigma_list = (.1*mm, .1*mm, 1*mm)
temp_para = 4*8.62e-5
temp_perp = 4*8.62e-5

velarray = fill_velocity(mass, energy, 100, Vx, Vy, Vz, temp_para, temp_perp)
