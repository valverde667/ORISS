import numpy as np
import matplotlib.pyplot as plt
import warp as wp


def fill_position(num_of_particles, sigma, avg_coordinates):
    X,Y,Z = avg_coordinates[0],avg_coordinates[1],avg_coordinates[2]
    sigmax,sigmay,sigmaz = sigma[0],sigma[1],sigma[2]

    #initialize coordinates for routine
    x = np.random.normal(X, sigmax)
    y = np.random.normal(Y, sigmay)
    z = np.random.normal(Z, sigmaz)
    rperp = np.sqrt(x**2 + y**2)
    rz = z

    counter = 0
    position_array = np.zeros([num_of_particles, 3], dtype = float)
    while counter < num_of_particles:

        Ux = np.random.uniform(0,1)
        Uy = np.random.uniform(0,1)
        Uz = np.random.uniform(0,1)

        dx = rperp*(-1 + 2*Ux)
        dy = rperp*(-1 + 2*Uy)
        dz = z*(-1 + 2*Uz)

        term1 = (dx**2 + dy**2)/rperp**2
        term2 = dz**2/rz**2

        condition = term1 + term2

        if condition < 1:
            position_array[counter][0] = dx
            position_array[counter][1] = dy
            position_array[counter][2] = dz
            counter += 1
        else:
            pass

    return position_array


def fill_velocity(mass, energy,num_of_particles, avg_velocities, Vz, temperature):

        #Unpack tuples

        Vx, Vy = avg_velocities[0], avg_velocities[1]
        eVtoK = 8.62e-5 #conversion from eV to Kelvin
        Vz = np.sqrt(2*wp.jperev*energy/mass)
        temp_para, temp_perp = temperature[0]*eVtoK, temperature[1]*eVtoK #convert to K
        vperp = np.sqrt(5*wp.boltzmann*temp_perp/(2*mass))
        vpara = np.sqrt(5*wp.boltzmann*temp_para/mass)

        vx = Vx + vperp
        vy = Vy + vperp
        vz = Vz + vpara

        #Fill routine
        counter = 0
        velocity_array = np.zeros([num_of_particles, 3], dtype = float)
        while counter < num_of_particles:

            Ux = np.random.uniform(0,1)
            Uy = np.random.uniform(0,1)
            Uz = np.random.uniform(0,1)

            dvx = vx*(-1 + 2*Ux)
            dvy = vy*(-1 + 2*Uy)
            dvz = vz*(-1 + 2*Uz)

            term1 = (dvx**2 + dvy**2)/(vx**2 + vy**2)
            term2 = dvz**2/vz**2

            condition = term1 + term2

            if condition < 1:
                velocity_array[counter][0] = dvx
                velocity_array[counter][1] = dvy
                velocity_array[counter][2] = abs(dvz)
                counter += 1
            else:
                pass

        return velocity_array
