import numpy as np
import matplotlib.pyplot as plt
from warp import *


def fill_position(num_of_particles, sigma, avg_coordinates):
        if ('sigma' in kwargs) == False:
            raise Exception("Must provide standard deviations for ellipse 'sigma=(sigma_x, sigma_y, sigma_z)'")
        else:
            #Create positions lists to be populated
            x_list = []
            y_list = []
            z_list = []

            #Initialize sigmas
            sigmax = kwargs['sigma'][0]
            sigmay = kwargs['sigma'][1]
            sigmaz = kwargs['sigma'][2]


            #Set average coordinates
            if 'avg_coordinates' in kwargs:
                X, Y, Z = kwargs['avg_coordinates'][0], kwargs['avg_coordinates'][1], kwargs['avg_coordinates'][2]
            else:
                X, Y, Z = 0., 0., 0.


            counter = 0
            position_array = []
            while counter < num_of_particles:
                x = np.random.normal(X,sigmax)
                y = np.random.normal(Y,sigmay)
                z = np.random.normal(Z,sigmaz)

                #Set up conditional statement:
                axis1 = (sigmax**2 + sigmay**2)/(x**2 + y**2)
                axis2 = sigmaz**2/z**2
                value = axis1 + axis2 #Condition
                #Test condition, if greater than 1, repeat while loop.
                if value > 1:
                    continue
                else: #Condition is satisfied so these coordinates are good. Append to list.
                    position_array.append((x,y,z))

                    counter += 1

            position_array = np.array(position_array)
            return position_array


def fill_velocity(mass, energy,num_of_particles, avg_velocities, Vz, temeprature):

        #Unpack tuples
        Vx, Vy = avg_velocities[0], avg_velcities[1]
        temp_para, temp_perp = temperature[0], temperature[1]

        #Solve for velocitiy deviations
        sigmax = np.sqrt((5/2)*jperev*temp_perp/mass)
        sigmay = sigmax
        sigmaz = np.sqrt(5*jperev*temp_para/mass)

        #Random Guassian routine
        while counter < num_of_particles:
            vx = np.random.normal(Vx,sigmax)
            vy = np.random.normal(Vy,sigmay)
            vz = np.random.normal(Vz,sigmaz)

            #Set up conditional statement:
            axis1 = (sigmax**2 + sigmay**2)/(vx**2 + vy**2)
            axis2 = sigmaz**2/vz**2
            value = axis1 + axis2 #Condition
            #Test condition, if greater than 1, repeat while loop.
            if value > 1:
                continue
            else:#Condition was met so these velocities are good. Append to list
                velocity_array.append((vx,vy,vz))

                counter += 1


        velocity_array = np.array(velocity_array)

        return velocity_array
