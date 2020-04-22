import numpy as np
from warp import *
import matplotlib.pyplot as plt




class MyParticle(object):
    """
    For creating various characteristics of a particle that will ultimately be loaded on a beam.

    Methods:
        constructor(energy, beam): Sets energy of particle (in eV) and gives the beam from the warp library. The beam variable,
                                   which is type string, sets the particle type which gives mass, charge state, etc. This works off
                                   the warp package.The beam should be set before the particle class is called.

        position(distribution, num_of_particles, **kwargs): This will output a position array that is determined by the distribution (type string) given. The distribution
                                          types are:
                                              -gaussian
                                              -uniform_ellipsoid
                                              -None.
                                          If uniform_ellipsoid is selected sigma (see below) must be entered or the program will terminate with
                                          an exception.

                                        -num_of_particles: An integer that specifies how many position arrays will be created and outputted.

                                         -**kwargs: Additional parameters used for a given distribution.
                                            -sigma: Input as an array in transverse direction (sigmax, sigmay, sigmaz). Must be specified for uniform_ellipsoid.
                                         If no kwargs are given all values are set to 0 and the method returns the position array (0, 0, 0)
                                         for the num_of_particles entered.

        velocity (distribution, num_of_particles **kwargs): Outputs a velocity array (vx, vy, vz) based on the distribution. The types are identical to the position inputs:
                                                -gaussian
                                                -uniform_ellipsoid
                                                -None.
                                            If uniform_ellipsoid is selected the temperature must be inputted as a kwarg, see below. The velocity is
                                            outputed in mks units to match with Warp's beam.load specifications.

                                            -num_of_particles: An integer that specifies how many velocity arrays will be created and outputted.

                                            -**kwargs: Additional parameters used for a given distribution.
                                                -temperature. Input as an array 'temperature=(T_perpendicular, T_parallel)'. This kwarg is required
                                                for the uniform_ellipsoid distribution since it will determine the standard deviations. The temperature should be
                                                input in units of eV.

        data(energy, **kwargs): Outputs the regular and normalized emittances.
                                -energy: energy given by the initial set in Particle.
                                -kwargs: should include 'temperature = (T_perpendicular, T_parallel)' and
                                        'sigma = (sigmax, sigmay, sigmaz)' which will be used to determine emittances.


        loader(distribution, num_of_particles, **kwargs): This method will create a (num_of_particle x 6) array (x, y, z, vx, vy, vz) that can be used to load
                                                        onto a beam in the main script. The kwargs should match the requirments for each of the above functions. For
                                                        example, for a 20 particle uniform_ellipsoid load the arguments should look as follows:
                                                            load('uniform_ellipsoid', 20, sigma=(sigmx,sigmay,sigmaz), temperature = (T_perpendicular, T_parallel)).
                                                        This will return a (20x6) particle array that can be looped through and individually loaded onto a beam.
                                                        It will also print out an emittance table giving the transverse and parallel emittance values along with
                                                        the normalized versions.


    """


    def __init__(self, energy, beam):
        self.energy = energy
        self.beam = beam

    def position(self, distribution, num_of_particles=1, sigma, avg_coordinates):
        #Unpack tuples in coordinates and sigma
        sigmax, sigmay, sigmaz = sigma[0], sigma[1], sigma[2]
        X, Y, Z = avg_coordinates[0], avg_coordinates[1], avg_coordinates[2]

        #################### Guassian Routine ##############################
        ##--Sift through Kwargs and initialize variables
        if distribution == 'gaussian':

            ##--Create position arrays based on a Gaussian random number generator.
            ##--The output will be a an array containing positional arrays for the number of particles.

            #Gaussian routine for position
            counter = 0
            position_array = [] #Initialize empty list to populate with positional arrays (x,y,z)
            while counter < num_of_particles:
                #Randomly generate positions
                x = np.random.normal(X, sigmax)
                y = np.random.normal(Y, sigmay)
                z = np.random.normal(Z, sigmaz)
                position_array.append((x,y,z)) #append positions to list

                counter += 1

            position_array = np.array(position_array) #Turn list into an array
            return position_array


        ################# Uniform Ellipsoid Routine ##########################
        elif distribution == 'uniform_ellipsoid':
            from fill_ellipse import fill_position #import function to fill ellipsoid if needed.
                                                  #Returns an array of partticle coordinates for given number of particles.

            #Check if any kwargs were given. If not, initialize variables to zero.
            if len(kwargs) == 0: #Chekc if any kwargs were given. If not, initialize variables to zero.
                sigmax, sigmay, sigmaz = 0., 0., 0.
                X, Y, Z= 0., 0., 0.
                Xp, Yp, Zp = 0., 0., 0.

            #Search Kwargs for key words and initialize variables. If not there, set to 0.
            else:
                position_array = fill_position(num_of_particles, **kwargs)

                return position_array

        else:
            return print("No distribution selected")


    def velocity(self, distribution, num_of_particles=1, **kwargs):
        #set longitudinal centroid velocity from outset based on energy input
        Vz = np.sqrt(2*jperev*self.energy/self.beam.mass)

        ################## Gaussian Routine ##################################
        if distribution == 'gaussian':
            if len(kwargs) == 0: #Chekc if any kwargs were given. If not, initialize variables to zero.
                temp_perp, temp_para = 0., 0.
                Vx,Vy = 0., 0.
            else:
                if 'temperature' in kwargs:
                    temp_perp, temp_para = kwargs['temperature'][0], kwargs['temperature'][1]
                else:
                    temp_perp, temp_para = 0., 0.
                if 'avg_velocity' in kwargs:
                    Vx, Vy = kwargs['velocity'][0], kwargs['velocity'][1]
                else:
                    Vx, Vy = 0., 0.

            #Calculate stand deviations in transverse and logitudinal directions based on Temperatures
            sigma_transverse_velocity = np.sqrt(jperev*temp_perp/self.beam.mass)
            sigma_parallel_velocity = np.sqrt(jperev*temp_para/self.beam.mass)

            #Gaussian Algorithm to randomaly generate velocities
            counter = 0
            velocity_array = []
            while counter < num_of_particles:
                vx = np.random.normal(Vx, sigma_transverse_velocity)
                vy = np.random.normal(Vy, sigma_transverse_velocity)
                vz = np.random.normal(Vz, sigma_parallel_velocity)
                velocity_array.append((vx,vy,vz))

                counter += 1

            velocity_array = np.array(velocity_array)

            return velocity_array

        ################## Uniform Ellipsoid Routine ##########################
        elif distribution == 'uniform_ellipsoid':
            #Import fill_velocity function from the fill_ellipse module
            from fill_ellipse import fill_velocity
            mass = beam.mass
            energy = self.energy
            velocity_array = fill_velocity(mass, energy, num_of_particles, avg_velocities, Vz, sigma, temeprature)

            return velocity_array


        else:
            return print("No distribution selected.")



    def loader(self, distribution, num_of_particles=1, sigma = (0., 0., 0.), avg_coordinates = (0., 0., 0.), avg_angles = (0., 0., 0.),
    avg_velcities = (0, 0), temperature = (0, 0), fill = True):
    """This function will return an array of (x, y, z, vx, vy, vz) that can be looped through to load onto a beam in warp."""
        #Unpack tuples
        sigamx, sigmay, sigmaz = sigma[0], sigma[1], sigma[2]
        temp_para, temp_perp = temperature[0], temperature[1]

        position = self.position(distribution, num_of_particles, sigma, avg_coordinates, avg_angles)
        velocity = self.velocity(distribution, num_of_particles, avg_velocities, temperature)
        load = np.concatenate((position,velocity),axis=1)

        #--Data Output
        #Create Table function to call if sigma and temperature are given
        def emittance_table(transverse_emittance, parallel_emittance, norm_transverse_emittance, norm_parallel_emittance):
            print(40*"=")
            print("Transverse emittance in mm-mrad = ", transverse_emittance/(mm*mm))
            print("Parallel emittance in mm-mrad = ", parallel_emittance/(mm*mm))
            print(40*'-')
            print("Normalized Transverse emittance in mm-mrad = ", norm_transverse_emittance/(mm*mm))
            print("Normalized Parallel emittance in mm-mrad = ", norm_parallel_emittance/(mm*mm))
            print(40*'=')
            return ''

        if ('sigma' in kwargs) and ('temperature' in kwargs):
            sigmax, sigmay, sigmaz = kwargs['sigma'][0], kwargs['sigma'][1], kwargs['sigma'][2]
            temp_perp, temp_para = kwargs['temperature'][0], kwargs['temperature'][1]

            Vz = np.sqrt(2*jperev*self.energy/self.beam.mass) #parallel velocity

            #Calculate Emittances
            transverse_emittance = sigmax*np.sqrt(temp_perp/self.energy)/np.sqrt(2)
            parallel_emittance = sigmaz*np.sqrt(temp_para/self.energy)/np.sqrt(2)

            #Calculate Normalized Emittances
            beta = Vz/clight
            norm_transverse_emittance = beta*transverse_emittance
            norm_parallel_emittance = beta*parallel_emittance

            data = emittance_table(transverse_emittance, parallel_emittance, norm_transverse_emittance, norm_parallel_emittance)
        else:
            #data = "-Emittance not calculated since sigma and temperature werent provided"
            pass

        #print(data)

        return load





### Example Scripts ###
#Create beams
def example():
    uranium_beam = Species(type=Uranium,charge_state=+1,name="Uranium Beam",weight=0) #weight = 0 no spacecharge, 1 = spacecharge. Both go through Poisson sovler.
    neptunium_beam = Species(type=Neptunium,charge_state=+1,name="Neptunium Beam",weight=0)
    potassium_beam = Species(type=Potassium, charge_state=+2, name="Phosphourous Beam", weight=0)


#-Single uranium Particle Load 10eV
    print("Single uranium particle gaussian load with E=10eV")
    single_particle = MyParticle(10, uranium_beam)
    #Create Gaussian Load
    gauss_load = single_particle.loader('gaussian') #default
    print(gauss_load)
    print(80*'x')
    print("\n")
    #Create uniform ellipse load
    print("Single uranium particle ellipse load with E=10eV")
    ellipse_load = single_particle.loader('uniform_ellipsoid', temperature=(.12,.13), sigma=(1*mm,1*mm,1*mm))
    print(ellipse_load)
    print(80*'x')
    print("\n")

    #--Create a 10eV potassium loader with 10 particles
    print("10 potasium particle gaussian load with E=1eV")
    particles = MyParticle(1, potassium_beam)
    gauss_load = particles.loader('gaussian', 10, sigma = (1*mm,1*mm,1*mm), temperature=(.12, .13))
    print(gauss_load)
    print(80*'x')
    print("\n")
    #Uniform ellipse
    print("10 potasium particle ellipse load with E=1eV")
    ellipse_load = particles.loader('uniform_ellipsoid', 10, temperature=(.12,.13), sigma=(1*mm,1*mm,1*mm))
    print(ellipse_load)
    print(80*'x')
    print("\n")


#--Create a mixed beam load with equal energy of 1eV
    load_list = []
    beams = [uranium_beam, neptunium_beam, potassium_beam]
    species_ratio = [30, 30, 10]
    for beam, ratio in zip(beams,species_ratio):
        print("Emittances for", beam.name)
        particles = MyParticle(1, beam)
        load_list.append(particles.loader('gaussian', ratio, sigma = (1*mm,1*mm,1*mm), temperature = (.12,.13)))

    load_list = np.vstack(load_list)
    print(load_list)

    return print("Success")
