#First version with actual mass spectrometry
#Using Uranium (238) and Neptunium (237)
#IMPORTANT DO NOT NAME ANYTHING LOAD, this interrupts with a declared functionion Species.py


#--Import python packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import numpy as np
import time

#--Import Warp Packages and modules
import warp as wp
from warp.particles.singleparticle import TraceParticle

#--Import user created files
from Particle_Class import *
from fill_ellipse import *
import diagnostics.diagplt as diagnostics


start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))


####################################################################
# Simulation Mesh, create in w3d but run field solve in r-z
####################################################################

exec(open("ORISS_Geometry.py").read()) #Create simulation mesh



####################################################################
# Particle Moving and Species
####################################################################
mm = wp.mm
kV = wp.kV
wp.top.dt     = 1e-7 # Timestep of particle advance

#Set particle boundary conditions at mesh ends
wp.top.pbound0 = wp.absorb  #boundary condition at iz == 0
wp.top.pboundnz = wp.absorb #boundary condition at iz == nz
wp.top.pboundxy = wp.absorb #boundary condition at edge or radial mesh


#Create Species of particles to advance
uranium_beam = wp.Species(type=wp.Uranium,charge_state=+1,name="Beam species",weight=0)
               #weight = 0 no spacecharge, 1 = spacecharge.

wp.top.lbeamcom = False #Specify grid does not move to follow beam center




####################################################################
# Generate Code
####################################################################

wp.package("w3d") # Specify 3D code.  But fieldsolver will be r-z and particles will deposit on r-z mesh
wp.generate()     # Initate code, this will also make an initial fieldsolve
print("--- %s seconds ---" % (time.time() - start_time))
solver.ldosolve = False #False/True turns off/on space-charge


#Load Parameters
Np = 3000
particle_energy = .3570*kV

z = wp.w3d.zmesh
sigma_list = (.1*mm, .1*mm, 1*mm)               #Standard deviations (sx, sy, sz)
temp_list = (4*8.62e-5, 4*8.62e-5)                #8.62e-5[eV] = 1 [K]
pos_dist = 'uniform_ellipsoid'                         #Distribution for position
vel_dist = 'uniform_ellipsoid'                         #Distribution for velocity
p = MyParticle(particle_energy, uranium_beam)
load_list  = p.loader(position_distribution=pos_dist, velocity_distribution=vel_dist,
num_of_particles=Np, sigma=sigma_list, temperature=temp_list)
uranium_beam.addparticles(load_list[:,0], load_list[:,1], load_list[:,2],
                        load_list[:,3], load_list[:,4], load_list[:,5])
# energyspread = [(i*0.005+1)*particle_energy for i in range(-4,6)]
# for energy in energyspread:
#     p = MyParticle(energy, uranium_beam) #Create particle instance
#
#     load_list = p.loader(position_distribution=pos_dist, velocity_distribution=vel_dist, \
#                          num_of_particles=Np, sigma=sigma_list, temperature=temp_list)
#     #Add arrays to beam (x, y, z, vx, vy, vz)
#     uranium_beam.addparticles(load_list[:,0], load_list[:,1], load_list[:,2], \
#                               load_list[:,3], load_list[:,4], load_list[:,5])

#Write parameters to txt file
initial_bunchlength = max(load_list[:,2]) - min(load_list[:,2])
value_list = [particle_energy, uranium_beam.mass, \
              Np, temp_list[0], temp_list[1], \
              sigma_list[0], sigma_list[1], sigma_list[2], \
              wp.top.dt, initial_bunchlength]

parameterfile = open("parameters.txt", "w")
for value in value_list:
    parameterfile.write('{}'.format(value) + "\n")
    parameterfile.flush()
parameterfile.close()

#--Add tracer particle
tracked_uranium =  wp.Species(type=wp.Uranium,charge_state=+1,name="Beam species",weight=0)
tracked_uranium = TraceParticle(vz = np.sqrt(2*particle_energy*wp.jperev/uranium_beam.mass))

#--Add scraper at r=25mm
aperture_radius = 25*mm
def scrapebeam():
    rsq = uranium_beam.xp**2 + uranium_beam.yp**2
    uranium_beam.gaminv[rsq >= aperture_radius**2] = 0 #does not save particle, set 0 does.

wp.installparticlescraper(scrapebeam)
wp.top.lsavelostpart = 1 #save lost particles.



####################################################################
# Setup Diagnostics and make initial diagnostic plots

field_diagnostic_file_string = ('/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/Diagnostics/Fields/')

# #--The beam lives within a 25mm radius. It will be convenient to define the index
# #  for slicing the potential arrays for this region
# beam_index = int(25*mm*w3d.nx/w3d.xmmax) + 1 #index for slicing arrays to r = 25mm
#
# #--Field Line Plots
# Er, Ez = getselfe(comp='x'), getselfe(comp='z') #Grab E fields from warp
#
# #Create plots
# fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = (7,7))
# phi_plot, Ezplot, Erplot = axes[0], axes[1], axes[2]
#
# #phi plotting
# phi_plot.plot(z, getphi(ix=0)/kV, lw=1, label = r'$\Phi(z,r=0)$ [kV/m]')
# phi_plot.axhline(y=particle_energy/kV, c='r', linestyle = '--', lw=.75,
#                  label='P Particle Energy {:.5f} [kV]'.format(particle_energy/kV))
# phi_plot.axhline(y=max(getphi(ix=0))/kV, c='b', linestyle='--', lw=.75,
#                  label='Max Potential {:.5f} [kV]'.format(max(getphi(ix=0))/kV))
# phi_plot.legend(fontsize=8)
#
# phi_plot.set_title('Applied Electric Potential on Axis', fontsize=16)
# phi_plot.set_ylabel('Potential (KV)', fontsize=10)
#
# #Electric field plotting
# Ezr0 = Ez[0] #E_z at a r = 0
# Err0 = Er[1]/w3d.xmesh[1] #E_r/r two step sizes away from axis
#
# Ezplot.plot(z, Ezr0/kV, c='m', lw=1, label=r'$E_z(z,r=0)$ [kV/$m^2$]')
# Ezplot.axhline(y=0, c='k', lw=.5)
#
# Erplot.plot(z, Err0/kV, c='m', lw=1, label=r'$\frac{E_r(z,r)}{r}|_{\approx0}$ [kV/$m^3$]')
# Erplot.axhline(y=0, c='k', lw=.5)
#
# #Set Labels
# Ezplot.set_title(r"Applied $E_z$ on Axis", fontsize=16)
# Ezplot.set_ylabel(r'Electric Field [kV/$m^2$]', fontsize=10)
# Ezplot.legend(loc='upper center')
#
# Erplot.set_title(r"Applied $E_r/r$ on Axis", fontsize=16)
# Erplot.set_xlabel('z[m]', fontsize=12)
# Erplot.set_ylabel(r'Scaled Electric Field [kV/$m^3$]', fontsize=10)
# Erplot.legend(loc='upper center')
#
# plt.legend()
# plt.tight_layout()
# plt.savefig(field_diagnostic_file_string + 'E{:.6f}'.format(particle_energy/kV)
#             + 'Fields_on-axis.png', dpi=300)
# plt.show()
#
#
#
# #--Wire Plot
# #Get local data points
# zlocal = w3d.zmesh[int(w3d.nz/2):-1]
# #zlocal = w3d.zmesh
# rlocal = w3d.xmesh[0:beam_index +1]
# philocal = getphi()[0:beam_index+1,int(w3d.nz/2):-1]
#
# #Create plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# zwire = zlocal[int(len(zlocal)/2):-1]
# R, Z = np.meshgrid(rlocal, zwire)
#
# phiwire = philocal[::, int(len(zlocal)/2):-1]
#
# ax.plot_wireframe(R/mm, Z, phiwire.T/kV, rstride=8, cstride=8)
# ax.set_title("Local Applied Electric Potential in r-z ")
# ax.set_xlabel('r[mm]')
# ax.set_ylabel('z [m]')
# ax.set_zlabel('Potential [kV]')
# plt.savefig(field_diagnostic_file_string + 'potential_wireframe.png', dpi=400)
# plt.show()
#
#


#Filestring is useful for saving images.
dist_filestring = '/Users/nickvalverde/Dropbox/Research/ORISS'

#-- Create figure for each dist plot and plot.
#  It is import to create a separate figure for each or else data will mix into
#  other plots.
phaseplots = True
while phaseplots:
    fig,axes = plt.subplots(nrows=2, ncols=3)
    #--top row: position distributions
    #Bottom row: velocity distributions

    #Set position axes
    xzax = diagnostics.xzscatplot(axes[0,0], uranium_beam)
    yzax = diagnostics.yzscatplot(axes[0,1], uranium_beam)
    xyax = diagnostics.xyscatplot(axes[0,2], uranium_beam)


    #Set velocity axes
    vxvzax = diagnostics.vxvzscatplot(axes[1,0], uranium_beam)
    vyvzax = diagnostics.vyvzscatplot(axes[1,1], uranium_beam)
    vxvyax = diagnostics.vxvyscatplot(axes[1,2], uranium_beam)

    #Add Global title
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=3)
    #--phase-space plots
    vxxax = diagnostics.vxxscatplot(axes[0], uranium_beam)
    vyyax = diagnostics.vyyscatplot(axes[1], uranium_beam)
    vzzax = diagnostics.vzzscatplot(axes[2], uranium_beam)

    plt.tight_layout()
    plt.show()

    # rzplot = plt.figure(4)
    # plt.scatter(zpos_list/mm, rpos_list/mm, c = 'b', s = .5)
    # plt.title('r-z', fontsize = 16)
    # plt.xlabel("z[mm]", fontsize = 14)
    # plt.ylabel("r[mm]", fontsize = 14)
    # plt.tight_layout()
    # plt.show()
    #
    # vrzplot = plt.figure(8)
    # plt.scatter(zpos_list/mm, vr_list, s = .5)
    # plt.title('vr-z', fontsize = 16)
    # plt.xlabel("z[mm]", fontsize = 14)
    # plt.ylabel(r"$v_r$[m/s]", fontsize = 14)
    # plt.tight_layout()
    # plt.show()

    phaseplots = False

raise Exception()

#--Plot fields with warp
# winon() #Turn on window graphic
# pfzr(plotphi=1, plotselfe=0, contours = 50, cmin=0, cmax = 1000) #plot phi or E. Comp= component to plot
# limits(min(w3d.zmesh),max(w3d.zmesh), min(w3d.xmesh),max(w3d.xmesh))
# fma() #clear frame and send to cgm file.


####################################################################
# Generate Output files for post-process
####################################################################

#Create files
trajectoryfile = open("trajectoryfile.txt","w")  # saves marker trajectories
trackedfile = open("tracked_particle.txt", "w")


#Trajectory File
#Columns Particle, Iter, zp[i], uzp[i], xp[i], uxp[i]

#Test moment calculator in Warp
#top.ifzmmnt = 1 #Specifies global z moment calculates
#top.lspeciesmoments = True #Calcluates moment for each species separately and combined.
#top.lhist = True #saves histories for moment calculations


for i in range(0,uranium_beam.getz().size):
    trajectoryfile.write('{},{},{},{},{},{},{},{}'.format(i, 0, uranium_beam.zp[i], uranium_beam.uzp[i],
                                                    uranium_beam.xp[i], uranium_beam.uxp[i], \
                                                        uranium_beam.yp[i], uranium_beam.uyp[i]) + "\n")
    trajectoryfile.flush()

#Columns Iter, zp[i], uzp[i], xp[i], uxp[i], # lost particles
trackedfile.write('{},{},{},{},{},{},{},{}'.format(0, tracked_uranium.getz()[0], tracked_uranium.getvz()[0],
                                                    tracked_uranium.getx()[0], tracked_uranium.getvx()[0],
                                                    tracked_uranium.gety()[0], tracked_uranium.getvy()[0],
                                                    len(uranium_beam.getx(lost=1))) + "\n")

bounce_count = 0
iteration = 0
while bounce_count <=2:
#while iteration < 10:

    wp.step(1)  # advance particles
    sign_list = np.sign(tracked_uranium.getvz())

    if sign_list[iteration] != sign_list[iteration + 1]:
        bounce_count +=1
    else:
        pass


    #print(0.5*uranium_beam.mass*uranium_beam.getvz()**2/jperev)
    #print("beam velocity is: ", uranium_beam.getvz())
    iteration += 1



    trackedfile.write('{},{},{},{},{},{},{},{}'.format(iteration, tracked_uranium.getz()[iteration], tracked_uranium.getvz()[iteration],
                                                     tracked_uranium.getx()[iteration], tracked_uranium.getvx()[iteration],
                                                     tracked_uranium.gety()[iteration], tracked_uranium.getvy()[iteration],
                                                     len(uranium_beam.getx(lost=1))) + "\n")
    trackedfile.flush()

    for i in range(0,uranium_beam.getz().size):
        trajectoryfile.write('{},{},{},{},{},{},{},{}'.format(i, iteration, uranium_beam.zp[i], uranium_beam.uzp[i],
                                                        uranium_beam.xp[i], uranium_beam.uxp[i], \
                                                            uranium_beam.yp[i], uranium_beam.uyp[i]) + "\n")
        trajectoryfile.flush()

# Close files
trajectoryfile.close()
trackedfile.close()

print("--- %s seconds ---" % (time.time() - start_time))
print(uranium_beam.getz().size)
#prwall top.v

# Print run timing statistics
# printtimers()
