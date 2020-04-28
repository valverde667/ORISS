#First version with actual mass spectrometry
#Using Uranium (238) and Neptunium (237)
#IMPORTANT DO NOT NAME ANYTHING LOAD, this interrupts with a declared functionion Species.py
from warp import *
from warp.particles.singleparticle import TraceParticle
from Forthon import *
from Particle_Class import *
from fill_ellipse import *
import matplotlib.pyplot as plt
import numpy as np

import time
start_time = time.time()

print("--- %s seconds ---" % (time.time() - start_time))


####################################################################
# Simulation Mesh, create in w3d but run field solve in r-z
####################################################################

exec(open("ORISS_Geometry.py").read()) #Create simulation mesh



####################################################################
# Particle Moving and Species
####################################################################

top.dt     = 1e-7 # Timestep of particle advance

#Set particle boundary conditions at mesh ends
top.pbound0 = absorb  #boundary condition at iz == 0
top.pboundnz = absorb #boundary condition at iz == nz
top.pboundxy = absorb #boundary condition at edge or radial mesh


#Create Species of particles to advance
uranium_beam = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0) #weight = 0 no spacecharge, 1 = spacecharge. Both go through Poisson sovler.

top.lbeamcom = False # Specify grid does not move to follow beam center of mass as appropriate for trap simulation



####################################################################
# Generate Code
####################################################################

package("w3d") # Specify 3D code.  But fieldsolver will be r-z and particles will deposit on r-z mesh
generate()     # Initate code, this will also make an initial fieldsolve



solver.ldosolve = False #This sets up the field solver with the beam fields.
                     #particles are treated as interacting. False, turns off space charge
limits(-1.,1.)
z = linspace(-1.,1.,w3d.nz+1)

particle_energy = 2.77*kV #2.77 kV is good for both point-point and ||-point

# load_list = []
# p = MyParticle(particle_energy, uranium_beam) #Create particle instance
# sigma_list = (1*mm, 1*mm, 5*mm)               #Standard deviations (sx, sy, sz)
# temp_list = (8.62e-5, 8.62e-5)                #[eV] corresponds to 1K
pos_dist = 'gaussian'                         #Distribution for position
vel_dist = 'gaussian'                         #Distribution for velocity
#load_list = p.loader(position_distribution = pos_dist, velocity_distribution = vel_dist, \
#                     num_of_particles = Np, sigma = sigma_list, temperature = temp_list)





 # =============================================================================
#-- Create Distribution Plots
#
# xpos_list = []
# vx_list = []
# ypos_list = []
# vy_list =[]
# zpos_list = []
# vz_list = []
#
# for elem in load_list:
#     xpos_list.append(elem[0])
#     vx_list.append(elem[3])
#
#     ypos_list.append(elem[1])
#     vy_list.append(elem[4])
#
#     zpos_list.append(elem[2])
#     vz_list.append(elem[5])
#
# xpos_list = np.array(xpos_list)
# ypos_list = np.array(ypos_list)
# zpos_list = np.array(zpos_list)
#
# vx_list = np.array(vx_list)
# vy_list = np.array(vy_list)
# vz_list = np.array(vz_list)
#
# rpos_list = np.sqrt(xpos_list**2 + ypos_list**2)
#
#
# filestring = '/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/'
# plt.scatter(zpos_list/mm,xpos_list/mm, s = .5)
# plt.title('x-z', fontsize = 16)
# plt.xlabel("z[mm]", fontsize = 14)
# plt.ylabel("x[mm]", fontsize = 14)
# plt.tight_layout()
# plt.savefig(filestring + 'x-z.png', dpi=300)
# plt.show()
#
# plt.scatter(zpos_list/mm,ypos_list/mm, s = .5)
# plt.title('y-z', fontsize = 16)
# plt.xlabel("z[mm]", fontsize = 14)
# plt.ylabel("y[mm]", fontsize = 14)
# plt.tight_layout()
# plt.savefig(filestring + 'y-z.png', dpi=300)
# plt.show()
#
# plt.scatter(xpos_list/mm, ypos_list/mm, s = .5)
# plt.title('x-y', fontsize = 16)
# plt.xlabel("x[mm]", fontsize = 14)
# plt.ylabel("y[mm]", fontsize = 14)
# plt.tight_layout()
# plt.savefig(filestring + 'y-x.png', dpi=300)
# plt.show()
#
# filestring = '/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/'
# fig, ax = plt.subplots(figsize = (7,7))
# ax.scatter(zpos_list/mm, rpos_list/mm, c = 'b', s = .5)
# ax.scatter(zpos_list/mm, -rpos_list/mm,c = 'b', s = .5)
# ax.set_title('r-z', fontsize = 16)
# ax.set_xlabel("z[mm]", fontsize = 14)
# ax.set_ylabel("r[mm]", fontsize = 14)
# plt.tight_layout()
# plt.savefig(filestring + 'r-z.png', dpi=300)
# plt.show()
#
#
#
#
# plt.scatter(xpos_list/mm, vx_list, s = .5)
# plt.title('vx-x', fontsize = 16)
# plt.xlabel("x [m]", fontsize = 14)
# plt.ylabel(r"$v_x$[m/s]", fontsize = 14)
# plt.tight_layout()
# plt.savefig(filestring + 'vx-x.png', dpi=300)
# plt.show()
#
#
# plt.scatter(ypos_list/mm, vy_list, s = .5)
# plt.title('vy-y', fontsize = 16)
# plt.xlabel("y[mm]", fontsize = 14)
# plt.ylabel(r"$v_y$[m/s]", fontsize = 14)
# plt.tight_layout()
# plt.savefig(filestring + 'vy-y.png', dpi=300)
# plt.show()
#
#
# plt.scatter(zpos_list/mm, vz_list, s = .5)
# plt.title('vz-z', fontsize = 16)
# plt.xlabel("z[mm]", fontsize = 14)
# plt.ylabel(r"$v_z$[m/s]", fontsize = 14)
# plt.tight_layout()
# plt.savefig(filestring + 'vz-z.png', dpi=300)
# plt.show()
#
#
# plt.scatter(vz_list, vx_list, s = .5)
# plt.title('vx-vz', fontsize = 16)
# plt.xlabel(r"$v_z$[m/s]", fontsize = 14)
# plt.ylabel(r"$v_x$[m/s]", fontsize = 14)
# plt.tight_layout()
# plt.savefig(filestring + 'vx-vz.png', dpi=300)
# plt.show()
#
# plt.scatter(vz_list, vy_list, s = .5)
# plt.title('vy-vz', fontsize = 16)
# plt.xlabel(r"$v_z$[m/s]", fontsize = 14)
# plt.ylabel(r"$v_y$[m/s]", fontsize = 14)
# plt.tight_layout()
# plt.savefig(filestring + 'vy-vz.png', dpi=300)
# plt.show()
#
# plt.scatter(vx_list, vy_list, s = .5)
# plt.title('vy-vx', fontsize = 16)
# plt.xlabel(r"$v_x$[m/s]", fontsize = 14)
# plt.ylabel(r"$v_y$[m/s]", fontsize = 14)
# plt.tight_layout()
# plt.savefig(filestring + 'vy-vx.png', dpi=300)
# plt.show()
# =============================================================================


#--Load particles onto beam
xcoord = [(i+1)*mm/10 for i in range(200)]
load_list = []
Np = len(xcoord)
p = MyParticle(particle_energy, uranium_beam)

for xpos in xcoord:
    load_list.append(p.loader(position_distribution = pos_dist, velocity_distribution = vel_dist, \
    avg_coordinates = (xpos, 0, 0)))
        
    load_list.append(p.loader(position_distribution = pos_dist, velocity_distribution = vel_dist, \
    avg_coordinates = (-xpos, 0, 0)))


for array in load_list:
      elem = array[0]
      xpos,ypos,zpos = elem[0], elem[1], elem[2]
      vxpos, vypos, vzpos = elem[3], elem[4], elem[5]

      uranium_beam.addparticles(xpos,ypos,zpos,vxpos,vypos,vzpos)

#--Add tracer particle
tracked_uranium =  Species(type=Uranium,charge_state=+1,name="Beam species",weight=0)
tracked_uranium = TraceParticle(vz = np.sqrt(2*particle_energy*jperev/uranium_beam.mass))


#top.prwall = 25*mm
aperture_radius = 25*mm
def scrapebeam():
    rsq = uranium_beam.xp**2 + uranium_beam.yp**2
    uranium_beam.gaminv[rsq >= aperture_radius**2] = -1 #does not save particle, set 0 does.

installparticlescraper(scrapebeam)





####################################################################
# Setup Diagnostics and make initial diagnostic plots
#
#  Make nice initial field diagnostic plots with labels.  Make plots of:
#   1) phi(r=0,z) vs z in volts
#   1a) repeat 1) and put the inital bunch energy on in units of volts

fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (7,7))

ax1, ax2 = axes[0], axes[1]
ax1.plot(z, getphi(ix=0)/kV, label = 'Potential at r = 0')
ax1.axhline(y=particle_energy/kV, color = 'r', linestyle = '--', label = 'Initial Particle Energy')
ax1.set_title('Initial Potential on Axis vs z ')
ax1.set_ylabel('Potential (KV)')
ax1.legend()
    #   2) E_z vs z labeled
    #   3) (E_r/r)|_r = 0 vs z labeled ... transverse field gradient vs z

ax2.plot(z,np.gradient(getphi(ix=0)), label = 'Transverse Field Gradient')
ax2.axhline(y = 0, color = 'k', lw = .5)
ax2.set_ylabel('Electric Field (V/m)')
ax2.set_xlabel('Longitudinal Position (m)')
ax2.legend()

plt.tight_layout()
plt.savefig('/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/initial_potential.png', dpi=300)

####################################################################



####################################################################
# Generate Output files for post-process
####################################################################

#Create files
trajectoryfile = open("trajectoryfile.txt","w")  # saves marker trajectories
trackedfile = open("tracked_particle.txt", "w")



#Trajectory File
#Columns Particle, Iter, zp[i], uzp[i], xp[i], uxp[i]
for i in range(0,uranium_beam.getz().size):
    trajectoryfile.write('{},{},{},{},{},{}'.format(i, 0, uranium_beam.zp[i], uranium_beam.uzp[i],
                                                    uranium_beam.xp[i], uranium_beam.uxp[i]) + "\n")
    trajectoryfile.flush()

trackedfile.write('{},{},{},{},{}'.format(0, tracked_uranium.getz()[0], tracked_uranium.getvz()[0],
                                                    tracked_uranium.getx()[0], tracked_uranium.getvx()[0]) + "\n")

#Stdev File
bounce_count = 0
iteration = 0
while bounce_count < 3:
#while iteration < 10:

    step(1)  # advance particles
    sign_list = np.sign(tracked_uranium.getvz())

    if sign_list[iteration] != sign_list[iteration + 1]:
        bounce_count +=1
    else:
        pass


    #print(0.5*uranium_beam.mass*uranium_beam.getvz()**2/jperev)
    #print("beam velocity is: ", uranium_beam.getvz())
    iteration += 1



    trackedfile.write('{},{},{},{},{}'.format(iteration, tracked_uranium.getz()[iteration], tracked_uranium.getvz()[iteration],
                                                     tracked_uranium.getx()[iteration], tracked_uranium.getvx()[iteration]) + "\n")
    trackedfile.flush()

    for i in range(0,uranium_beam.getz().size):
        trajectoryfile.write('{},{},{},{},{},{}'.format(i, iteration, uranium_beam.zp[i], uranium_beam.uzp[i],
                                                        uranium_beam.xp[i], uranium_beam.uxp[i]) + "\n")
        trajectoryfile.flush()

# Close files
trajectoryfile.close()
trackedfile.close()


parameterfile = open("parameters.txt", "w")
variable_list = ["particle_energy", "mass", "time_step", "Np_initial", "zcentr8", "zcentr7", \
                    "zcentr6", "zcentr5", "zcentr4", "zcentr3", "zcentr2", "zcentr1"]

value_list = [particle_energy, uranium_beam.mass, top.dt, Np, zcentr8, zcentr7, zcentr6, \
              zcentr5, zcentr4, zcentr3, zcentr2, zcentr1]

for value in value_list:
    parameterfile.write('{}'.format(value) + "\n")

parameterfile.close()

print("--- %s seconds ---" % (time.time() - start_time))

#prwall top.v

# Print run timing statistics
# printtimers()
