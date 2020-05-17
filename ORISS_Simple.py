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
uranium_beam = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0)
               #weight = 0 no spacecharge, 1 = spacecharge.
               #Both go through Poisson sovler.

top.lbeamcom = False #Specify grid does not move to follow beam center
                     #of mass as appropriate for trap simulation



####################################################################
# Generate Code
####################################################################

package("w3d") # Specify 3D code.  But fieldsolver will be r-z and particles will deposit on r-z mesh
generate()     # Initate code, this will also make an initial fieldsolve

solver.ldosolve = False #This sets up the field solver with the beam fields.
                     #particles are treated as interacting. False, turns off space charge

particle_energy = 59.85
z = w3d.zmesh
# load_list = []
# p = MyParticle(particle_energy, uranium_beam) #Create particle instance
# sigma_list = (1*mm, 1*mm, 5*mm)               #Standard deviations (sx, sy, sz)
# temp_list = (8.62e-5, 8.62e-5)                #[eV] corresponds to 1K
pos_dist = 'gaussian'                         #Distribution for position
vel_dist = 'gaussian'                         #Distribution for velocity
#load_list = p.loader(position_distribution = pos_dist, velocity_distribution = vel_dist, \
#                     num_of_particles = Np, sigma = sigma_list, temperature = temp_list)

#--Load particles onto beam

# #Transverse Tuning
xcoord = np.linspace(-.1,.1,9)*mm
p = MyParticle(particle_energy, uranium_beam)

for elem in xcoord:
    load_coords = p.loader(position_distribution=pos_dist,
                           velocity_distribution=vel_dist,
                           avg_coordinates=(elem, 0, 0))

    xpos,ypos,zpos = load_coords[0][0], load_coords[0][1], load_coords[0][2]
    vxpos,vypos,vzpos = load_coords[0][3], load_coords[0][4], load_coords[0][5]

    uranium_beam.addparticles(xpos,ypos,zpos,vxpos,vypos,vzpos)


# #Z-tuning
# energyspread = [(i*0.005+1)*particle_energy for i in range(-4,5)]
# for energy in energyspread:
#     p = MyParticle(energy, uranium_beam)
#     load_coords = p.loader()
#     xpos,ypos,zpos = load_coords[0][0], load_coords[0][1], load_coords[0][2]
#     vxpos,vypos,vzpos = load_coords[0][3], load_coords[0][4], load_coords[0][5]
#     uranium_beam.addparticles(xpos,ypos,zpos,vxpos,vypos,vzpos)


#--Add tracer particle
tracked_uranium =  Species(type=Uranium,charge_state=+1,name="Beam species",weight=0)
tracked_uranium = TraceParticle(vz = np.sqrt(2*particle_energy*jperev/uranium_beam.mass))


#top.prwall = 25*mm #Could not get to work. Loaded particles well out of 25mm and they survived.
aperture_radius = 25*mm
def scrapebeam():
    rsq = uranium_beam.xp**2 + uranium_beam.yp**2
    uranium_beam.gaminv[rsq >= aperture_radius**2] = 0 #does not save particle, set 0 does.

#target = ZCylinder(25*mm,2,zcent=0)
#scraper = ParticleScraper(conductors = [target], lcollectlpdata=1)
installparticlescraper(scrapebeam)
top.lsavelostpart = 1



####################################################################
# Setup Diagnostics and make initial diagnostic plots
#
#  Make nice initial field diagnostic plots with labels.  Make plots of:
#   1) phi(r=0,z) vs z in volts
#   1a) repeat 1) and put the inital bunch energy on in units of volts
field_diagnostic_file_string = ('/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/Diagnostics/Fields/')

#--The beam lives within a 25mm radius. It will be convenient to define the index
#  for slicing the potential arrays for this region
beam_index = int(25*mm*w3d.nx/w3d.xmmax) + 1 #index for slicing arrays to r = 25mm

#--Field Line Plots
Er, Ez = getselfe(comp='x'), getselfe(comp='z') #Grab E fields from warp

#Create plots
fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize = (7,7))
phi_plot, Ezplot, Erplot = axes[0], axes[1], axes[2]

#phi plotting
phi_plot.plot(z, getphi(ix=0)/kV, lw=1, label = r'$\Phi(z,r=0)$ [kV/m]')
phi_plot.axhline(y=particle_energy/kV, c='r', linestyle = '--', lw=.75,
                 label='P Particle Energy {:.5f} [kV]'.format(particle_energy/kV))
phi_plot.axhline(y=max(getphi(ix=0))/kV, c='b', linestyle='--', lw=.75,
                 label='Max Potential {:.5f} [kV]'.format(max(getphi(ix=0))/kV))
phi_plot.legend(fontsize=8)

phi_plot.set_title('Applied Electric Potential on Axis', fontsize=16)
phi_plot.set_ylabel('Potential (KV)', fontsize=10)

#Electric field plotting
Ezr0 = Ez[0] #E_z at a r = 0
Err0 = Er[1]/w3d.xmesh[1] #E_r/r two step sizes away from axis

Ezplot.plot(z, Ezr0/kV, c='m', lw=1, label=r'$E_z(z,r=0)$ [kV/$m^2$]')
Ezplot.axhline(y=0, c='k', lw=.5)

Erplot.plot(z, Err0/kV, c='m', lw=1, label=r'$\frac{E_r(z,r)}{r}|_{\approx0}$ [kV/$m^3$]')
Erplot.axhline(y=0, c='k', lw=.5)

#Set Labels
Ezplot.set_title(r"Applied $E_z$ on Axis", fontsize=16)
Ezplot.set_ylabel(r'Electric Field [kV/$m^2$]', fontsize=10)
Ezplot.legend(loc='upper center')

Erplot.set_title(r"Applied $E_r/r$ on Axis", fontsize=16)
Erplot.set_xlabel('z[m]', fontsize=12)
Erplot.set_ylabel(r'Scaled Electric Field [kV/$m^3$]', fontsize=10)
Erplot.legend(loc='upper center')

plt.legend()
plt.tight_layout()
plt.savefig(field_diagnostic_file_string + 'E{:.6f}'.format(particle_energy/kV)
            + 'Fields_on-axis.png', dpi=300)
plt.show()



# #--Global/Local Field Contour Plots
# numcntrs = 25 #Set number of contour lines.
# #Get local data points
# zlocal = w3d.zmesh[int(w3d.nz/2):-1]
# #zlocal = w3d.zmesh
# rlocal = w3d.xmesh[0:beam_index +1]
# philocal = getphi()[0:beam_index+1,int(w3d.nz/2):-1]
#
#
#
# #Create plot
# fig,axes = plt.subplots(nrows = 2, ncols = 1, figsize=(8,8))
# phiax = axes[0]
# philocalax = axes[1]
#
# #Plot Globals Field
# phicnt = phiax.contour(w3d.zmesh, w3d.xmesh, getphi()/kV,
#                        cmap=cm.hsv, linewidths=.5)
# phicb = fig.colorbar(phicnt, shrink=0.8, ax=phiax, label=r'$\Phi(z,r)$ [kV]')
#
# phiax.set_title('Global Applied Electric Potential in r-z', fontsize=16)
# phiax.set_xlabel('z[m]', fontsize=14)
# phiax.set_ylabel('r[m]', fontsize=14)
#
#
# #Plot local field
# locallvls = np.linspace(0, max(philocal[0])/kV, 25)
# philocalcnt = philocalax.contour(zlocal, rlocal, philocal/kV,
#                     levels=locallvls, cmap=cm.hsv, linewidths=.5)
# philocalcb= fig.colorbar(philocalcnt, shrink=0.8, ax=philocalax,
#                          label=r'$\Phi(z,r)$ [kV]')
#
# philocalax.set_title("Local Applied Electric Potential in r-z", fontsize=16)
# philocalax.set_xlabel('z [m]', fontsize=14)
# philocalax.set_ylabel('r [m]', fontsize=14)
#
# #p = ax.pcolor(w3d.zmesh/mm, w3d.xmesh/mm, getphi(), cmap=cm.gray, vmin=abs(getphi()).min(), vmax=abs(getphi()).max())
# #cb = fig.colorbar(p, shrink = 0.8, ax=ax)
#
# #l, b, w, h = ax.get_position().bounds
# #-Adjust gray color bar
# #ll, bb, ww, hh = cb.ax.get_position().bounds
# #cb.ax.set_position([ll, b + 0.1*h, ww, h*0.8])
#
# plt.tight_layout()
# plt.show()
# plt.savefig(field_diagnostic_file_string + 'potential_countours.png', dpi=400)
#
#
# #--Wire Plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# zwire = zlocal[int(len(zlocal)/2):-1]
# R, Z = np.meshgrid(rlocal, zwire)
#
# phiwire = philocal[::, int(len(zlocal)/2):-1]
# # Plot a basic wireframe.
#
# ax.plot_wireframe(R/mm, Z, phiwire.T/kV, rstride=8, cstride=8)
# ax.set_title("Local Applied Electric Potential in r-z ")
# ax.set_xlabel('r[mm]')
# ax.set_ylabel('z [m]')
# ax.set_zlabel('Potential [kV]')
# plt.savefig(field_diagnostic_file_string + 'potential_wireframe.png', dpi=400)
# plt.show()


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
parameterfile = open("parameters.txt", "w")
variable_list = ["particle_energy", "mass", "time_step", "Np_initial", "zcentr8", "zcentr7", \
                    "zcentr6", "zcentr5", "zcentr4", "zcentr3", "zcentr2", "zcentr1"]

value_list = [particle_energy, uranium_beam.mass, top.dt, zcentr8, zcentr7, zcentr6, \
              zcentr5, zcentr4, zcentr3, zcentr2, zcentr1]

for value in value_list:
    parameterfile.write('{}'.format(value) + "\n")
    parameterfile.flush()
parameterfile.close()


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
trackedfile.write('{},{},{},{},{},{}'.format(0, tracked_uranium.getz()[0], tracked_uranium.getvz()[0],
                                                    tracked_uranium.getx()[0], tracked_uranium.getvx()[0],
                                                    len(uranium_beam.getx(lost=1))) + "\n")

bounce_count = 0
iteration = 0
while bounce_count <=1:
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



    trackedfile.write('{},{},{},{},{},{}'.format(iteration, tracked_uranium.getz()[iteration], tracked_uranium.getvz()[iteration],
                                                     tracked_uranium.getx()[iteration], tracked_uranium.getvx()[iteration],
                                                     len(uranium_beam.getx(lost=1))) + "\n")
    trackedfile.flush()

    for i in range(0,uranium_beam.getz().size):
        trajectoryfile.write('{},{},{},{},{},{}'.format(i, iteration, uranium_beam.zp[i], uranium_beam.uzp[i],
                                                        uranium_beam.xp[i], uranium_beam.uxp[i], \
                                                            uranium_beam.yp[i], uranium_beam.uyp[i]) + "\n")
        trajectoryfile.flush()

# Close files
trajectoryfile.close()
trackedfile.close()

print("--- %s seconds ---" % (time.time() - start_time))

#prwall top.v

# Print run timing statistics
# printtimers()
