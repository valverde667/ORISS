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

particle_energy = .825*kV #Somehwere between .82 and .83

# load_list = []
# p = MyParticle(particle_energy, uranium_beam) #Create particle instance
# sigma_list = (1*mm, 1*mm, 5*mm)               #Standard deviations (sx, sy, sz)
# temp_list = (8.62e-5, 8.62e-5)                #[eV] corresponds to 1K
pos_dist = 'gaussian'                         #Distribution for position
vel_dist = 'gaussian'                         #Distribution for velocity
#load_list = p.loader(position_distribution = pos_dist, velocity_distribution = vel_dist, \
#                     num_of_particles = Np, sigma = sigma_list, temperature = temp_list)




#--Load particles onto beam
xcoord = [(i)*mm/10 for i in range(70)] #6mm Threshold for no particles lost
xcoord = np.arange(0,6,1)*mm
load_list = []
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
field_diagnostic_file_string = '/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/Diagnostics/Fields/'

#--The beam lives within a 25mm radius. It will be convenient to define the index
#  for slicing the potential arrays for this region
beam_index = int(25*mm*w3d.nx/w3d.xmmax) #index for slicing arrays to r = 25mm

#--Field Line Plots
fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (7,7))
Er, Ez = np.gradient(getphi()) #Compute Electric field Components


phi_plot, Eplot = axes[0], axes[1]
phi_plot.plot(z, getphi(ix=0)/kV, label = r'$\Phi(z,r=0)$')
phi_plot.axhline(y=particle_energy/kV, color = 'r', linestyle = '--', label = 'Initial Particle Energy')
phi_plot.set_title('Initial Electric Potential on Axis', fontsize = 16)
phi_plot.set_ylabel('Potential (KV)', fontsize = 14)

Ezr0 = Ez[0] #E_z at a r = 0
Erz0 = Er[0] #E_r at r = 0
Eplot.plot(z, Ezr0/kV, label = r'$E_z(z,r=0)$ [kV/m]')
Eplot.plot(z, Erz0/kV, label = r'$E_r(z, r=0)$ [kV/m]')

Eplot.set_title("Intial Electric Field on Axis", fontsize = 16)
Eplot.set_xlabel('z[m]', fontsize = 14)
Eplot.set_ylabel('Electric Field [kV/m]', fontsize = 14)
plt.legend()
plt.tight_layout()
plt.savefig(field_diagnostic_file_string + 'Fields_on-axis.png', dpi=300)


#--Global Field Contour Plots
fig,axes = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize=(8,8))
phiax = axes[0]
Eax = axes[1]
phiax.set_title('Electric Potential Contours in r-z', fontsize = 16)
phiax.set_ylabel('r [m]', fontsize = 14)
phicnt = phiax.contour(w3d.zmesh, w3d.xmesh, getphi(), cmap=cm.hsv, linewidths = .7)
phicb = fig.colorbar(phicnt, shrink= 0.8, ax = phiax, label = r'$\Phi(z,r)$ [kV]')

#p = ax.pcolor(w3d.zmesh/mm, w3d.xmesh/mm, getphi(), cmap=cm.gray, vmin=abs(getphi()).min(), vmax=abs(getphi()).max())
#cb = fig.colorbar(p, shrink = 0.8, ax=ax)


Ezcnt = Eax.contour(w3d.zmesh, w3d.xmesh, Ez/kV, cmap = cm.hsv, linewidths = .7)
Ecb = fig.colorbar(Ezcnt, shrink = 0.8, ax = Eax, label = r'$E_z(z,r)$ [kV/m]')

Eax.set_title("On-axis Electric Field Contours", fontsize = 16)
Eax.set_xlabel('z [m]', fontsize = 14)
Eax.set_ylabel('r [m]', fontsize = 14)

#l, b, w, h = ax.get_position().bounds
#-Adjust gray color bar
#ll, bb, ww, hh = cb.ax.get_position().bounds
#cb.ax.set_position([ll, b + 0.1*h, ww, h*0.8])

plt.tight_layout()
plt.show()
plt.savefig(field_diagnostic_file_string + 'global_contour_E-V_fields.png', dpi=400)

#--Local Field Contour Plots
phi_local =  getphi()[0:beam_index,0:int(w3d.nz/2)+1]
#levels = np.linspace(0, phi_local.max()/kV, 101)
levels = np.linspace(0, .7, 101)
fig,axes = plt.subplots(nrows = 2, ncols = 1, figsize =(8,8), sharex=True)
phiax = axes[0]
Ezax = axes[1]
phiax.set_title('Electric Potential Contours in r-z', fontsize = 16)
phiax.set_ylabel('r[mm]', fontsize = 14)
#p = ax.pcolor(w3d.zmesh/mm, w3d.xmesh/mm, getphi(), cmap=cm.gray, vmin=abs(getphi()).min(), vmax=abs(getphi()).max())
#cb = fig.colorbar(p, shrink = 0.8, ax=ax)

#Plot contour for r = [0:25mm] and z = [-650mm:0]
Ez_local = np.gradient(getphi()[0:beam_index,0:int(w3d.nz/2)+1])[1]

phicnt = phiax.contour(w3d.zmesh[0:int(w3d.nz/2)+1], w3d.xmesh[0:beam_index],  getphi()[0:beam_index,0:int(w3d.nz/2)+1], levels = levels,
                 cmap=cm.hsv, linewidths = .7)
phicb = fig.colorbar(phicnt, shrink= 0.8, ax = phiax, label = r'$\Phi(z,r)$ [kV]')

Ezcnt = Ezax.contour(w3d.zmesh[0:int(w3d.nz/2)+1], w3d.xmesh[0:beam_index], Ez_local/kV,
                     cmap=cm.hsv, linewidths = .7)
Ezcb = fig.colorbar(Ezcnt, shrink = 0.8, ax = Ezax, label = r'$/Ez(r,z)$ [kV/m]')

Ezax.set_xlabel('z[m]', fontsize = 14)
Ezax.set_ylabel('r[m]', fontsize = 14)
Ezax.set_title('Electric Field Contours in r-z', fontsize = 16)
#l, b, w, h = ax.get_position().bounds
#-Adjust gray color bar
#ll, bb, ww, hh = cb.ax.get_position().bounds
#cb.ax.set_position([ll, b + 0.1*h, ww, h*0.8])

plt.tight_layout()
plt.show()
plt.savefig(field_diagnostic_file_string + 'local_contour_E-V_fields.png', dpi=400)


#--Plot fields with warp
# winon() #Turn on window graphic
# pfzr(plotphi=1, plotselfe=0, contours = 50, cmin=0, cmax = 1000) #plot phi or E. Comp= component to plot
# limits(min(w3d.zmesh),max(w3d.zmesh), min(w3d.xmesh),max(w3d.xmesh))
# fma() #clear frame and send to cgm file.
raise Exception()


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
top.ifzmmnt = 1 #Specifies global z moment calculates
top.lspeciesmoments = True #Calcluates moment for each species separately and combined.
top.lhist = True #saves histories for moment calculations


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
while bounce_count <= 1:
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
