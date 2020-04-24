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

setup()  # setup graphics etc.

####################################################################
# Simulation Mesh, create in w3d but run field solve in r-z
####################################################################

# x-mesh (radial in r-z)
w3d.xmmax =   +.1 # Upper limit of mesh r_max
w3d.xmmin =    0. # Lower limit of mesh r_min (should be zero for r-z)
#w3d.nx    =   4096. # Mesh points are 0,...,nx
w3d.nx    = 1024      ### less fine mesh

# y-mesh (do not define in r-z simulations)
#w3d.ymmax =   +.1 # Upper limit of mesh
#w3d.ymmin =   -.1 # Lower limit of mesh
#w3d.ny    =   128 # Mesh points are 0,...,ny

# z-mesh
w3d.zmmax =   +1. # Upper limit of mesh
w3d.zmmin =   -1. # Lower limit of mesh
#w3d.nz    =   4096 # Mesh points are 0,...,nz
w3d.nz    = 1024   ### less fine

####################################################################
# Field Solve
####################################################################

w3d.solvergeom = w3d.RZgeom # Setting Solver Geometry. In this case this is
                            # a Poisson solver in 2-D rz-geometry

##Set boundary conditions

#Boundary conditions for mesh
w3d.bound0  = dirichlet # for iz == 0
w3d.boundnz = dirichlet # for iz == nz
w3d.boundxy = dirichlet #in all transverse directions


f3d.mgmaxiters = 1000 #  Max iterations of multigrid field solve

#solver=MultiGrid3D() # 3-D field solver (need to define x-mesh and y-mesh consistent)
solver=MultiGridRZ() # multi-grid Poisson solver in 2-D r-z geometry


#child1=solver.addchild(mins=[0.,0.,-1.],maxs=[.01,0.,.1],refinement=[100,1,100]  # Define mesh refinement

solver.mgverbose = +1 #cannot find what these settings do.
solver.mgtol = 1.e-4  # Absolute tolerance (convergance) of field solver in potential [Volts]

registersolver(solver) # register field solver so pic cycle uses


####################################################################
# Define Conductors to Load on Mesh
####################################################################

# Geometry parameters for rings and cones in ORISS device

gap = 4.51*mm  # insulating gap between rings and ring/cone etc
zsrf = [-.04499,0.0,.04499]  # downstream, upstream z-extent ring
rsrf = [0.071,0.025,.071] #coordinates for creating cone of revolution
Rmin = 70.*mm # Inner radius of electrode (center to inner interface) annulus.
RminIF = 60.*mm# # Inner radius of chamber before drift space
Rmax = 71.*mm # Outer radius of electrode (center to outer interface) annulus
RmaxIF = 61.*mm # Outer radius of chamber before drift space.
Length = 27.98*mm
LengthIF = 43.05*mm
Cone = 44.99*mm
drift = 430.68*mm

# Voltages of rings and cone number from left to right on rings from end
Vcone = +0
V00   = +0000
V01   = -0
#V01  = -0
V02   = 0
V03   = 0.
V04   = 0.
V05   = +00
V06   = +00
V07   = +00
V08   = +8.*kV

# z-centers of right rings
zcentr8=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+7*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr7=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+6*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr6=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+5*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr5=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+4*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr4=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+3*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr3=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+2*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr2=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+1*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr1=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+0*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)

# z-center of right cone
zcentrcone=Cone+gap+LengthIF+drift+LengthIF+gap+Cone-(Cone+gap+LengthIF+drift/2)

# central pipe segment
zcentrIF=Cone+gap+LengthIF+drift+LengthIF/2.-(Cone+gap+LengthIF+drift/2)
zcentdrift = Cone+gap+LengthIF+drift/2-(Cone+gap+LengthIF+drift/2)
zcentlIF=Cone+gap+LengthIF/2-(Cone+gap+LengthIF+drift/2)

# z-centers of left cone
zcentlcone=0.-(Cone+gap+LengthIF+drift/2)

# z-centers of left rings
zcentl1=-Cone-gap-0.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl2=-Cone-gap-1.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl3=-Cone-gap-2.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl4=-Cone-gap-3.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl5=-Cone-gap-4.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl6=-Cone-gap-5.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl7=-Cone-gap-6.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl8=-Cone-gap-7.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)


# Conductors for right rings with bias voltages, placed by z-centers
zr8=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V08,zcent=zcentr8)
zr7=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V07,zcent=zcentr7)
zr6=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V06,zcent=zcentr6)
zr5=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V05,zcent=zcentr5)
zr4=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V04,zcent=zcentr4)
zr3=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V03,zcent=zcentr3)
zr2=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V02,zcent=zcentr2)
zr1=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V01,zcent=zcentr1)

# Conductor for right cone with bias voltage, placed by z-center
hrcone = ZSrfrv(rsrf=rsrf,zsrf=zsrf,voltage=Vcone,zcent=zcentrcone)

zr0 = ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=LengthIF,voltage=V00,zcent=zcentrIF)

zdrift = ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=drift,voltage=V00,zcent=zcentdrift)

# Conductor for right cone with bias voltage, placed by z-center
hlcone = ZSrfrv(rsrf=rsrf,zsrf=zsrf,voltage=Vcone,zcent=zcentlcone)

zl0 = ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=LengthIF,voltage=V00,zcent=zcentlIF)

# Conductors for left rings with bias voltages, placed by z-centers
zl1=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V01,zcent=zcentl1)
zl2=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V02,zcent=zcentl2)
zl3=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V03,zcent=zcentl3)
zl4=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V04,zcent=zcentl4)
zl5=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V05,zcent=zcentl5)
zl6=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V06,zcent=zcentl6)
zl7=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V07,zcent=zcentl7)
zl8=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V08,zcent=zcentl8)


#--Install conductors on mesh.  These are placed with subgrid precision
installconductor(zr8)
installconductor(zr7)
installconductor(zr6)
installconductor(zr5)
installconductor(zr4)
installconductor(zr3)
installconductor(zr2)
installconductor(zr1)
installconductor(hrcone)
installconductor(zr0)
installconductor(zdrift)
installconductor(zl0)
installconductor(hlcone)
installconductor(zl1)
installconductor(zl2)
installconductor(zl3)
installconductor(zl4)
installconductor(zl5)
installconductor(zl6)
installconductor(zl7)
installconductor(zl8)


####################################################################
# Particle Moving and Species
####################################################################
target1 = ZCylinder(radius = 25*mm, length = 600*mm, zcent=0)
target2 = ZCylinder(radius = 25*mm, length = -600*mm, zcent=0)

top.dt     = 1e-7 # Timestep of particle advance
desired_run_time = .001 #s
run_time = desired_run_time/top.dt

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

# xbanana_list = np.arange(1,10,1)
# p = MyParticle(particle_energy, uranium_beam) #Create particle instance
# load_list = []
# for elem in xbanana_list:
#     load = p.loader(position_distribution = pos_dist, velocity_distribution = vel_dist, \
#     avg_coordinates = (elem, 0, 0))
#     load_list.append(load)




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
xcoord = [i*mm for i in range(20)]
load_list = []
p = MyParticle(particle_energy, uranium_beam)

for xpos in xcoord:
    load_list.append(p.loader(position_distribution = pos_dist, velocity_distribution = vel_dist, \
    avg_coordinates = (xpos, 0, 0)))


for array in load_list:
      elem = array[0]
      xpos,ypos,zpos = elem[0], elem[1], elem[2]
      vxpos, vypos, vzpos = elem[3], elem[4], elem[5]
      
      uranium_beam.addparticles(xpos,ypos,zpos,vxpos,vypos,vzpos)

#Add tracer particle
tracked_uranium =  Species(type=Uranium,charge_state=+1,name="Beam species",weight=0)
tracked_uranium = TraceParticle(vz = np.sqrt(2*particle_energy*jperev/uranium_beam.mass))








####################################################################
# Setup Diagnostics and make initial diagnostic plots
#
#  Make nice initial field diagnostic plots with labels.  Make plots of:
#   1) phi(r=0,z) vs z in volts
#   1a) repeat 1) and put the inital bunch energy on in units of volts

fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (7,7))

ax1, ax2 = axes[0], axes[1]
ax1.plot(z, getphi(ix=0), label = 'Potential at r = 0')
ax1.axhline(y=particle_energy, color = 'r', linestyle = '--', label = 'Initial Particle Energy')
ax1.set_title('Initial Potential on Axis vs z ')
ax1.set_ylabel('Potential (V)')
ax1.legend()
    #   2) E_z vs z labeled
    #   3) (E_r/r)|_r = 0 vs z labeled ... transverse field gradient vs z

ax2.plot(z,np.gradient(getphi(ix=0)), label = 'Transverse Field Gradient')
ax2.set_ylabel('Electric Field (V/m)')
ax2.set_xlabel('Longitudinal Position (m)')
ax2.legend()

plt.tight_layout()
plt.savefig('/Users/nickvalverde/Dropbox/Research/ORISS/Runs_Plots/initial_potential.png', dpi=300)

####################################################################


#Plotting
#plg(getphi()[0]/1000.,z,color=black)
#fma()
#plg(numpy.gradient(getphi()[0]),z,color=black)
#fma()
#plg(-.5*numpy.gradient(numpy.gradient(getphi()[10])),z,color=black)
#fma()

print(zcentr8 - zcentrIF)




# Open data file of particle initial coordinates


species_ratio = 0.1  # put 1/10 in one species and 9/10 in other.


####################################################################
# Generate Output files for post-process
####################################################################

#Create files
trajectoryfile = open("trajectoryfile.txt","w")  # saves marker trajectories
trackedfile = open("tracked_particle.txt", "w")
poincarefile = open("poincarefile.txt","w")
deltazfile = open("deltazfile.txt","w")
stddevfile = open("stddevfile.txt","w")


#Trajectory File
#Columns Particle, Iter, zp[i], uzp[i], xp[i], uxp[i]
for i in range(0,uranium_beam.getz().size):
    trajectoryfile.write('{},{},{},{},{},{}'.format(i, 0, top.pgroup.zp[i], top.pgroup.uzp[i], top.pgroup.xp[i], top.pgroup.uxp[i]) + "\n")
    trajectoryfile.flush()

trackedfile.write('{},{},{},{},{}'.format(0, tracked_uranium.getz()[0], tracked_uranium.getvz()[0],
                                                     tracked_uranium.getx()[0], tracked_uranium.getvx()[0]) + "\n")
trackedfile.flush()

#Stdev File
bounce_count = 0
iteration = 0
while bounce_count < 5:
    stddevfile.write(str(np.mean(top.pgroup.zp))+" "+str(np.std(top.pgroup.zp))+" "+str(np.mean(top.pgroup.xp))+" "+str(np.std(top.pgroup.xp))+"\n")
    stddevfile.flush()
    
    print(top.pgroup.xp)
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
        trajectoryfile.write('{},{},{},{},{},{}'.format(i, iteration, top.pgroup.zp[i], top.pgroup.uzp[i], top.pgroup.xp[i], top.pgroup.uxp[i]) + "\n")
        trajectoryfile.flush()


# Close files
trajectoryfile.close()
trackedfile.close()
stddevfile.close()
poincarefile.close()
deltazfile.close()




# Print run timing statistics
printtimers()
