#First version with actual mass spectrometry
#Using Uranium (238) and Neptunium (237)
#

from warp import *
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


# Install conductors on mesh.  These are placed with subgrid precision
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

top.dt     = 1e-7 # Timestep of particle advance

#Set particle boundary conditions at mesh ends
top.pbound0 = absorb  #boundary condition at iz == 0
top.pboundnz = absorb #boundary condition at iz == nz
top.pboundxy = absorb #boundary condition at edge or radial mesh


#Create Species of particles to advance
uranium_beam = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0) #weight = 0 no spacecharge, 1 = spacecharge. Both go through Poisson sovler.

print("mass of uranium_beam is "+str(uranium_beam.sm));

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

load_list = []

particle_energy = 2.77*kV #2.77 kV is good for both point-point and ||-point
#velocity = 47388.2515713436 m/s
#--Point to point load
particle_energy_range = particle_energy*np.array([.997, .998, .999, 1.000, 1.001, 1.002, 1.003])
#particle_energy_range = particle_energy*np.array([1., 1., 1., 1., 1., 1., 1.])
#--Parallel to point load
x_list = np.array([i*.001 for i in range(-4, 4 + 1)])

for energy in particle_energy_range:
    counter = 0
    Np = 10
    while counter < Np:
        p = MyParticle(energy,uranium_beam)
        load = p.loader('gaussian', sigma = (2*mm, 0, 0))
        load_xpos = load[0][0]
        if abs(load_xpos) > 6*mm:
            pass
        elif abs(load_xpos) < 1*mm:
            pass
        else:
            load_list.append(load)
            counter += 1
  
for array in load_list:
    load = array[0]
    xpos, ypos, zpos = load[0], load[1], load[2]
    vx, vy, vz = load[3], load[4], load[5] #1.178 to math V4 code
    uranium_beam.addparticles(xpos, ypos, zpos, vx, vy, vz)


###Paralell to point below

#uranium_beam.addparticles(x=-.004, y=0., z =0., vx = 0., vy=0., vz= velocity)
#uranium_beam.addparticles(x=-0.003, y=0., z =0., vx = 0., vy=0., vz= velocity)
#uranium_beam.addparticles(x=-0.002, y=0., z =0., vx = 0., vy=0., vz= velocity)
#uranium_beam.addparticles(x=-0.001, y=0., z =0., vx = 0., vy=0., vz= velocity)
#uranium_beam.addparticles(x=0., y=0., z =0., vx = 0., vy=0., vz= velocity)
#uranium_beam.addparticles(x=0.001, y=0., z =0., vx = 0., vy=0., vz= velocity)
#uranium_beam.addparticles(x=0.002, y=0., z =0., vx = 0., vy=0., vz= velocity)
#uranium_beam.addparticles(x=0.003, y=0., z =0., vx = 0., vy=0., vz= velocity)
#uranium_beam.addparticles(x=0.004, y=0., z =0., vx = 0., vy=0., vz= velocity)
#uranium_beam.addparticles(x=0.005, y=0., z =0., vx = 0., vy=0., vz= velocity)



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
plt.savefig('initial_potential.png', dpi=300)

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
poincarefile = open("poincarefile.txt","w")
deltazfile = open("deltazfile.txt","w")
stddevfile = open("stddevfile.txt","w")


#Trajectory File
#Columns Particle, Iter, zp[i], uzp[i], xp[i], uxp[i]
for i in range(0,uranium_beam.getz().size):
    trajectoryfile.write('{},{},{},{},{},{}'.format(i, 0, top.pgroup.zp[i], top.pgroup.uzp[i], top.pgroup.xp[i], top.pgroup.uxp[i]) + "\n")
    trajectoryfile.flush()

#Stdev File
for iteration in range(1,2066*10):
    stddevfile.write(str(numpy.mean(top.pgroup.zp))+" "+str(numpy.std(top.pgroup.zp))+" "+str(numpy.mean(top.pgroup.xp))+" "+str(numpy.std(top.pgroup.xp))+"\n")
    stddevfile.flush()





    step(1)  # advance particles
    print(0.5*uranium_beam.mass*uranium_beam.getvz()**2/jperev)
    print("beam velocity is: ", uranium_beam.getvz())





    for i in range(0,uranium_beam.getz().size):
        trajectoryfile.write('{},{},{},{},{},{}'.format(i, iteration, top.pgroup.zp[i], top.pgroup.uzp[i], top.pgroup.xp[i], top.pgroup.uxp[i]) + "\n")
        trajectoryfile.flush()


# Close files
trajectoryfile.close()
stddevfile.close()
poincarefile.close()
deltazfile.close()

# Print run timing statistics
printtimers()
