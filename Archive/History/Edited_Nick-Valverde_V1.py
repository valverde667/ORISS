#First version with actual mass spectrometry
#Using Uranium (238) and Neptunium (237)
#

from warp import *
from Forthon import *
import matplotlib.pyplot as plt

setup()  # setup graphics etc.

####################################################################
# Simulation Mesh, create in w3d but run field solve in r-z
####################################################################

# x-mesh (radial in r-z)
w3d.xmmax =   +.1 # Upper limit of mesh r_max
w3d.xmmin =    0. # Lower limit of mesh r_min (should be zero for r-z)
w3d.nx    =   4096. # Mesh points are 0,...,nx
#w3d.nx    = 64      ### less fine mesh

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

gap = .00451  # insulating gap between rings and ring/cone etc
zsrf = [-.04499,0.0,.04499]  # downstream, upstream z-extent ring
rsrf = [0.071,0.025,.071] #coordinates for creating cone of revolution
Rmin = .07 # Inner radius of electrode (center to inner interface) annulus.
RminIF = .06# # Inner radius of chamber before drift space
Rmax = .071 # Outer radius of electrode (center to outer interface) annulus
RmaxIF = .061 # Outer radius of chamber before drift space.
Length = .02798
LengthIF = .04305
Cone = .04499
drift = .43068

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
V08   = +8000.

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
beam00 = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0) #weight = 0 no spacecharge, 1 = spacecharge. Both go through Poisson sovler.
beam01 = Species(type=Neptunium,charge_state=+1,name="Beam species",weight=0)

print("mass of beam00 is "+str(beam00.sm));
print("mass of beam01 is "+str(beam01.sm));

top.lbeamcom = False # Specify grid does not move to follow beam center of mass as appropriate for trap simulation


####################################################################
# Generate Code
####################################################################

package("w3d") # Specify 3D code.  But fieldsolver will be r-z and particles will deposit on r-z mesh
generate()     # Initate code, this will also make an initial fieldsolve



solver.ldosolve = True #This sets up the field solver with the beam fields.
                     #particles are treated as interacting. False, turns off space charge
limits(-1.,1.)
z = linspace(-1.,1.,1024+1)

#raise Exception("To Here")

####################################################################
# Setup Diagnostics and make initial diagnostic plots
#
#  Make nice initial field diagnostic plots with labels.  Make plots of:
#   1) phi(r=0,z) vs z in volts

    plt.plot(z, getphi()[0])
    plt.title('Initial Potential')
    plt.xlabel('z')
    plt.ylabel('V')
    plt.savefig('initial_potential.png', dpi=300)
#   1a) repeat 1) and put the inital bunch energy on in units of volts
#   2) E_z vs z labeled
#   3) (E_r/r)|_r = 0 vs z labeled ... transverse field gradient vs z
####################################################################






#Plotting
#plg(getphi()[0]/1000.,z,color=black)
#fma()
#plg(numpy.gradient(getphi()[0]),z,color=black)
#fma()
#plg(-.5*numpy.gradient(numpy.gradient(getphi()[10])),z,color=black)
#fma()

print(zcentr8 - zcentrIF)


####################################################################
# Generate Particles
#    Particles generated in load script loader.py
#    and then read in via text file "inputfile"
#
#    Please modify ... have loader in this script
#    Make version repeating what Robert has here.  But we should discuss
#    options to make.  I think Robert's script generates uniform sphere,
#    maybe Gaussian dist, or could be simulation data from Ryan Ringle.
#    You will want to generate:
#     1) Uniformly filled ellipsolid (r_perp, r_z)
#     2) Gaussian distribution in r and z with specified rms parameters
#     3) Input particles from a dump file of another code (maybe change units ... see how Robert read in)
####################################################################

# Open data file of particle initial coordinates
inputfile = open("inputfile","r")

species_ratio = 0.1  # put 1/10 in one species and 9/10 in other.

for line_str in inputfile:
 fields = line_str.split(' ')
 if random.random()>species_ratio:
  beam00.addparticles(x=float(fields[2]),y=0.,z=float(fields[0]),vx=float(fields[3]),vy=0.,vz=float(fields[1]))
 else:
  beam01.addparticles(x=float(fields[2]),y=0.,z=float(fields[0]),vx=float(fields[3]),vy=0.,vz=float(fields[1]))

inputfile.close()

####################################################################
# Generate Output files for post-process
####################################################################

#Create files
trajectoryfile = open("trajectoryfile","w")  # saves marker trajectories
poincarefile = open("poincarefile","w")
deltazfile = open("deltazfile","w")
stddevfile = open("stddevfile","w")

#Trajectory File
trajectoryfile.write('{0:<10}{1:5}            {2:15}{3:15}{4:15}{5:15}'.format('Particle', 'Iter', 'zp[i]', 'uzp[i]', 'xp[i]', 'uxp[i]')+'\n' )

for i in range(0,beam00.getz().size):
 trajectoryfile.write('{0:<2d}            {1:d}          {2:2.4f}        {3:2.4f}        {4:2.4f}        {5:.4f}'.format(i, 0, top.pgroup.zp[i], top.pgroup.uzp[i], top.pgroup.xp[i], top.pgroup.uxp[i]) + "\n")
 trajectoryfile.flush()

#Stdev File
for iteration in range(1,2066*10):
 stddevfile.write(str(numpy.mean(top.pgroup.zp))+" "+str(numpy.std(top.pgroup.zp))+" "+str(numpy.mean(top.pgroup.xp))+" "+str(numpy.std(top.pgroup.xp))+"\n")
 stddevfile.flush()
# if numpy.mean(top.pgroup.zp[0]) < 0:
#  break
 #zcentroid = getbeamcom(top.pgroup) #Movies
 #beam00.ppzx(color="blue",pplimits=(zcentroid-.005,zcentroid+.005,-.005,.005));plt( " "+str(zcentroid), -.01, -190,tosys=1);
 #beam01.ppzx(color="red",pplimits=(zcentroid-.005,zcentroid+.005,-.005,.005));plt( " "+str(zcentroid), -.01, -190,tosys=1);
 #fma();

# ppxvx(pplimits=(-.01,+.01,-200,200),marker='\5');plt( " "+str(zcentroid), -.01, -190,tosys=1);fma()

# zprev = beam.getz()[0]

# deltazfile.write(str(beam.getz()[1]-beam.getz()[0])+" "+str(beam.getvz()[1]-beam.getvz()[0])+"\n")
# deltazfile.flush()

 step(1)  # advance particles
 print(0.5*beam00.mass*beam00.getvz()**2/jperev)
# for ii in range(0,beam.getz().size):
# if zprev*beam.getz()[0] < 0.:
#  poincarefile.write(str(beam.getz()[1])+" "+str(beam.getvz()[1]-beam.getvz()[0])+" "+str(beam.getx()[1])+" "+str(beam.getvx()[1])+"\n")
#  poincarefile.flush()

 for i in range(0,beam00.getz().size):
  trajectoryfile.write('{0:<2d}         {1:4d}          {2:2.4f}        {3:2.4f}        {4:2.4f}        {5:.4f}'.format(i, iteration, top.pgroup.zp[i], top.pgroup.uzp[i], top.pgroup.xp[i], top.pgroup.uxp[i])+"\n")
  trajectoryfile.flush()


# Close files
trajectoryfile.close()
stddevfile.close()
poincarefile.close()
deltazfile.close()

# Print run timing statistics
printtimers()
