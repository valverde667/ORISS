#First version with actual mass spectrometry
#Using Uranium (238) and Neptunium (237)
#
from warp import *
from Forthon import *

setup()
#What is this top library? Where can I find complete documentation? Neither the warp
#wiki or the USPAS lecture slides contain information on it. doc('top') does not
#give an output.
top.debug = False
top.tracelevel = 0
top.indentlevel = 0
top.ltoptimesubs = True #Line 3598 in top.v. Doesn't say much about what it is.
top.dt     = 1e-7 #Specifying time step size. It is specified number of steps
                  #per lattice period

####################################################################
####################################################################
####################################################################

###Here the mesh is created using w3d
w3d.solvergeom = w3d.RZgeom # Setting Solver Geometry. In this case this is
                            # a Poisson solver in 2-D rz-geometry

# x-mesh
w3d.xmmax =   +.1 # Upper limit of mesh
w3d.xmmin =    0. # Lower limit of mesh
w3d.nx    =   4096. # Mesh points are 0,...,nx
#w3d.nx    = 64      ### less fine mesh

# y-mesh
#w3d.ymmax =   +.1 # Upper limit of mesh
#w3d.ymmin =   -.1 # Lower limit of mesh
#w3d.ny    =   128 # Mesh points are 0,...,ny

# z-mesh
w3d.zmmax =   +1. # Upper limit of mesh
w3d.zmmin =   -1. # Lower limit of mesh
#w3d.nz    =   4096 # Mesh points are 0,...,nz
w3d.nz    = 1024   ### less fine

####################################################################
####################################################################
####################################################################

##Set boundary conditions

#Boundary conditions for mesh
w3d.bound0  = dirichlet # for iz == 0
w3d.boundnz = dirichlet # for iz == nz
w3d.boundxy = dirichlet #in all transverse directions

#Setup particle boundary conditions
top.pbound0 = absorb  #boundary condition at iz == 0
top.pboundnz = absorb #boundary condition at iz == nz

####################################################################
####################################################################
####################################################################

f3d.mgmaxiters = 1000 #I'm guessing this is iterations but not sure what for.
                      #Analogous to top, what is the f3d library?



####################################################################
####################################################################
####################################################################

#solver=MultiGrid3D()
solver=MultiGridRZ() # multi-grid Poisson solver in 2-D r-z geometry
#child1=solver.addchild(mins=[0.,0.,-1.],maxs=[.01,0.,.1],refinement=[100,1,100]
solver.mgverbose = +1 #cannot find what these settings do.
solver.mgtol = 1.e-4
registersolver(solver)


####################################################################
####################################################################
####################################################################

###Create geometry

gap = .00451
zsrf = [-.04499,0.0,.04499]
rsrf = [0.071,0.025,.071]
Rmin = .07
RminIF = .06
Rmax = .071
RmaxIF = .061
Length = .02798
LengthIF = .04305
Cone = .04499
drift = .43068

# Voltage cones
Vcone = +0
V00   = +0000
V01   = -0
#V01  = -0
V02   = +5000
V03   = 0.
V04   = 0.
V05   = +00
V06   = +00
V07   = +00
V08   = +8000.

zcentr8=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+7*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr7=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+6*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr6=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+5*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr5=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+4*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr4=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+3*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr3=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+2*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr2=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+1*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentr1=Cone+gap+LengthIF+drift+LengthIF+gap+2*Cone+gap+0*(Length+gap)+Length/2.-(Cone+gap+LengthIF+drift/2)
zcentrcone=Cone+gap+LengthIF+drift+LengthIF+gap+Cone-(Cone+gap+LengthIF+drift/2)
zcentrIF=Cone+gap+LengthIF+drift+LengthIF/2.-(Cone+gap+LengthIF+drift/2)
zcentdrift = Cone+gap+LengthIF+drift/2-(Cone+gap+LengthIF+drift/2)
zcentlIF=Cone+gap+LengthIF/2-(Cone+gap+LengthIF+drift/2)
zcentlcone=0.-(Cone+gap+LengthIF+drift/2)
zcentl1=-Cone-gap-0.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl2=-Cone-gap-1.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl3=-Cone-gap-2.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl4=-Cone-gap-3.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl5=-Cone-gap-4.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl6=-Cone-gap-5.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl7=-Cone-gap-6.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)
zcentl8=-Cone-gap-7.*(Length+gap)-Length/2.-(Cone+gap+LengthIF+drift/2)


#Right geometry
zr8=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V08,zcent=zcentr8)
zr7=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V07,zcent=zcentr7)
zr6=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V06,zcent=zcentr6)
zr5=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V05,zcent=zcentr5)
zr4=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V04,zcent=zcentr4)
zr3=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V03,zcent=zcentr3)
zr2=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V02,zcent=zcentr2)
zr1=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V01,zcent=zcentr1)
hrcone = ZSrfrv(rsrf=rsrf,zsrf=zsrf,voltage=Vcone,zcent=zcentrcone)
zr0 = ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=LengthIF,voltage=V00,zcent=zcentrIF)

zdrift = ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=drift,voltage=V00,zcent=zcentdrift)

#Left Geometry
zl0 = ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=LengthIF,voltage=V00,zcent=zcentlIF)
hlcone = ZSrfrv(rsrf=rsrf,zsrf=zsrf,voltage=Vcone,zcent=zcentlcone)
zl1=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V01,zcent=zcentl1)
zl2=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V02,zcent=zcentl2)
zl3=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V03,zcent=zcentl3)
zl4=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V04,zcent=zcentl4)
zl5=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V05,zcent=zcentl5)
zl6=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V06,zcent=zcentl6)
zl7=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V07,zcent=zcentl7)
zl8=ZAnnulus(rmin=Rmin,rmax=Rmax,length=Length,voltage=V08,zcent=zcentl8)


installconductor(zr8)
installconductor(zr7)
installconductor(zr6)
installconductor(zr5)
installconductor(zr4)
installconductor(zr3)
installconductor(zr2)
installconductor(zr1)
#installconductor(hrcone)
#installconductor(zr0)
#installconductor(zdrift)
#installconductor(zl0)
#installconductor(hlcone)
installconductor(zl1)
installconductor(zl2)
installconductor(zl3)
installconductor(zl4)
installconductor(zl5)
installconductor(zl6)
installconductor(zl7)
installconductor(zl8)


####################################################################
####################################################################
####################################################################

#Create Beams
beam00 = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0) #weight = 0 no spacecharge, 1 = spacecharge. Both go through Poisson sovler.
beam01 = Species(type=Neptunium,charge_state=+1,name="Beam species",weight=0)
print("mass of beam00 is "+str(beam00.sm));
print("mass of beam01 is "+str(beam01.sm));

top.lbeamcom = False ## Line 1247 in top.v 'When true, zbeam follows the beam
                     ## center of mass (minus zbeamcomoffset).'



package("w3d") #Tell warp using 3D mesh
generate()



limits(-1.,1.)
z = linspace(-1.,1.,1024+1)

print(getphi().shape)


####################################################################
####################################################################
####################################################################

#Plotting
#plg(getphi()[0]/1000.,z,color=black)
#fma()
#plg(numpy.gradient(getphi()[0]),z,color=black)
#fma()
#plg(-.5*numpy.gradient(numpy.gradient(getphi()[10])),z,color=black)
#fma()

print(zcentr8 - zcentrIF)


solver.ldosolve=True #This sets up the field solver with the beam fields.
                     #particles are treated as interacting. False, turns off space charge


####################################################################
####################################################################
####################################################################


inputfile = open("inputfile","r") #What exatly is this input data?
#I am not sure what is going on below this line. What is this random splitting?
#Perhaps it is randomly selecting particles with different characteristics to
#create a random entourage of particles. But why 0.1?
for line_str in inputfile:
 fields = line_str.split(' ')
 if random.random()>0.1:
  beam00.addparticles(x=float(fields[2]),y=0.,z=float(fields[0]),vx=float(fields[3]),vy=0,vz=float(fields[1]))
 else:
  beam01.addparticles(x=float(fields[2]),y=0.,z=float(fields[0]),vx=float(fields[3]),vy=0,vz=float(fields[1]))
inputfile.close()

#Create files
trajectoryfile = open("trajectoryfile","w")
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

 step(1)

# for ii in range(0,beam.getz().size):
# if zprev*beam.getz()[0] < 0.:
#  poincarefile.write(str(beam.getz()[1])+" "+str(beam.getvz()[1]-beam.getvz()[0])+" "+str(beam.getx()[1])+" "+str(beam.getvx()[1])+"\n")
#  poincarefile.flush()

 for i in range(0,beam00.getz().size):
  trajectoryfile.write('{0:<2d}         {1:4d}          {2:2.4f}        {3:2.4f}        {4:2.4f}        {5:.4f}'.format(i, iteration, top.pgroup.zp[i], top.pgroup.uzp[i], top.pgroup.xp[i], top.pgroup.uxp[i])+"\n")
  trajectoryfile.flush()


 #Close files
trajectoryfile.close()
stddevfile.close()
poincarefile.close()
deltazfile.close()
printtimers()
