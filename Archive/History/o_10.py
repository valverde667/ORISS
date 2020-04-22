#First version with actual mass spectrometry
#Using Uranium (238) and Neptunium (237)
#
from warp import *
from Forthon import *

setup()

top.debug = False
top.tracelevel = 0
top.indentlevel = 0
top.ltoptimesubs = True
top.dt     = 1e-7
top.pbound0 = absorb
top.pboundnz = absorb
#Use these settings for XYZ
w3d.solvergeom = w3d.RZgeom
w3d.xmmax =   +.1 # Upper limit of mesh
w3d.xmmin =    0. # Lower limit of mesh
w3d.nx    =   4096. # Mesh points are 0,...,nx


#w3d.ymmax =   +.1 # Upper limit of mesh
#w3d.ymmin =   -.1 # Lower limit of mesh
#w3d.ny    =   128 # Mesh points are 0,...,ny

w3d.zmmax =   +1. # Upper limit of mesh
w3d.zmmin =   -1. # Lower limit of mesh
w3d.nz    =   4096 # Mesh points are 0,...,nz
w3d.nz = 1024

w3d.bound0  = dirichlet
w3d.boundnz = dirichlet
w3d.boundxy = dirichlet

f3d.mgmaxiters = 1000

#solver=MultiGrid3D()
solver=MultiGridRZ()
#child1=solver.addchild(mins=[0.,0.,-1.],maxs=[.01,0.,.1],refinement=[100,1,100]
solver.mgverbose = +1
solver.mgtol = 1.e-4
registersolver(solver)


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

Vcone = +0
V00   = +0000
V01   = -0
#V01  = -0
V02   = +0
V03   = +0
V04   = -4000
V05   = +00
V06   = +00
V07   = +00
V08   = +8000

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

beam00 = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0)
beam01 = Species(type=Neptunium,charge_state=+1,name="Beam species",weight=0)
print("mass of beam00 is "+str(beam00.sm));
print("mass of beam01 is "+str(beam01.sm));

top.lbeamcom = False

package("w3d");generate()

limits(-1.,1.)
z = linspace(-1.,1.,1024+1)
raise Exception('Stop')
print(getphi().shape)



plg(getphi()[0]/1000.,z,color=black)
fma()
plg(numpy.gradient(getphi()[0]),z,color=black)
fma()
plg(-.5*numpy.gradient(numpy.gradient(getphi()[10])),z,color=black)
fma()


print(zcentr8 - zcentrIF)


solver.ldosolve=True


inputfile = open("inputfile","r")
for line_str in inputfile:
 fields = line_str.split(' ')
 if random.random()>0.1:
  beam00.addparticles(x=float(fields[2]),y=0.,z=float(fields[0]),vx=float(fields[3]),vy=0,vz=float(fields[1])) 
 else:
  beam01.addparticles(x=float(fields[2]),y=0.,z=float(fields[0]),vx=float(fields[3]),vy=0,vz=float(fields[1])) 
inputfile.close()

trajectoryfile = open("trajectoryfile","w")
poincarefile = open("poincarefile","w")
deltazfile = open("deltazfile","w")
stddevfile = open("stddevfile","w")

for i in range(0,beam00.getz().size):
 trajectoryfile.write(str(i) + " " + "0" + " " + str(top.pgroup.zp[i])+" "+str(top.pgroup.uzp[i])+" "+str(top.pgroup.xp[i])+" "+str(top.pgroup.uxp[i])+" "+"\n")
 trajectoryfile.flush()

for iteration in range(1,2066*10):
 stddevfile.write(str(numpy.mean(top.pgroup.zp))+" "+str(numpy.std(top.pgroup.zp))+" "+str(numpy.mean(top.pgroup.xp))+" "+str(numpy.std(top.pgroup.xp))+"\n")
 stddevfile.flush()
# if numpy.mean(top.pgroup.zp[0]) < 0:
#  break
 zcentroid = getbeamcom(top.pgroup)
 beam00.ppzx(color="blue",pplimits=(zcentroid-.005,zcentroid+.005,-.005,.005));plt( " "+str(zcentroid), -.01, -190,tosys=1);
 beam01.ppzx(color="red",pplimits=(zcentroid-.005,zcentroid+.005,-.005,.005));plt( " "+str(zcentroid), -.01, -190,tosys=1);
 fma();

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
  trajectoryfile.write(str(i) + " " + str(iteration) + " " + str(top.pgroup.zp[i])+" "+str(top.pgroup.uzp[i])+" "+str(top.pgroup.xp[i])+" "+str(top.pgroup.uxp[i])+" "+"0"+" "+"\n")
  trajectoryfile.flush()
trajectoryfile.close()
stddevfile.close()
poincarefile.close()
deltazfile.close()
printtimers()

