setup()  # setup graphics etc.

####################################################################
# Simulation Mesh, create in w3d but run field solve in r-z
####################################################################

# x-mesh (radial in r-z)
w3d.xmmax =   72.*mm # Upper limit of mesh r_max
w3d.xmmin =    0. # Lower limit of mesh r_min (should be zero for r-z)
#w3d.nx    =   4096. # Mesh points are 0,...,nx
#w3d.nx = 2560
w3d.nx    = 1024      ### less fine mesh

# y-mesh (do not define in r-z simulations)
#w3d.ymmax =   +.1 # Upper limit of mesh
#w3d.ymmin =   -.1 # Lower limit of mesh
#w3d.ny    =   128 # Mesh points are 0,...,ny

# z-mesh
w3d.zmmax =   +680.*mm # Upper limit of mesh
w3d.zmmin =   -680.*mm # Lower limit of mesh
#w3d.nz    =   4096 # Mesh points are 0,...,nz
#w3d.nz  = 2560
w3d.nz    = 1024   ### less fine


####################################################################
# Field Solve
####################################################################

w3d.solvergeom = w3d.RZgeom # Setting Solver Geometry. In this case this is
                            # a Poisson solver in 2-D rz-geometry

##Set boundary conditions

#Boundary conditions for mesh
w3d.bound0  = dirichlet # for iz == -nz
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
entry_radius = 25*mm #inner radius of entry point (left side injector)
Rmin = 70.*mm # Inner radius of electrode (center to inner) annulus.
RminIF = 60.*mm# # Inner radius of interface (portion between cone and drift)
Rmax = 71.*mm # Outer radius of electrode (center to outer) annulus
RmaxIF = 61.*mm # Outer radius of interface (portion between cone and drift)
length = 27.98*mm #length of annulus
lengthIF = 43.05*mm #length of chamber between cone and drit
cone = 44.99*mm #Radius of Cone segment
wall_length = length/3 #1/4 length of potential rings
end_cyl_length = 2*length #twice the length of potential rings

drift = 430.68*mm

# Voltages of rings and cone number from left to right on rings from end
ground = 0. #ground voltage
Vcone = 0. #Voltage for cones
V00   = 0. #Voltage for drift region and interface annulus
V01   = 0. #Voltages for annulus 1-8 below
V02   = 0.
V03   = 0.
V04   = 0.
V05   = 0.
V06   = 0.
V07   = 0.
V08   = 10.*kV
Vcap  = 10.*kV

#--Distances to object centers measured from midpoint of ORISS
#z-centers of right rings
end_cyl_centr = drift/2. + lengthIF + gap + 2*cone + 8*gap + 8*length + wall_length + end_cyl_length/2
zwallcentr = drift/2. + lengthIF + gap + 2*cone + 8*gap + 8*length + wall_length/2
zcentr8 = drift/2. + lengthIF + gap + 2*cone + 8*gap + 7*length + length/2.
zcentr7 = drift/2. + lengthIF + gap + 2*cone + 7*gap + 6*length + length/2.
zcentr6 = drift/2. + lengthIF + gap + 2*cone + 6*gap + 5*length + length/2.
zcentr5 = drift/2. + lengthIF + gap + 2*cone + 5*gap + 4*length + length/2.
zcentr4 = drift/2. + lengthIF + gap + 2*cone + 4*gap + 3*length + length/2.
zcentr3 = drift/2. + lengthIF + gap + 2*cone + 3*gap + 2*length + length/2.
zcentr2 = drift/2. + lengthIF + gap + 2*cone + 2*gap + 1*length + length/2.
zcentr1 = drift/2. + lengthIF + gap + 2*cone + 1*gap + 0*length + length/2.


#z-center of right cone
zcentrcone = drift/2. + lengthIF + gap + cone

#Central pipe segment
zcentrIF = drift/2. + lengthIF/2.
zcentdrift = 0. #The origin

#Create Conductors 
#Create Annlus conductors
zrend_cyl=ZAnnulus(rmin=entry_radius, rmax=entry_radius+10*mm, length=end_cyl_length,
               voltage=ground, zcent=end_cyl_centr)
zrwall=ZAnnulus(rmin=entry_radius, rmax=Rmax, length=wall_length, voltage=V08, zcent=zwallcentr)
zr8=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V08,zcent=zcentr8)
zrcap = zrwall + zr8 #Combine last annulus with wall to creat one conductor
zr7=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V07,zcent=zcentr7)
zr6=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V06,zcent=zcentr6)
zr5=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V05,zcent=zcentr5)
zr4=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V04,zcent=zcentr4)
zr3=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V03,zcent=zcentr3)
zr2=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V02,zcent=zcentr2)
zr1=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V01,zcent=zcentr1)

#Create cone conductor
hrcone = ZSrfrv(rsrf=rsrf,zsrf=zsrf,voltage=Vcone,zcent=zcentrcone)

#Create interface annulus conductor on right side
zr0 = ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=lengthIF,voltage=V00,zcent=zcentrIF)

#Create drift conductor
zdrift = ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=drift,voltage=V00,zcent=zcentdrift)

#Create left Cone
hlcone = ZSrfrv(rsrf=rsrf,zsrf=zsrf,voltage=Vcone,zcent=-zcentrcone)

#Create Left interface annulus
zl0 = ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=lengthIF,voltage=V00,zcent=-zcentrIF)

#--Create Conductors for left side of device (symmetric so just replace zcent with negative values of right side)
zl1=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V01,zcent=-zcentr1)
zl2=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V02,zcent=-zcentr2)
zl3=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V03,zcent=-zcentr3)
zl4=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V04,zcent=-zcentr4)
zl5=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V05,zcent=-zcentr5)
zl6=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V06,zcent=-zcentr6)
zl7=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V07,zcent=-zcentr7)
zl8=ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V08,zcent=-zcentr8)
zlwall=ZAnnulus(rmin=entry_radius, rmax=Rmax,length=wall_length, voltage=V08, zcent=-zwallcentr)
zlcap = zl8 + zlwall #Create left cap conductor by combing wall and annulus
zlend_cyl=ZAnnulus(rmin=entry_radius, rmax=entry_radius+10*mm, length=end_cyl_length,
               voltage=ground, zcent=-end_cyl_centr)
#--Install conductors on mesh.  These are placed with subgrid precision
installconductor(zrend_cyl)
installconductor(zrcap)
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
installconductor(zlcap)
installconductor(zlend_cyl)
