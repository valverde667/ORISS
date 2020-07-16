#First version with actual mass spectrometry
#Using Uranium (238) and Neptunium (237)
#IMPORTANT DO NOT NAME ANYTHING LOAD, this interrupts with a declared functionion Species.py


#--Import python packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import numpy as np
import scipy.integrate as integrate
import time
import os

#--Import Warp Packages and modules
import warp as wp

wp.setup()  # setup graphics etc.

e = wp.echarge
mm = wp.mm
kV = wp.kV
####################################################################
# Simulation Mesh, create in w3d but run field solve in r-z
####################################################################

# x-mesh (radial in r-z)
wp.w3d.xmmax =   72.*mm
wp.w3d.xmmin = 0.
wp.w3d.nx    = 144


# y-mesh (do not define in r-z simulations)
wp.w3d.ymmax =  72.*mm
wp.w3d.ymmin =  0.
wp.w3d.ny    =  144

# z-mesh
wp.w3d.zmmin = 0
wp.w3d.zmmax = 680*mm
wp.w3d.nz = 1250
# wp.w3d.nz = 1200
# wp.w3d.nz = 1150
# wp.w3d.nz = 1100
# wp.w3d.nz = 1050
# wp.w3d.nz = 1000
# wp.w3d.nz = 950
# wp.w3d.nz = 900
# wp.w3d.nz = 850
# wp.w3d.nz = 800
# wp.w3d.nz = 750
# wp.w3d.nz = 700
# wp.w3d.nz = 650
# wp.w3d.nz = 600
# wp.w3d.nz = 550
# wp.w3d.nz = 500
# wp.w3d.nz = 450
# wp.w3d.nz = 400
# wp.w3d.nz = 350
# wp.w3d.nz = 300
# wp.w3d.nz = 250

####################################################################
# Field Solve
####################################################################

wp.w3d.solvergeom = wp.w3d.XYZgeom

##Set boundary conditions

#Boundary conditions for mesh
wp.w3d.bound0  = wp.dirichlet # for iz == -nz
wp.w3d.boundnz = wp.dirichlet # for iz == nz
wp.w3d.boundxy = wp.dirichlet #in all transverse directions


wp.f3d.mgmaxiters = 1000 #  Max iterations of multigrid field solve
wp.w3d.l4symtry = True  # True
solver = wp.MRBlock3D() # multi-grid Poisson solver in 2-D r-z geometry
solver.mgverbose = +1 #cannot find what these settings do.
solver.mgtol = 1.e-4  # Absolute tolerance (convergance) of field solver in potential [Volts]
wp.registersolver(solver)

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
V03   = .5*kV
V04   = .5*kV
V05   = .5*kV
V06   = .5*kV
V07   = 1.*kV
V08   = 0.

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
zrend_cyl=wp.ZAnnulus(rmin=entry_radius, rmax=entry_radius+10*mm, length=end_cyl_length,
               voltage=ground, zcent=end_cyl_centr)
zrwall=wp.ZAnnulus(rmin=entry_radius, rmax=Rmax, length=wall_length, voltage=V08, zcent=zwallcentr)
zr8=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V08,zcent=zcentr8)
zrcap = zrwall + zr8 #Combine last annulus with wall to creat one conductor
zr7=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V07,zcent=zcentr7)
zr6=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V06,zcent=zcentr6)
zr5=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V05,zcent=zcentr5)
zr4=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V04,zcent=zcentr4)
zr3=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V03,zcent=zcentr3)
zr2=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V02,zcent=zcentr2)
zr1=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V01,zcent=zcentr1)

#Create cone conductor
hrcone = wp.ZSrfrv(rsrf=rsrf,zsrf=zsrf,voltage=Vcone,zcent=zcentrcone)

#Create interface annulus conductor on right side
zr0 = wp.ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=lengthIF,voltage=V00,zcent=zcentrIF)

#Create drift conductor
zdrift = wp.ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=drift,voltage=V00,zcent=zcentdrift)

#Create left Cone
hlcone = wp.ZSrfrv(rsrf=rsrf,zsrf=zsrf,voltage=Vcone,zcent=-zcentrcone)

#Create Left interface annulus
zl0 = wp.ZAnnulus(rmin=RminIF,rmax=RmaxIF,length=lengthIF,voltage=V00,zcent=-zcentrIF)

#--Create Conductors for left side of device (symmetric so just replace zcent with negative values of right side)
zl1=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V01,zcent=-zcentr1)
zl2=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V02,zcent=-zcentr2)
zl3=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V03,zcent=-zcentr3)
zl4=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V04,zcent=-zcentr4)
zl5=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V05,zcent=-zcentr5)
zl6=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V06,zcent=-zcentr6)
zl7=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V07,zcent=-zcentr7)
zl8=wp.ZAnnulus(rmin=Rmin,rmax=Rmax,length=length,voltage=V08,zcent=-zcentr8)
zlwall=wp.ZAnnulus(rmin=entry_radius, rmax=Rmax,length=wall_length, voltage=V08, zcent=-zwallcentr)
zlcap = zl8 + zlwall #Create left cap conductor by combing wall and annulus
zlend_cyl=wp.ZAnnulus(rmin=entry_radius, rmax=entry_radius+10*mm, length=end_cyl_length,
               voltage=ground, zcent=-end_cyl_centr)
#--Install conductors on mesh.  These are placed with subgrid precision
wp.installconductor(zrend_cyl)
wp.installconductor(zrcap)
wp.installconductor(zr7)
wp.installconductor(zr6)
wp.installconductor(zr5)
wp.installconductor(zr4)
wp.installconductor(zr3)
wp.installconductor(zr2)
wp.installconductor(zr1)
wp.installconductor(hrcone)
wp.installconductor(zr0)
wp.installconductor(zdrift)
wp.installconductor(zl0)
wp.installconductor(hlcone)
wp.installconductor(zl1)
wp.installconductor(zl2)
wp.installconductor(zl3)
wp.installconductor(zl4)
wp.installconductor(zl5)
wp.installconductor(zl6)
wp.installconductor(zl7)
wp.installconductor(zlcap)
wp.installconductor(zlend_cyl)


#Set particle boundary conditions at mesh ends
wp.top.pbound0 = wp.absorb  #boundary condition at iz == 0
wp.top.pboundnz = wp.absorb #boundary condition at iz == nz
wp.top.pboundxy = wp.absorb #boundary condition at edge or radial mesh


#Create Species of particles to advance
uranium_beam = wp.Species(type=wp.Uranium,charge_state=+1,name="Beam species",weight=0)
wp.top.lbeamcom = False #Specify grid does not move to follow beam center

#--Generate Mesh
wp.package("w3d") # Specify 3D code.  But fieldsolver will be r-z and particles will deposit on r-z mesh
wp.generate()     # Initate code, this will also make an initial fieldsolve

#Simplifigy mesh variable
z = wp.w3d.zmesh
#Graph potential on axis
phi = wp.getphi(ix=0, iy=0)
#Set particle energy to unit Joules
particle_energy = 0.357*kV*wp.jperev

#Set index for maximum phi
maxphi_index = np.where(phi == max(phi))[0][0]
#Find and set grid point jmax just before turning point zt
jmax = np.where(e*phi > particle_energy)[0][0]-1 #Minus 1 to move back one cell

#Evaluate first and second derivatives of potential along z
dphi = np.gradient(phi, z)
ddphi = np.gradient(dphi, z)

#Evaluate the constant in the integral formula
I_const = -e/(2*particle_energy)

#Partition integral formula for easier evaluation
numerator = ddphi[:jmax]
denom = np.sqrt(1-e*phi[:jamx]/(particle_energy))
#Evaluate integrand
integrand = numerator/denom
#Evaluate integral
I1 = integrate.simps(integrand) #integral up to max z (before turning point)

#--Compute portion of integral at turning point It
dz = wp.w3d.dz #grid spacing

#Evalute constants
# 'a' constant
anum = phi[jmax+1]/phi[jmax] - 1
aden = 1 - e*phi[jmax]/particle_energy
a = anum/aden * e*phi[jmax]/dz

#evaluate C1 and C2
C_t1 = np.sqrt( particle_energy / (particle_energy-e*phi[jmax]) )
C_t2 = (phi[jmax+1] - 2*phi[jmax] - phi[jmax-1]) / (dz*dz)

C_t = -C_t1*C_t2*2/a

#--Visualize integrand
#evaluate zt
ztnum = particle_energy - e*phi[jmax]
ztden = e*(phi[jmax+1] - phi[jmax])

zt = ztnum/ztden + z[jmax]

#evaluate upper limit on integral
upperlimit_num = particle_energy - e*phi[jmax]
upperlimit_den = e*(phi[jmax+1] - phi[jmax])

upperlimit = np.sqrt(1 - upperlimit_num/upperlimit_den)
u = np.linspace(1, upperlimit, 100)

integrand_t = 1

print(const_t)
fig,ax = plt.subplots()
ax.plot(u, integrandt_t)
ax.set_xlabel('u')
ax.set_ylabel('Integrand')
ax.set_title("Integral near turning point")
plt.tight_layout()
plt.show()


raise Exception()




#--Visulization--
#--Visualize turning point
potential_energy = e*phi

fig,ax = plt.subplots()
ax.set_xlabel('z [mm]')
ax.set_ylabel('Energy [J]')
ax.set_ylim(.99*potential_energy[jmax-1], 1.01*potential_energy[jmax+1])
ax.scatter(z[jmax-1:jmax+1]/mm,
        potential_energy[jmax-1:jmax+1],
        c='b', s=8, label='Potential Energy [J]')
for zpoint in z[jmax-1:jmax+1]:
    ax.axvline(x=zpoint/mm, ls='--', lw=.5, c='k')
ax.axhline(y=particle_energy, ls='--',lw=.5, c='r', label='particle energy')
plt.legend()
plt.tight_layout()
plt.savefig(os.getcwd() + '/meshturning.png')
plt.show()


#--Visualize potential and Integrand
fig,axes = plt.subplots(nrows=2, ncols=1, sharex=True)
phiplot, integrandplot = axes[0], axes[1]

#Plot Potential
phiplot.plot(z, phi/kV, lw=.8)
phiplot.fill_between(z[:jmax],0, phi[:jmax]/kV,
                     alpha=0.3, color='green' )

phiplot.axvline(x=z[max_index], c='k', ls='--', lw=.5)
phiplot.axhline(y=particle_energy/wp.jperev/kV, c='r', ls='--', lw=.5, label='Particle Energy [eV]')
phiplot.axhline(y=0, c='k', lw=.5)

phiplot.set_title("On-axis Potential vs z")
phiplot.set_ylabel('Potential [kV]')

#Plot Integrand
#Mark negative and positive region
neg_i = np.where(integrand<0)[0][0]
neg_f = np.where(integrand<0)[0][-1]
pos_i = np.where(integrand>0)[0][0]
pos_f = np.where(integrand>0)[0][-1]

integrandplot.plot(z[:jmax], integrand, lw=.8)
integrandplot.fill_between(z[:jmax], 0, integrand,
                     alpha=0.3, color='green' )

integrandplot.axvline(x=z[max_index], c='k', ls='--', lw=.5, label="Integrand up to turning point")
integrandplot.axvline(x=z[neg_i], c='b', ls='--', lw=.5)
integrandplot.axvline(x=z[neg_f], c='b', ls='--', lw=.5)
integrandplot.axvline(x=z[pos_i], c='b', ls='--', lw=.5)
integrandplot.axvline(x=z[pos_f], c='b', ls='--', lw=.5)


integrandplot.axhline(y=0, c='k', lw=.5)

integrandplot.set_xlabel('z [m]')
integrandplot.set_ylabel(r'Integrand [1/m$^2$]')
integrandplot.set_title('Integrand vs z')

plt.tight_layout()
plt.savefig(os.getcwd()+'/integrand.png')
plt.show()
