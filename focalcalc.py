#First version with actual mass spectrometry
#Using Uranium (238) and Neptunium (237)
#IMPORTANT DO NOT NAME ANYTHING LOAD, this interrupts with a declared functionion Species.py


#--Import python packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import numpy as np
import time

#--Import Warp Packages and modules
import warp as wp

exec(open("3Dgeometry.py").read()) #Create simulation mesh


#--Useful Constants
mm = wp.mm
kV = wp.kV
e = wp.echarge


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




phiz = wp.getphi(ix=0, iy=0)
particle_energy =  max(phiz)*wp.jperev

dphi = np.gradient(phiz, wp.w3d.dz)
ddphi = np.gradient(dphi, wp.w3d.dz)

integralconst = -e/(4*particle_energy)

x = e*phiz/particle_energy
xx = x**2
xxx = x*x*x


integrand = integralconst*ddphi*(1 + x + xx + xxx)

fig,ax = plt.subplots()
ax.setxlabel('z [m]')
ax.setylabel('Integrand [1/m]')
ax.plot(wp.w3d.zmesh, integrand)
plt.show()


