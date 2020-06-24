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


zeroindex = np.where(wp.w3d.zmesh==0)[0][0]
z = wp.w3d.zmesh[zeroindex:]
phiz = wp.getphi(ix=0, iy=0)[zeroindex:]
maxindex = np.where(phiz==max(phiz))[0][0]

particle_energy =  0.98*max(phiz)*wp.jperev

dphi = np.gradient(phiz, z)
ddphi = np.gradient(dphi, z)

integralconst = -e/(4*particle_energy)

x = e*phiz/particle_energy
xx = x**2
xxx = x*x*x

integrand = integralconst*ddphi*(1 + x + xx + xxx)

fig,axes = plt.subplots(nrows=2, ncols=1, sharex=True)
phiplot, integrandplot = axes[0], axes[1]
phiplot.plot(z, phiz)
phiplot.axvline(x=z[maxindex], c='k', ls='--', lw=.5)
phiplot.set_ylabel('Potential [kV]')

integrandplot.plot(z,integrand)
integrandplot.axvline(x=z[maxindex], c='k', ls='--', lw=.5)
integrandplot.set_xlabel('z [m]')
integrandplot.set_ylabel(r'Integrand [1/m$^2$]')

I = integrate.simps(integrand)
focalpoint = 1/I
print("Focal Point at %f [m] " %focalpoint)
plt.tight_layout()
plt.show()
