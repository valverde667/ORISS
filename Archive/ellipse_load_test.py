from warp import *
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

uranium_beam = Species(type=Uranium,charge_state=+1,name="Beam species",weight=0) #weight = 0 no spacecharge, 1 = spacecharge. Both go through Poisson sovler.
mass = uranium_beam.mass
eVtoK = 8.62e-5
kV = 1e3
mm = 1e-3
Np = 10000






#Position Filler
X,Y,Z = 0,0,0
sigma = (1*mm, 1*mm, 0.1*mm)
x_array = []
y_array = []
z_array = []

#initialize coordinates for routine
sigmax, sigmay, sigmaz = sigma[0], sigma[1], sigma[2]
x = np.random.normal(X, sigmax)
y = np.random.normal(Y, sigmay)
z = np.random.normal(Z, sigmaz)
rperp = sqrt(x**2 + y**2)
rz = z

counter = 0
while counter < Np:
    sigmax, sigmay, sigmaz = sigma[0], sigma[1], sigma[2]


    Ux = np.random.uniform(0,1)
    Uy = np.random.uniform(0,1)
    Uz = np.random.uniform(0,1)

    dx = rperp*(-1 + 2*Ux)
    dy = rperp*(-1 + 2*Uy)
    dz = z*(-1 + 2*Uz)

    term1 = (dx**2 + dy**2)/rperp**2
    term2 = dz**2/rz**2

    condition = term1 + term2

    if condition < 1:
        x_array.append(dx)
        y_array.append(dy)
        z_array.append(dz)
        counter += 1
    else:
        pass

x_array = np.array(x_array)
y_array = np.array(y_array)
z_array = np.array(z_array)

plt.scatter(x = z_array/mm,  y = x_array/mm, s = .5)
plt.xlabel("z [mm]")
plt.ylabel("x [mm]")
plt.tight_layout()
plt.show()

plt.scatter(x = z_array/mm,  y = y_array/mm, s = .5)
plt.xlabel("z [mm]")
plt.ylabel("y [mm]")
plt.tight_layout()
plt.show()

plt.scatter(x = x_array/mm,  y = y_array/mm, s = .5)
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.tight_layout()
plt.show()





#Velocity Filler
energy = 2.77*kV
t_perp = 8.62e-5 #[eV] corresponds to 1K
t_para = 8.62e-5 #[eV]
Vx, Vy, Vz = 0, 0, np.sqrt(energy*jperev*2/mass)
vperp = sqrt(5*boltzmann*t_perp/(2*eVtoK*mass))
vz = Vz + sqrt(5*boltzmann*t_para/(eVtoK*mass))
vx_array = []
vy_array = []
vz_array = []
counter = 0
while counter < Np:

    Ux = np.random.uniform(0,1)
    Uy = np.random.uniform(0,1)
    Uz = np.random.uniform(0,1)

    dvx = vperp*(-1 + 2*Ux)
    dvy = vperp*(-1 + 2*Uy)
    dvz = vz*(-1 + 2*Uz)

    term1 = (dvx**2 + dvy**2)/vperp**2
    term2 = dvz**2/vz**2

    condition = term1 + term2

    if condition < 1:
        vx_array.append(dvx)
        vy_array.append(dvy)
        vz_array.append(dvz)
        counter += 1
    else:
        pass

vx_array = np.array(vx_array)
vz_array = np.array(vz_array)

plt.scatter(x = vz_array,  y = vx_array, s = .5)
plt.xlabel(r"$v_z$[m/s]", fontsize = 14)
plt.ylabel(r"$v_x$[m/s]", fontsize = 14)
plt.tight_layout()
plt.show()

plt.scatter(x = vz_array,  y = vy_array, s = .5)
plt.xlabel(r"$v_z$[m/s]", fontsize = 14)
plt.ylabel(r"$v_y$[m/s]", fontsize = 14)
plt.tight_layout()
plt.show()

plt.scatter(x = vx_array,  y = vy_array, s = .5)
plt.xlabel(r"$v_x$[m/s]", fontsize = 14)
plt.ylabel(r"$v_y$[m/s]", fontsize = 14)
plt.tight_layout()
plt.show()



plt.scatter(x = x_array/mm,  y = vx_array, s = .5)
plt.xlabel("x[mm]", fontsize = 14)
plt.ylabel(r"$v_x$[m/s]", fontsize = 14)
plt.tight_layout()
plt.show()

plt.scatter(x = y_array/mm,  y = vy_array, s = .5)
plt.xlabel("y[mm]", fontsize = 14)
plt.ylabel(r"$v_y$[m/s]", fontsize = 14)
plt.tight_layout()
plt.show()

plt.scatter(x = z_array/mm,  y = vz_array, s = .5)
plt.xlabel("$z[mm]", fontsize = 14)
plt.ylabel(r"$v_z$[m/s]", fontsize = 14)
plt.tight_layout()
plt.show()
