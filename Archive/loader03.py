#This is loader03.
#It allows for a distribution of velocities.
#The first particle must be the reference particle
import math
import random
inputfile = open("inputfile","w")
#first do the reference particle
#turning point is -.48m
#vref=0
#vref=-6897.932849510406  # turning point -.58
#vref=-6838.617829271197  # turning point -.57
#vref=-6743.593998710405  # turning point -.56
#vref=-6632.476498869057  # turning point -.55
#vref=-6516.349380553174  # turning point -.54
#vref=-6397.93937980242   # turning point -.53
#vref=-6273.5238997883525 # turning point -.52
#vref=-6135.061335008831  # turning point -.51
#vref=-5971.777827465989  # turning point -.50
#vref=-5771.615032779739  # turning point -.49   nearing focus at 0
#vref=+5523.136140213046  # turning point -.48 This looks good. Go with this.
vref = 8000
#vref=-5218.431398813392  # turning point -.47
#vref=-4856.771075758839  # turning point -.46
#vref=-4447.4507487667315 # turning point -.45

#inputfile.write(".0000"+str(vref)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*0.95)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*0.96)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*0.97)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*0.98)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*0.99)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*1.00)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*1.01)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*1.02)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*1.03)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*1.04)+" 0.0000 0. 0 \n")
inputfile.write(".0000 "+str(vref*1.05)+" 0.0000 0. 0 \n")

#inputfile.write(".0000 "+str(vref)+" 0.0000 0. 0 \n")
#inputfile.write(".0000 "+str(vref)+" 0.0001 0. 0 \n")
#inputfile.write(".0000 "+str(vref)+" 0.0002 0. 0 \n")
#inputfile.write(".0000 "+str(vref)+" 0.0003 0. 0 \n")
#inputfile.write(".0000 "+str(vref)+" 0.0004 0. 0 \n")
#inputfile.write(".0000 "+str(vref)+" 0.0005 0. 0 \n")
#inputfile.write(".0000 "+str(vref)+" 0.0006 0. 0 \n")
#inputfile.write(".0000 "+str(vref)+" 0.0007 0. 0 \n")
#inputfile.write(".0000 "+str(vref)+" 0.0008 0. 0 \n")
#inputfile.write(".0000 "+str(vref)+" 0.0009 0. 0 \n")

inputfile.close()
quit()


#This creates a circular distribution in z and x
#single mass
hits = 0
while hits < 999:
 z = random.random()*.001-.0005
 x = random.random()*.001-.0005
# vz = vref+(random.random()*.01*vref)-.005*vref
 vz = vref
 vx = 0.
 masspart = 0
 if (z/(.001-.0005))**2+((x)/(.001-.0005))**2 < 1.:
  hits = hits + 1
  if masspart == 0:
   inputfile.write(str(z)+" "+str(vz)+" "+str(x)+" "+str(vx)+" "+str(masspart)+"\n")
  if masspart == 1:
   inputfile.write(str(z-.55)+" "+str(vz)+" "+str(x)+" "+str(vx)+" "+str(masspart)+"\n")

quit()


#Try a matched distribution, this time with z
hits = 0
while hits < 999:
 z = random.random()*.0002-.0001
 x = random.random()*.0002-.0001
 vz = vref
 vx = random.random()*34.-17.
 masspart = 0
 if (x/.0001)**2+(vx/17.)**2 < 1.:
  hits = hits + 1
  if masspart == 0:
   inputfile.write(str(z)+" "+str(vz)+" "+str(x)+" "+str(vx)+" "+str(masspart)+"\n")
  if masspart == 1:
   inputfile.write(str(z-.55)+" "+str(vz)+" "+str(x)+" "+str(vx)+" "+str(masspart)+"\n")

quit()



hits = 0
while hits < 999:
 z = random.random()*.0001-.00005
# z = 0.
 x = random.random()*.0001-.00005
# x = 0.
# dvz = random.random()*100.-50.
 dvz = 0.
# dvx = random.random()*100.-50.
 dvx = 0.

# masspart = int(random.random()*2)
 masspart = 0
 vz_ref = v+dvz
 vz_general = vz_ref*math.sqrt(1/1.005)+dvz
 vx = dvx
 if (z/(.0001-.00005))**2+(x/(.0001-.00005))**2 < 1.:
  if (z/(.0001-.00005))**2+(dvz/(100.-50.))**2 < 1.:
   if (x/(.0001-.00005))**2+(dvx/(100.-50.))**2 < 1.:
    hits = hits + 1
    if masspart == 0:
     inputfile.write(str(z-.00)+" "+str(vz_ref)+" "+str(x)+" "+str(vx)+" "+str(masspart)+"\n")
    if masspart == 1:
     inputfile.write(str(z-.00)+" "+str(vz_general)+" "+str(x)+" "+str(vx)+" "+str(masspart)+"\n")

quit()

hits = 0
while hits < 500:
 z = random.random()*.001-.0005
# z = 0.
 x = random.random()*.001-.0005
# x = 0.
# masspart = int(random.random()*2)
 masspart = 0
 vz_ref = v*math.sqrt(1/1.005)
 vx = 0.
 if x**2+z**2 < (.001-.0005)*(.001-.0005):
  hits = hits + 1
  if masspart == 0:
   inputfile.write(str(z-.02)+" "+str(vz_ref)+" "+str(x)+" "+str(vx)+" "+str(masspart)+"\n")
  if masspart == 1:
   inputfile.write(str(z-.02)+" "+str(vz_general)+" "+str(x)+" "+str(vx)+" "+str(masspart)+"\n")
