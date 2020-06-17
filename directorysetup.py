import os

parent_dir = os.getcwd()

#--Diagnostic plotting
prediagnostics = 'prediagnostics'
path = os.path.join(parent_dir,prediagnostics)
os.mkdir(path)
print("prediagnostics folder created")

#--Outputs
outputs = 'outputs/'

#datafiles
data='data'
path = os.path.join(parent_dir,outputs+data)
os.mkdir(path)
print("outputs/data folder created")

#animations
animations = 'animations'
path = os.path.join(parent_dir,outputs+animations)
os.mkdir(path)
print("oputs/animations folder created")

#diagnostics
diagnostics='diagnostics'
path = os.path.join(parent_dir, outputs+diagnostics)
os.mkdir(path)
print("outputs/diagnostics folder created")
