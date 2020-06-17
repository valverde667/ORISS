import os

currentdir = os.getcwd()+'/'

#--Diagnostic plotting
diagnostics = 'prediagnostics/'
os.mkdir(currentdir+diagnostics)

#--Outputs
outputs = 'outputs/'

#datafiles
data='data'
os.mkdir(currentdir+outputs+data)

#animations
animations = 'animations'
os.makedir(currentdir+outputs+animations)

#diagnostics
diagnostics='diagnostics'
os.mkdir(currentdir+outputs+diagnostics)
