#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:33:02 2020

@author: nickvalverde
"""


f = open("parameters.txt", "r")
for x in f:
  parameters = x.split(',')
  variable = parameters[0]
  value = parameters[1]
  exec("%s = %s" %(variable, value))
 
f.close()