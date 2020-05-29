#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 21:03:15 2020

@author: nickvalverde
"""

import matplotlib.pyplot as plt
import numpy as np

tforward = np.linspace(0,5,101)
tbackward = 5+np.linspace(-5, 0, 101)
x = np.linspace(0,8.75, 101)

t = np.hstack((tforward,tbackward))
v = 1

xhigh = np.array(list(map(lambda t: 1.5*v*t, t)))
xref = np.array(list(map(lambda t: v*t, t)))
xlow = np.array(list(map(lambda t: v*t/2, t)))

yhigh = 3*np.ones(t.shape, dtype = float)
yref = 2*np.ones(t.shape, dtype = float)
ylow = 1*np.ones(t.shape, dtype = float)

potential = list(map(lambda x: 2*x/5, x))


fig, ax = plt.subplots()
ax.plot(xhigh, yhigh, c= 'm')
ax.plot(xref, yref, c='r')
ax.plot(xlow, ylow, c='k')
ax.plot(x, potential, c='b')
plt.show()
