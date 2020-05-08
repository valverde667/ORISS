#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:38:28 2020

@author: nickvalverde
"""

import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#
# x_list = np.linspace(0, 2*pi, 101)
# y = lambda x: cos(x)
# z = lambda x: sin(x)
# #
# # fig,ax = plt.subplots(nrows = 1, ncols = 2,figsize=(7,7), sharey = True)
# # xax = ax[0]
# # zax = ax[1]
# #
# # xax.set_xlim(0, 2*pi)
# # zax.set_xlim(0, 2*pi)
# # xax.set_ylim(-1,1 )
# # zax.set_ylim(-1,1 )
# #
# # xax.set_xlabel("x",fontsize=20)
# # xax.set_ylabel("y",fontsize=20)
# # xax.set_title(r'$\cos(x)$')
# # zax.set_title(r'$\sin(x)$')
# #
# # def animate(i):
# #     xax.scatter(x = x[i], y = y(x[i]), s = .5, c = 'k')
# #     zax.scatter(x = x[i], y = z(x[i]), s = .5, c = 'k')
# #
# # Writer = animation.writers['ffmpeg']
# # writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=-1)
# #
# # ani = animation.FuncAnimation(fig, animate, frames = len(x), interval = 1, repeat=False)
# # plt.show()
# # ani.save('/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Plots/test.mp4', writer=writer)
#




# # with Blitting

# fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize =(7,7), sharey = True)
# xax = ax[0]
# zax = ax[1]
#
# xax.set_xlim(0, 2*pi)
# zax.set_xlim(0, 2*pi)
# xax.set_ylim(-1,1 )
# zax.set_ylim(-1,1 )
#
# xax.set_xlabel("x",fontsize=20)#     xax.set_ylabel("y",fontsize=20)
# xax.set_title(r'$\cos(x)$')
# zax.set_title(r'$\sin(x)$')
#
# line1, = xax.plot([], [], lw = 3, c = 'k')
# line2, = zax.plot([], [], lw = 3, c = 'k')
#
# def init():
# 	line1.set_data([], [])
# 	line2.set_data([], [])
# 	return line1,
#
# xdata, ydata, zdata = [], [], []
#
# def animate(i):
# 	x = i/100
# 	y = cos(x)
# 	z = sin(x)
#
#
# 	xdata.append(x)
# 	ydata.append(y)
# 	zdata.append(z)
#
# 	line1.set_data(x, y)
# 	line2.set_data(x, z)
#
# 	return line1,
#
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
# anim = animation.FuncAnimation(fig, animate, init_func=init,
# 							frames=len(x_list), interval=200, blit=True)
#
# # save the animation as mp4 video file
# anim.save('/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Plots/Blit_test.mp4',writer=writer)



#
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# plt.style.use('dark_background')
#
# fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (7,7))
# ax = axes[0]
# axx = axes[1]
# ax = plt.axes(xlim=(-50, 50), ylim=(-50, 50))
# axx = plt.axes(xlim=(-50, 50), ylim=(-50, 50))
# line, = ax.plot([], [], lw=2)
# lline, = axx.plot([], [], lw=2)
#
# # initialization function
# def init():
# 	# creating an empty plot/frame
# 	line.set_data([], [])
# 	lline.set_data([],[])
# 	return line,lline,
#
# # lists to store x and y axis points
# xdata, ydata = [], []
# xdata, yydata = [], []
#
# # animation function
# def animate(i):
# 	# t is a parameter
# 	t = 0.1*i
#
# 	# x, y values to be plotted
# 	x = t*np.sin(t)
# 	y = t*np.cos(t)
# 	yy = t*np.cos(t)*np.sin(t)
#
# 	# appending new points to x, y axes points list
# 	xdata.append(x)
# 	ydata.append(y)
# 	yydata.append(yy)
# 	line.set_data(xdata, ydata)
# 	lline.set_data(xdata, yydata)
# 	return line,lline,
#
# # setting a title for the plot
# plt.title('Creating a growing coil with matplotlib!')
# # hiding the axis details
# plt.axis('off')
#
# # call the animator
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
# anim = animation.FuncAnimation(fig, animate, init_func=init,
# 							frames=500, interval=20, blit=True)
#
# # save the animation as mp4 video file
# anim.save('/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Plots/Blit_test.mp4',writer=writer)


#--Create two subplots.
# Works, now to move on to trying to make a scatter plot
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# plt.style.use('dark_background')
#
# fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (7,7), sharey = True, sharex = True)
# ax = axes[0]
# axx = axes[1]
# ax.set_xlim(0, 2*np.pi)
# ax.set_ylim(-1,1)
# axx.set_xlim(0,2*np.pi)
# axx.set_ylim(-1,1)
#
# line, = ax.plot([], [], lw=2)
# lline, = axx.plot([], [], lw=2)
#
# # initialization function
# def init():
# 	# creating an empty plot/frame
# 	line.set_data([], [])
# 	lline.set_data([],[])
# 	return line,lline,
#
# # lists to store x and y axis points
# xdata, ydata = [], []
# xdata, yydata = [], []
#
# x_list = np.linspace(0, 2*np.pi, 101)
# # animation function
# def animate(i):
# 	# t is a parameter
#
# 	x = x_list[i]
#
# 	# x, y values to be plotted
# 	y = np.cos(x)
# 	yy = np.sin(x)
#
# 	# appending new points to x, y axes points list
# 	xdata.append(x)
# 	ydata.append(y)
# 	yydata.append(yy)
# 	line.set_data(xdata, ydata)
# 	lline.set_data(xdata, yydata)
# 	return line,lline,
#
# # setting a title for the plot
# plt.title('Creating a growing coil with matplotlib!')
# # hiding the axis details
#
#
# # call the animator
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
# anim = animation.FuncAnimation(fig, animate, init_func=init,
# 							frames=len(x_list), interval=20, blit=True)
#
# # save the animation as mp4 video file
# anim.save('/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Plots/Blit_test.mp4',writer=writer)


#--Blit test scatter plot
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import numpy as np
# plt.style.use('dark_background')
#
# fig, axes = plt.subplots(figsize = (7,7))
# axes.set_xlim(0, 2*np.pi)
# axes.set_ylim(-1,1)
# scat = axes.scatter([], [], s = 1)
#
# func = lambda x: np.cos(x)
# # initialization function
# def init():
# 	scat.set_offets([])
#
# 	return scat,
#
# # lists to store x and y axis points
# xdata, ydata = [], []
# xdata, yydata = [], []
#
# x = np.linspace(0, 2*np.pi, 101)
# y = func(x)
# # animation function
# def animate(i):
# 	data = np.hstack( (x[:i, np.newaxis], y[:i, np.newaxis]) )
# 	scat.set_offsets(data)
# 	return scat
#
# # setting a title for the plot
# plt.title('Creating a growing coil with matplotlib!')
# # hiding the axis details
#
#
# # call the animator
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
# anim = animation.FuncAnimation(fig, animate, init_func=init,
# 							frames=len(x), interval=20, blit=True)



# #-- Success using scatter with two points. Now to do multiple plots
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
#
#  # close all previous plots
#
# # create a random line to plot
# #------------------------------------------------------------------------------
# func = lambda x: np.cos(x)
# zfunc = lambda x: np.sin(x)
# x = np.linspace(0,2*pi, 1001)
# y = func(x)
# z = zfunc(x)
#
#
# # animation of a scatter plot using x, y from above
# #------------------------------------------------------------------------------
#
# fig, axes = plt.subplots(figsize = (7,7))
# plt.style.use('dark_background')
# axes.set_xlabel('x')
# ax = plt.axes(xlim=(0, 2*pi), ylim=(-1.1, 1.1))
#
# scat1 = ax.scatter([], [], s=10, c = 'c' , label = r'$\cos(x)$')
# scat2 = ax.scatter([], [], s=10, c = 'r', label = r'$\sin(x)$')
# plt.legend()
#
# def init():
#     scat1.set_offsets([])
#     scat2.set_offsets([])
#
#     return scat1, scat2
#
#
# def animate(i):
#     data1 = np.hstack((x[i, np.newaxis], y[i, np.newaxis])) #new axis keeps dimension (N,2). Can plot all particles by using x[0:i]
#     data2 = np.hstack((x[i,np.newaxis], z[i, np.newaxis]))
#
#     scat1.set_offsets(data1)
#     scat2.set_offsets(data2)
#
#     return scat1, scat2
#
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
# anim = animation.FuncAnimation(fig, animate, init_func=init,
# 							frames=len(x), interval=20, blit=True)
# # save the animation as mp4 video file
# anim.save('/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Plots/Blit_test.mp4',writer=writer)





#-- Blit Scatter with Multiple Plots
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import animation
#
#
#
#
# func = lambda x: np.cos(x)
# zfunc = lambda x: np.sin(x)
# x = np.linspace(0,2*pi, 1001)
# y = func(x)
# z = zfunc(x)
#
# fig, ax = plt.subplots()
# for i in range(len(x)):
#     if x[i] <= np.pi:
#         yscat = ax.scatter(x[i],y[i], s = .2)
#         yscat.set_color('r')
#     else:
#         yscat = ax.scatter(x[i], y[i], c = 'm', s = .2)
#
#     zscat = ax.scatter(x,z, s = .2, c= 'k')
#
# yscat.set_label('y')
# zscat.set_label('z')
# plt.legend()
# plt.show()
#

func = lambda x: np.cos(x)
zfunc = lambda x: np.sin(x)
x = np.linspace(0,2*pi, 301)
y = func(x)
z = zfunc(x)

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (7,7), sharex = True, sharey = True)

yax = axes[0]
zax = axes[1]

yax.set_xlim(0, 2*pi)
yax.set_ylim(-1.1, 1.1)
yax.set_title(r'$\cos(x)$', fontsize = 20)
yax.set_ylabel('y', fontsize = 14)

zax.set_xlim(0, 2*pi)
zax.set_ylim(-1.1, 1.1)
zax.set_title(r'$\sin(x)$', fontsize = 20)
zax.set_ylabel('z', fontsize = 14)


fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel("x", fontsize = 14)
plt.tight_layout()

scat1 = yax.scatter([], [], s=10, label = r'$\cos(x)$')
scat2 = zax.scatter([], [], s=10, c = 'r', label = r'$\sin(x)$')

def init():
    scat1.set_offsets([])
    scat2.set_offsets([])

    return scat1, scat2


def animate(i):
    print(i)
    if x[i] <= np.pi:
        scat1.set_color('k')
    else:
        scat1.set_color('c')

    data1 = np.hstack((x[,np.newaxis], y[, np.newaxis]))
    data2 = np.hstack((x[i,np.newaxis], z[i, np.newaxis]))

    scat1.set_offsets(data1)
    scat2.set_offsets(data2)

    return scat1, scat2

Writer = animation.writers['ffmpeg']
writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
anim = animation.FuncAnimation(fig, animate, init_func=init,
							frames=len(x), interval=20, blit=True)
# save the animation as mp4 video file
anim.save('/Users/nickvalverde/Dropbox/Research/ORISS/color-switch-test.mp4',writer=writer)
