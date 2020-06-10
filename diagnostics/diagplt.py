import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warp as wp



def xzscatplot(beam,
    scale = wp.mm,
    markersize=.1,
    color = 'k',
    title = "x vs z",
    titlesize = 16,
    axlblsize = 14,
    xlabel = "z [mm]",
    ylabel = "x [mm]",
    ):

    x = beam.getx()/scale
    z = beam.getz()/scale

    fig, ax = plt.subplots()
    ax.scatter(z,x, s=markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax

def xyscatplot(beam,
    scale = wp.mm,
    markersize=.1,
    color = 'k',
    title = "x vs y",
    titlesize = 16,
    axlblsize = 14,
    xlabel = "y [mm]",
    ylabel = "x [mm]",
    ):

    x = beam.getx()/scale
    y = beam.gety()/scale

    fig, ax = plt.subplots()
    ax.scatter(y,x, s=markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax

def yzscatplot(beam,
    scale = wp.mm,
    markersize=.1,
    color = 'k',
    title = "y vs z",
    titlesize = 16,
    axlblsize = 14,
    xlabel = "z [mm]",
    ylabel = "y [mm]",
    ):

    z = beam.getz()/scale
    y = beam.gety()/scale

    fig, ax = plt.subplots()
    ax.scatter(z,y, s=markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax

def vxxscatplot(beam,
    scale = wp.mm,
    velscale = 1,
    markersize=.1,
    color = 'k',
    title = r"$v_x$ vs x",
    titlesize = 16,
    axlblsize = 14,
    xlabel = "x [mm]",
    ylabel = r"$v_x$ [m/s]",
    ):

    vx = beam.getvx()/velscale
    x = beam.getx()/scale

    fig, ax = plt.subplots()
    ax.scatter(x,vx, s=markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax

def vxxscatplot(beam,
    scale = wp.mm,
    velscale = 1,
    markersize=.1,
    color = 'k',
    title = r"$v_x$ vs x",
    titlesize = 16,
    axlblsize = 14,
    xlabel = "x [mm]",
    ylabel = r"$v_x$[m/s]",
    ):

    vx = beam.getvx()/velscale
    x = beam.getx()/scale

    fig, ax = plt.subplots()
    ax.scatter(y,x, s=markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax

def vyyscatplot(beam,
    scale = wp.mm,
    velscale = 1,
    markersize=.1,
    color = 'k',
    title = r"$v_y$ vs y",
    titlesize = 16,
    axlblsize = 14,
    xlabel = "y [mm]",
    ylabel = r"$v_y$ [m/s]",
    ):

    vy = beam.getvy()/velscale
    y = beam.gety()/scale

    fig, ax = plt.subplots()
    ax.scatter(y,vy, s=markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax

def vzzscatplot(beam,
    scale = wp.mm,
    velscale = 1,
    markersize = .1,
    color = 'k',
    title = r"$v_z$ vs z",
    titlesize = 16,
    axlblsize = 14,
    xlabel = "z [mm]",
    ylabel = r"$v_z$ [m/s]",
    ):

    vz = beam.getvz()/velscale
    z = beam.getz()/scale

    fig, ax = plt.subplots()
    ax.scatter(z,vz, s=markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax


def vxvyscatplot(beam,
    velscale = 1,
    markersize=.1,
    color = 'k',
    title = r"$v_y$ vs $v_x$",
    titlesize = 16,
    axlblsize = 14,
    xlabel = r"$v_x$ [m/s]",
    ylabel = r"$v_y$ [m/s]",
    ):

    vx = beam.getvx()/velscale
    vy = beam.getvy()/velscale

    fig, ax = plt.subplots()
    ax.scatter(vx,vy, s=markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax

def vxvzscatplot(beam,
    velscale = 1,
    markersize=.1,
    color = 'k',
    title = r"$v_x$ vs $v_z$",
    titlesize = 16,
    axlblsize = 14,
    xlabel = r"$v_z$ [m/s]",
    ylabel = r"$v_x$ [m/s]",
    ):

    vx = beam.getvx()/velscale
    vz = beam.getvz()/velscale

    fig, ax = plt.subplots()
    ax.scatter(vz,vx, s=markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax

def vyvzscatplot(beam,
    velscale = 1,
    markersize=.1,
    color = 'k',
    title = r"$v_y$ vs $v_z$",
    titlesize = 16,
    axlblsize = 14,
    xlabel = r"$v_z$ [m/s]",
    ylabel = r"$v_y$ [m/s]",
    ):

    vy = beam.getvy()/velscale
    vz = beam.getvz()/velscale

    fig, ax = plt.subplots()
    ax.scatter(vz,vy, s=markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax
