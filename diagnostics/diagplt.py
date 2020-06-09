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
    xlabel = "z[mm]",
    ylabel = "x [mm]",
    ):

    x = beam.getx()/scale
    z = beam.getx()/scale

    fig, ax = plt.subplots()
    ax.scatter(z,x, s =markersize, c=color)
    ax.set_title(title, fontsize=titlesize)
    ax.set_xlabel(xlabel, fontsize=axlblsize)
    ax.set_ylabel(ylabel, fontsize=axlblsize)

    return ax
