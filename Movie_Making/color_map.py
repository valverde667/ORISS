import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import pandas as pd
import seaborn as sns


mean = [0, 0]
cov = [[1, 1], [1, 2]]
# for i in range(3):
#     x, y = np.random.multivariate_normal(mean, cov, 100000).T
#     plt.hist2d(x, y, bins=100, cmap='Blues')
#     cb = plt.colorbar()
#     cb.set_label('counts in bin')
#
#     plt.show()

#--Create the plot with color bar
#x,y = np.random.multivariate_normal(mean,cov,10000).T
# fig = plt.figure(figsize=(8,8))
# ax = plt.subplot2grid((1,1), (0,0))
# ax.set_title('Color Map', fontsize = 14)
# ax.set_xlabel('x', fontsize = 12)
# ax.set_ylabel('y', fontsize = 12)
# color_map = ax.hist2d(x, y, bins=100, cmap='Blues') #returns 4. The 4th (Image) is used for the map in plt.colorbar
# plt.colorbar(color_map[3], ax=ax) #notice the argument is the Image from color_map
# plt.show()

#--Animation

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation

fig = plt.figure()
data = np.random.rand(100, 100)
color = sns.heatmap(data, square=True)

def init():
      color = sns.heatmap(np.zeros((100, 100)), square=True, cbar=False)
      return color,
def animate(i):
    print(i)
    data = np.random.rand(100, 100)
    color = sns.heatmap(data, vmax=.8, square=True, cbar=False)
    return color,

num_of_frames = 20 #number of times animate() will be called
Writer = animation.writers['ffmpeg'] #for saving purposes
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800) #Some video settings.
#Ani is the actual animations. interval is interval between frames in microsecons. Blit is what speeds up the animation process
#by only plotting the changes rather then replotting each frame.
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=20, repeat = True, blit = True)
#ani = animation.FuncAnimation(fig, animate, frames=num_of_frames-1, interval=200, repeat=True, blit = True) #The actual animator
plt.show()
ani.save('/Users/nickvalverde/Dropbox/Research/ORISS/Movie_Making/color_map.mp4', writer=writer)
