import numpy as np
import matplotlib.pyplot as plt

# Get lighting object for shading surface plots.
from matplotlib.colors import LightSource

# Get colormaps to use with lighting object.
from matplotlib import cm

ellipsoid_res = 20 #Resolution of error ellipsoids
ellipsoid_scale_factor = 10 #Scale factor of error ellipsoids

big_tick_locator = 0.5
small_tick_locator = 0.1

linewidth = 2

labelpad = 40

fontsize = 20

font = {'family' : 'sans',
        'size'   : fontsize}

plt.rc('font', **font)
plt.rc('legend',fontsize=fontsize) # using a size in points

def plot_ellispoid(origin, w, v):

    U = np.linspace(0, 2 * np.pi, ellipsoid_res)
    V = np.linspace(0, np.pi, ellipsoid_res)

    x = ellipsoid_scale_factor * w[0] * np.outer(np.cos(U), np.sin(V))
    y = ellipsoid_scale_factor * w[1] * np.outer(np.sin(U), np.sin(V))
    z = ellipsoid_scale_factor * w[2] * np.outer(np.ones(np.size(U)), np.cos(V))

    x_dash = x * v[0,0] + y * v[0,1] + z * v[0,2] + np.tile(origin[0], (ellipsoid_res, ellipsoid_res))
    y_dash = x * v[1,0] + y * v[1,1] + z * v[1,2] + np.tile(origin[1], (ellipsoid_res, ellipsoid_res))
    z_dash = x * v[2,0] + y * v[2,1] + z * v[2,2] + np.tile(origin[2], (ellipsoid_res, ellipsoid_res))

    # Plot the surface
    light = LightSource(90, 45,hsv_min_val=0.6)
    rgb = np.ones((z_dash.shape[0], z_dash.shape[1], 3))
    purple = np.array([1.0,0,1.0])
    purple_surface = light.shade_rgb(rgb * purple, z_dash)
    ax = plt.gca()
    ax.plot_surface(x_dash, y_dash, z_dash, antialiased=False,facecolors=purple_surface, alpha=0.2, linewidth=0)



def format_axis(ax, tick_size, X, Y, Z):
    # Gridline thickness
    ax.xaxis._axinfo["grid"].update({"linewidth":linewidth})
    ax.yaxis._axinfo["grid"].update({"linewidth":linewidth})
    ax.zaxis._axinfo["grid"].update({"linewidth":linewidth})

    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_linewidth(1.5*linewidth)

    ax.xaxis.set_major_locator(plt.MultipleLocator(tick_size))
    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_size))
    ax.zaxis.set_major_locator(plt.MultipleLocator(tick_size))

    plt.xticks(rotation=90)
    plt.yticks(rotation=90)

    ax.xaxis.labelpad = labelpad
    ax.yaxis.labelpad = labelpad
    ax.zaxis.labelpad = labelpad

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    # Bounding box
    ax.set_aspect('equal')
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    plt.tick_params(axis='x',labelsize=0.75*fontsize)
    plt.tick_params(axis='y',labelsize=0.75*fontsize)

    # fix z-axis ticks
    plt.tick_params(
        axis='z',
        which='both',
        pad=0.5*labelpad,
        labelsize=0.75*fontsize
    )
