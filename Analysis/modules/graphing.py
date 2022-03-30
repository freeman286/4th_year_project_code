import numpy as np
import matplotlib.pyplot as plt

ellipsoid_res = 10 #Resolution of error ellipsoids
ellipsoid_scale_factor = 1 #Scale factor of error ellipsoids

big_tick_locator = 0.25
small_tick_locator = 0.1

labelpad = 20

font = {'family' : 'sans',
        'size'   : 20}

plt.rc('font', **font)

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
    ax = plt.gca()
    ax.plot_surface(x_dash, y_dash, z_dash, color='purple')

def format_axis(ax, tick_size):
    ax.xaxis.set_major_locator(plt.MultipleLocator(tick_size))
    ax.yaxis.set_major_locator(plt.MultipleLocator(tick_size))
    ax.zaxis.set_major_locator(plt.MultipleLocator(tick_size))

    ax.xaxis.labelpad = labelpad
    ax.yaxis.labelpad = labelpad
    ax.zaxis.labelpad = labelpad

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
