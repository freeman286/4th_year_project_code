import csv
import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from modules.transformation import *
from modules.graphing import *

original_path = os.getcwd() + '/points/grid_calibration.csv'
original_points = genfromtxt(original_path, delimiter=',')

new_path = os.getcwd() + '/results/data2022.03.06.13.01.34.csv'
new_points = genfromtxt(new_path, delimiter=',')

point_count = np.shape(original_points)[0]

# Fit a plane through the new points to map them back onto the original points
new_centroid = np.mean(new_points, axis=0, keepdims=True)
old_centroid = np.mean(original_points, axis=0, keepdims=True)
svd = np.linalg.svd(np.transpose(new_points) -  np.transpose(new_centroid))

# Extract the left singular vectors
left = svd[0]
normal = left[:, -1]

r = rotation_matrix_from_vectors(normal, [1, 0, 0])

# Align around centroids
original_points = original_points - old_centroid
aligned_points = np.transpose(np.matmul(r, np.transpose(new_points - new_centroid)))

#sorted_points = np.zeros(np.shape(original_points))

# Sometimes our aligned points come out in the wrong order so we sort them back onto the original points
#sorted_points = sort_points(aligned_points, original_points)

sorted_points = np.multiply(aligned_points, np.tile(np.sign(np.multiply(original_points[0],aligned_points[0])),(point_count,1)))

# Correct for accidental rotations of the grid about the normal of the plane
r = kabsch_rotation(sorted_points, original_points)
sorted_points = np.transpose(np.matmul(r, np.transpose(sorted_points)))

error = np.linalg.norm(original_points-sorted_points,axis=1)

sd = np.std(error) # Standard deviation of the error

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(sorted_points[:,0], sorted_points[:,1], sorted_points[:,2], color='purple', label='aligned points')
ax.scatter(original_points[:,0], original_points[:,1], original_points[:,2], color='blue', label='original points')

plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

format_axis(ax,
            small_tick_locator,
            sorted_points[:,0],
            sorted_points[:,1],
            sorted_points[:,2]
            )

ax.view_init(elev=0, azim=0)

ax.legend()
pylab.show()
