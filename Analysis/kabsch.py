import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab

spool_radius = 37.5e-3
spool_offset = 0.281

reading_count = 4

read_path = os.getcwd() + '/angles/test.csv'
read_file = open(read_path, "r")

file_reader = csv.reader(read_file)

lines = len(list(file_reader))
point_count = int(lines / reading_count)-1

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

aligned_points = np.zeros((point_count, 3))
current_points = np.zeros((point_count, 3))

pressure_offset = [0.100, 0.100, -0.200]
pressure_points = np.zeros((reading_count, 3))

def kabsch_rotated_points(input_points):

    input_centroid = np.mean(input_points, axis=0)
    
    if (aligned_points.any()):
        repeat_factor = int(aligned_points.shape[0]/input_points.shape[0])

        mapping_points = np.subtract( np.tile(input_points, (repeat_factor, 1)),  np.tile(input_centroid, (repeat_factor * point_count, 1)) ) #Get the centroid at the origin
        
        h = mapping_points.T @ aligned_points
        u, s, vt = np.linalg.svd(h)
        v = vt.T

        d = np.linalg.det(v @ u.T)
        e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

        r = v @ e @ u.T

        pressure_point = np.matmul(r, np.subtract(pressure_offset, input_centroid))

        return np.dot(np.subtract(input_points, np.tile(input_centroid, (point_count, 1))), r.T), pressure_point

    else:
        return np.subtract(input_points, np.tile(input_centroid, (point_count, 1))), np.subtract(pressure_offset, input_centroid)
        
        
fig = plt.figure()
ax = fig.gca(projection='3d')

i = 0 #Element in set of points
j = 0 #Which set of points we are on
read_file.seek(0)
for line in file_reader:
    angles = np.asarray(line, dtype=np.float64)
    
    if (angles.size == 0):
        continue #Skip if we have an empty line

    r = np.deg2rad(angles[2]) * spool_radius + spool_offset

    x, y, z = sph2cart(np.deg2rad(angles[0]), np.deg2rad(angles[1]), r)

    current_points[i] = [x, y, z]

    if(i == point_count-1):
        new_points, pressure_points[j] = kabsch_rotated_points(current_points)
        aligned_points = np.concatenate((aligned_points, new_points), axis=0)
        ax.scatter(new_points[:,0], new_points[:,1], new_points[:,2], color='C'+str(j), label='C'+str(j))
        j += 1
        i = 0
    else:
        i += 1
        
ax.scatter(pressure_points[:,0], pressure_points[:,1], pressure_points[:,2], color='C'+str(reading_count), label='C'+str(reading_count))
    
ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-2, 2)
ax.legend()
pylab.show()
