import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from scipy.spatial.transform import Rotation as R
from modules.config import *
from modules.transformation import *

read_path = os.getcwd() + '/points/grid_calibration.csv'
write_path = os.getcwd() + '/angles/experiment.csv'

read_file = open(read_path, "r")
write_file = open(write_path, "w")

reading_count = 6

base_locations = np.array([
    [-1.4,-1,-1.81],
    [-1.9,0,-1.59],
    [-2.5,0.7,-1.52],
    [-2.4,1.4,-1.53],
    [-1.6,1.3,-1.79],
    [-1.5,2.2,-1.84],
])

water_level = 0.2

base_rotations = np.deg2rad([
    [-45 , -4, 0],
    [-10 , -2, 3],
    [0 , -3, -5],
    [15 , -6, 4],
    [30 , -3, -5],
    [40 , -8, 2]
]) #Rotation about z, y and z axis according to the right hand rule (Euler angles)

def mod_cart2sph(x, y, z, base_rotation, base_location):

    #Some matrix rotations to get us in the right frame of reference
    rot = R.from_euler('zyx', -base_rotation)

    [x_dash, y_dash, z_dash] = np.matmul(rot.as_matrix(), np.array([x, y, z]).T)

    d = water_level - (np.add(np.array(base_location).T, np.matmul(np.linalg.inv(rot.as_matrix()), np.array(pressure_offset).T)))[2]

    az, el, r = cart2sph(x_dash, y_dash, z_dash)
    return az, el, r, d

file_reader = csv.reader(read_file)

fig = plt.figure()
ax = fig.gca(projection='3d')

for reading in range(reading_count):
    read_file.seek(0)
    ax.scatter(base_locations[reading][0], base_locations[reading][1],base_locations[reading][2], color='blue')
    direction = np.matmul(R.from_euler('zyx', -base_rotations[reading]).as_matrix(), np.array([[1],[0],[0]]))
    ax.quiver(base_locations[reading][0], base_locations[reading][1],base_locations[reading][2], direction[0], direction[1], direction[2])
    for line in file_reader:
        points = np.asarray(line, dtype=np.float64)

        x,y,z = np.subtract(points, np.array(base_locations[reading]))

        az, el, r, d = mod_cart2sph(x,y,z, base_rotations[reading], base_locations[reading])

        coords = np.degrees(np.add([az, el, (r-spool_offset)/(spool_radius)], np.multiply([1, 1, spool_radius], np.random.normal(0, sensor_error_sd, size=(1,3)))))[0]

        write_file.write(', '.join(map('{0:.2f}'.format, coords)) + ', {0:.4f}'.format(d + np.random.normal(0, pressure_error_sd)) + '\n')

        if (reading == 0):
            ax.scatter(points[0], points[1], points[2], color='red')

    write_file.write('\n')

ax.scatter(base_locations[:,0], base_locations[:,1], base_locations[:,2], color='blue')

write_file.close()
read_file.close()

ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-2, 2)
pylab.show()
