import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab

spool_radius = 37.5e-3
sensor_error = np.deg2rad(0.1)
spool_offset = 0.281

read_path = os.getcwd() + '/points/test.csv'
write_path = os.getcwd() + '/angles/test.csv'

read_file = open(read_path, "r")
write_file = open(write_path, "w")

reading_count = 3

base_locations = [
    [0,0,0],
    [1,0,0],
    [2,0,0]
]

base_locations_rotations = np.deg2rad([
    [45 , 5],
    [90 , -4],
    [135 , -1]
])

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

file_reader = csv.reader(read_file)

fig = plt.figure()
ax = fig.gca(projection='3d')

for reading in range(reading_count):
    read_file.seek(0)
    ax.scatter(base_locations[reading][0], base_locations[reading][1],base_locations[reading][2], color='blue')
    for line in file_reader:
        points = np.asarray(line, dtype=np.float64)
        
        x,y,z = np.subtract(points, np.array(base_locations[reading]))

        az, el, r = np.add(np.array(cart2sph(x,y,z)), -np.append(np.array(base_locations_rotations[reading]), [0]))

        angles = np.degrees(np.add([az, el, np.pi*(r-spool_offset)/spool_radius], np.random.uniform(low = -sensor_error, high = sensor_error, size=(1,3))))[0]

        write_file.write(', '.join(map('{0:.2f}'.format, angles)) + '\n')

        if (reading == 0):
            ax.scatter(points[0], points[1], points[2], color='red')
            
    write_file.write('\n')


write_file.close()

ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-2, 2)
pylab.show()
