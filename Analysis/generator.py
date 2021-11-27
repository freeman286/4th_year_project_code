import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from scipy.spatial.transform import Rotation as R

spool_radius = 37.5e-3
sensor_error = np.deg2rad(0.1)
pressure_error = 0.16e-3
spool_offset = 0.281

read_path = os.getcwd() + '/points/test.csv'
write_path = os.getcwd() + '/angles/test.csv'

read_file = open(read_path, "r")
write_file = open(write_path, "w")

reading_count = 4

base_locations = np.array([
    [0,0,0],
    [1,-1,0.5],
    [2,1,-0.5],
    [3,1,1]
])

water_level = 2

base_rotations = np.deg2rad([
    [45 , 15, 0],
    [90 , -4, 3],
    [135 , -3, -5],
    [150 , 5, 8]
]) #Rotation about z, y and z axis according to the right hand rule (Euler angles)

pressure_offset = [0.100, 0.100, -0.200]

def cart2sph(x, y, z, base_rotation, base_location):

    #Some matrix rotations to get us in the right frame of reference
    rot = R.from_euler('zyx', -base_rotation)

    [x_dash, y_dash, z_dash] = np.matmul(rot.as_matrix(), np.array([x, y, z]).T)
    
    d = water_level - (np.add(np.array(base_location).T, np.matmul(np.linalg.inv(rot.as_matrix()), np.array(pressure_offset).T)))[2]
    
    hxy = np.hypot(x_dash, y_dash)
    r = np.hypot(hxy, z_dash)
    el = np.arctan2(z_dash, hxy)
    az = np.arctan2(y_dash, x_dash)
    return az, el, r, d

file_reader = csv.reader(read_file)

fig = plt.figure()
ax = fig.gca(projection='3d')

for reading in range(reading_count):
    read_file.seek(0)
    ax.scatter(base_locations[reading][0], base_locations[reading][1],base_locations[reading][2], color='blue')
    for line in file_reader:
        points = np.asarray(line, dtype=np.float64)
        
        x,y,z = np.subtract(points, np.array(base_locations[reading]))

        az, el, r, d = cart2sph(x,y,z, base_rotations[reading], base_locations[reading])

        coords = np.degrees(np.add([az, el, (r-spool_offset)/(spool_radius)], np.random.uniform(low = -sensor_error, high = sensor_error, size=(1,3))))[0]
        
        write_file.write(', '.join(map('{0:.2f}'.format, coords)) + ', {0:.4f}'.format(d + np.random.uniform(low = -sensor_error, high = sensor_error)) + '\n')
        
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
