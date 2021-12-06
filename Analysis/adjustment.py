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

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

aligned_points = np.zeros((point_count, 3))
current_points = np.zeros((point_count, 3))
collated_points = np.zeros((point_count, reading_count, 3))

base_points = np.zeros((reading_count, 3))
pressure_offset = [0.100, 0.100, -0.200]
pressure_points = np.zeros((reading_count, 3))

depths = np.zeros((reading_count))

#Least squares adjustment
k = np.ones((point_count, 3 * reading_count, 1)) #Residuals
A = np.zeros((point_count, 3 * reading_count, 3)) #Adjustment shape
m = np.zeros((point_count, 3 * reading_count, 1)) #Distance and angles in aligned frame of reference
d = np.zeros((point_count, 3 * reading_count, 1)) #Mean distance and angles in aligned frame of reference (initial guess)

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

        return np.dot(np.subtract(input_points, np.tile(input_centroid, (point_count, 1))), r.T), pressure_point, np.matmul(r, -input_centroid)

    else:
        return np.subtract(input_points, np.tile(input_centroid, (point_count, 1))), np.subtract(pressure_offset, input_centroid), -input_centroid

def rotation_matrix_from_vectors(v1, v2): #Rotation matrix from v1 to v2
    a, b = (v1 / np.linalg.norm(v1)).reshape(3), (v2 / np.linalg.norm(v2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
        
def find_water_surface_transformation():
    pressure_centroid = np.mean(pressure_points, axis=0)
    normalised_pressure_points = np.subtract(pressure_points,  np.tile(pressure_centroid, (reading_count, 1)) )
    
    depth_mean = np.mean(depths)
    normalised_depths = depths - depth_mean

    A = np.array([
        [np.sum(normalised_pressure_points[:,0]**2), np.sum(normalised_pressure_points[:,0]*normalised_pressure_points[:,1]), np.sum(normalised_pressure_points[:,0]*normalised_pressure_points[:,2])],
        [np.sum(normalised_pressure_points[:,0]*normalised_pressure_points[:,1]), np.sum(normalised_pressure_points[:,1]**2), np.sum(normalised_pressure_points[:,1]*normalised_pressure_points[:,2])],
        [np.sum(normalised_pressure_points[:,0]*normalised_pressure_points[:,2]), np.sum(normalised_pressure_points[:,1]*normalised_pressure_points[:,2]), np.sum(normalised_pressure_points[:,2]**2)],
    ])
    
    B = -np.array([[
        np.sum(normalised_pressure_points[:,0] * normalised_depths),
        np.sum(normalised_pressure_points[:,1] * normalised_depths),
        np.sum(normalised_pressure_points[:,2] * normalised_depths)
    ]]).T

    normal_vector = np.matmul(np.linalg.inv(A), B)

    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    r = rotation_matrix_from_vectors(normal_vector, np.array([[0,0,1]]).T)

    return r, depth_mean + pressure_centroid[2]

def mean():
    means = np.zeros((point_count, 3))
    for n in range(point_count):
        collated_points[n,:,:] = aligned_points[n:(point_count*reading_count+n):point_count]
        means[n] = np.mean(collated_points[n,:,:], axis = 0)

    return means

def least_squares_adjustment_setup():
    global k
    for n in range(point_count):
        ref_points = np.subtract(collated_points[n,:,:], base_points)
        mean_ref_points = np.subtract(np.tile(means[n], (reading_count,1)), base_points)
        
        for reading in range(reading_count):
            az, el, r = cart2sph(ref_points[reading, 0], ref_points[reading, 1], ref_points[reading, 2])
            m[n, 3*reading, 0] = az
            m[n, 3*reading+1, 0] = el
            m[n, 3*reading+2, 0] = r

            az, el, r = cart2sph(mean_ref_points[reading, 0], mean_ref_points[reading, 1], mean_ref_points[reading, 2])
            d[n, 3*reading, 0] = az
            d[n, 3*reading+1, 0] = el
            d[n, 3*reading+2, 0] = r

    k = np.subtract(m,d)
         

def least_squares_adjustment():
    least_squares_adjustment_setup()
        
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

    depths[j] = depths[j] + angles[3]

    current_points[i] = [x, y, z]

    if(i == point_count-1):
        new_points, pressure_points[j], base_points[j] = kabsch_rotated_points(current_points)
        if (j==0):
            aligned_points = new_points
        else:
            aligned_points = np.concatenate((aligned_points, new_points), axis=0)
            
        depths[j] = depths[j] / point_count
        j += 1
        i = 0
    else:
        i += 1


#Adjust everything to be in the same frame of reference as the water surface
r, h = find_water_surface_transformation()
aligned_points = np.subtract(np.dot(aligned_points, r.T), np.tile(np.array([[0,0,h]]), (reading_count * point_count, 1)))
pressure_points = np.subtract(np.dot(pressure_points, r.T), np.tile(np.array([[0,0,h]]), (reading_count, 1)))
base_points = np.subtract(np.dot(base_points, r.T), np.tile(np.array([[0,0,h]]), (reading_count, 1)))

means = mean()

least_squares_adjustment()
ax.scatter(means[:,0], means[:,1], means[:,2], color='red', label='mean points')  
ax.scatter(pressure_points[:,0], pressure_points[:,1], pressure_points[:,2], color='blue', label='pressure locations')
ax.scatter(base_points[:,0], base_points[:,1], base_points[:,2], color='green', label='base locations')

    
ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-2, 2)
ax.legend()
pylab.show()

