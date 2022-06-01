#!/usr/bin/python

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import pylab
from modules.transformation import *
from modules.graphing import *
from modules.config import *
import time

reading_count = 5

read_path = os.getcwd() + '/angles/data2022.02.24.17.02.08.csv'
read_file = open(read_path, "r")

write_path = os.getcwd() + '/results/data2022.02.24.17.02.08.csv'

file_reader = csv.reader(read_file)

lines = len(list(file_reader))
point_count = round(lines / reading_count-1)

aligned_points = np.zeros((point_count, 3))
current_points = np.zeros((point_count, 3))
collated_points = np.zeros((point_count, reading_count, 3))

base_points = np.zeros((reading_count, 3))
pressure_points = np.zeros((reading_count, 3))

depths = np.zeros((reading_count))

ESDangle = sensor_error_sd #Estimate of our standard deviation in sensors
ESDdist = ESDangle * spool_radius

#Least squares adjustment
k = np.ones((point_count, 3 * reading_count, 1)) #Residuals
A = np.zeros((point_count, 3 * reading_count, 3)) #Adjustment shape
m = np.zeros((point_count, 3 * reading_count, 1)) #Distance and angles in aligned frame of reference
d = np.zeros((point_count, 3 * reading_count, 1)) #Mean distance and angles in aligned frame of reference (initial guess)
R = np.zeros((point_count, 3 * reading_count, 3 * reading_count)) #Reciprocal of ESDs matrix
W = np.zeros((point_count, 3 * reading_count, 3 * reading_count)) #Weight matrix

sds = np.zeros((point_count, 3)) #Error ellipsoid size
v = np.zeros((point_count, 3, 3)) #Error ellipsoid principal axis

sigma_v = np.zeros(point_count)

LSQ_points = np.zeros((point_count, 3))

depth_sd = 0

def kabsch_rotated_points(input_points):

    input_centroid = np.mean(input_points, axis=0)

    if (aligned_points.any()):
        repeat_factor = int(aligned_points.shape[0]/input_points.shape[0])

        mapping_points = np.subtract( np.tile(input_points, (repeat_factor, 1)),  np.tile(input_centroid, (repeat_factor * point_count, 1)) ) #Get the centroid at the origin

        r = kabsch_rotation(mapping_points, aligned_points)

        pressure_point = np.matmul(r, np.subtract(pressure_offset, input_centroid))

        return np.dot(np.subtract(input_points, np.tile(input_centroid, (point_count, 1))), r.T), pressure_point, np.matmul(r, -input_centroid)

    else:
        return np.subtract(input_points, np.tile(input_centroid, (point_count, 1))), np.subtract(pressure_offset, input_centroid), -input_centroid


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

    if (np.all((B == 0))): #Check if we set up on a flat surface
        r = np.identity(3)
    else:
        normal_vector = np.matmul(np.linalg.inv(A), B)

        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        r = rotation_matrix_from_vectors(normal_vector, np.array([[0,0,1]]).T)

    return r, depth_mean + pressure_centroid[2]

def find_water_surface_error():
    return np.sum(np.square(np.add(pressure_points[:,2], depths)))/reading_count

def mean():
    means = np.zeros((point_count, 3))
    for n in range(point_count):
        collated_points[n,:,:] = aligned_points[n:(point_count*reading_count+n):point_count]
        means[n] = np.mean(collated_points[n,:,:], axis = 0)

    return means

def least_squares_adjustment_setup(): #Set up the matrixes for least squares adjustment
    global k
    for n in range(point_count):
        ref_points = np.subtract(collated_points[n,:,:], base_points)
        mean_ref_points = np.subtract(np.tile(means[n], (reading_count,1)), base_points)

        for reading in range(reading_count):
            az_m, el_m, r_m = cart2sph(ref_points[reading, 0], ref_points[reading, 1], ref_points[reading, 2])
            m[n, 3*reading, 0] = az_m
            m[n, 3*reading+1, 0] = el_m
            m[n, 3*reading+2, 0] = r_m

            az_d, el_d, r_d = cart2sph(mean_ref_points[reading, 0], mean_ref_points[reading, 1], mean_ref_points[reading, 2])
            d[n, 3*reading, 0] = az_d
            d[n, 3*reading+1, 0] = el_d
            d[n, 3*reading+2, 0] = r_d

            A[n, 3*reading, :] = [-np.cos(az_m)*np.sin(el_m)/r_m, np.cos(az_m)/r_m, -np.sin(az_m)*np.sin(el_m)/r_m]
            A[n, 3*reading+1, :] = [-np.sin(el_m)/r_m, 0, np.cos(el_m)/r_m]
            A[n, 3*reading+2, :] = [np.cos(az_m)*np.cos(el_m), np.sin(az_m), np.cos(az_m)*np.sin(el_m)]


        np.fill_diagonal(R[n, :, :], np.tile(np.array([1/ESDangle, 1/ESDangle, 1/ESDdist]), (1, reading_count)))

    k = np.subtract(m,d)


def least_squares_adjustment():
    least_squares_adjustment_setup()

    x_bar = np.array([0,0,0], dtype = float)

    for n in range(point_count):
        W[n,:,:] = np.matmul(R[n,:,:],R[n,:,:])

        x = np.linalg.inv((A[n,:,:].T).dot(W[n,:,:]).dot(A[n,:,:])).dot(A[n,:,:].T).dot(W[n,:,:]).dot(k[n,:,:])

        x_bar = np.add(x_bar,x.T)

        LSQ_points[n] = means[n] + x.T

        means[n] = LSQ_points[n]

        sigma_v[n] = np.sqrt( (k[n,:,:].T.dot(W[n,:,:]).dot(k[n,:,:])) /(reading_count * 3 - 3))

        var_x = np.linalg.inv((A[n,:,:].T).dot(W[n,:,:]).dot(A[n,:,:])) * sigma_v[n] ** 2

        w, u = np.linalg.eig(var_x)

        v[n,:,:] = u

        sds[n] = np.sqrt(w)

    return x_bar/point_count

i = 0 #Element in set of points
j = 0 #Which set of points we are on
read_file.seek(0)
for line in file_reader:
    angles = np.asarray(line[0:4], dtype=np.float64)

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

depth_sd = np.sqrt(find_water_surface_error())

means = mean()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while (not np.isclose(np.mean(sigma_v),1, atol=1e-15)): #Iterate until we have sensible standard deviation
    x_bar = 1
    while (np.linalg.norm(x_bar) > 1e-15):
        x_bar = least_squares_adjustment()
    ESDangle *= np.mean(sigma_v)
    ESDdist = ESDangle * spool_radius

np.savetxt(write_path, LSQ_points, fmt='%f', delimiter=',')

#for n in range(point_count):
#    plot_ellispoid(LSQ_points[n], sds[n], v[n])

def init():
    ax.scatter(LSQ_points[:,0], LSQ_points[:,1], LSQ_points[:,2], color='purple', label='measured points')
    ax.scatter(pressure_points[:,0], pressure_points[:,1], pressure_points[:,2], color='blue', label='pressure locations')
    ax.scatter(base_points[:,0], base_points[:,1], base_points[:,2], color='green', label='base locations')

    format_axis(ax,
                big_tick_locator,
                np.concatenate((LSQ_points[:,0],pressure_points[:,0],base_points[:,0])),
                np.concatenate((LSQ_points[:,1],pressure_points[:,1],base_points[:,1])),
                np.concatenate((LSQ_points[:,2],pressure_points[:,2],base_points[:,2]))
                )

    #format_axis(ax,
    #            small_tick_locator,
    #            LSQ_points[:,0],
    #            LSQ_points[:,1],
    #            LSQ_points[:,2]
    #            )

    ax.legend()

    return fig,

def animate(i):
    print("frame " + str(i))
    ax.view_init(elev=30., azim=i)
    return fig,

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
fig.set_size_inches(graph_width, graph_height, True)

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)

# Save
#anim.save('animation.gif', fps=30, writer='pillow',dpi=dpi)

writergif = animation.PillowWriter(fps = 30)

# fig.dpi is default dpi, but can also be specified explicitly if preferred
writergif.setup(fig, "animation.gif", dpi = dpi)

# May or may not need to specify dpi argument
anim.save("animation.gif", writer = writergif, dpi = dpi)
