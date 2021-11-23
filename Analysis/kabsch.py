import csv
import os
import numpy as np

spool_radius = 37.5e-3
spool_offset = 0.281

reading_count = 3

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

i = 0
read_file.seek(0)
for line in file_reader:
    angles = np.asarray(line, dtype=np.float64)

    if (angles.size == 0):
        continue

    r = np.deg2rad(angles[2]) * spool_radius + spool_offset

    x, y, z = sph2cart(np.deg2rad(angles[0]), np.deg2rad(angles[1]), r)

    if (i < point_count):
        aligned_points[i] = [x, y, z]
    
    i += 1


#h = mapping_points.T @ aligned_points
#u, s, vt = np.linalg.svd(h)
#v = vt.T

#d = np.linalg.det(v @ u.T)
#e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

#r = v @ e @ u.T
