import csv
import os
import numpy as np

spool_radius = 37.5e-3
sensor_error = np.deg2rad(0.1)
spool_offset = 0.281

read_path = os.getcwd() + '/points/test.csv'
write_path = os.getcwd() + '/angles/test.csv'

read_file = open(read_path, "r")
write_file = open(write_path, "w")


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

reader = csv.reader(read_file)

for line in reader:
    az, el, r = cart2sph(float(line[0]),float(line[1]),float(line[2]))

    angles = np.degrees(np.add([az, el, np.pi*(r-spool_offset)/spool_radius], np.random.uniform(low = -sensor_error, high = sensor_error, size=(1,3))))[0]
    
    write_file.write(', '.join(map('{0:.2f}'.format, angles)) + '\n')


write_file.close()
