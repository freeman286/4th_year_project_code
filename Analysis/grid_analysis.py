import csv
import os
import numpy as np
from numpy import genfromtxt

rows = 5
columns = 4

original_path = os.getcwd() + '/points/grid_test.csv'
original_points = genfromtxt(original_path, delimiter=',')

new_path = os.getcwd() + '/results/grid_test.csv'
new_points = genfromtxt(new_path, delimiter=',')

original_horizontal_distances = np.zeros((columns-1, rows))
original_vertical_distances = np.zeros((columns, rows-1))
new_horizontal_distances = np.zeros((columns-1, rows))
new_vertical_distances = np.zeros((columns, rows-1))

for i, row in enumerate(original_horizontal_distances):
    for j, element in enumerate(row):
        original_horizontal_distances[i,j] = np.linalg.norm(original_points[i+j*columns]-original_points[i+j*columns+1])
        new_horizontal_distances[i,j] = np.linalg.norm(new_points[i+j*columns]-new_points[i+j*columns+1])

for i, row in enumerate(original_vertical_distances):
    for j, element in enumerate(row):
        original_vertical_distances[i,j] = np.linalg.norm(original_points[i*columns+j]-original_points[(i+1)*columns+j])
        new_vertical_distances[i,j] = np.linalg.norm(new_points[i*columns+j]-new_points[(i+1)*columns+j])
