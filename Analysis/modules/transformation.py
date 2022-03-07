import numpy as np
from collections import Counter

def sph2cart(az, el, r):
    rcos_phi = r * np.cos(az)
    x = rcos_phi * np.cos(el)
    y = r * np.sin(az)
    z = rcos_phi * np.sin(el)
    return x, y, z

def cart2sph(x, y, z):
    hxz = np.hypot(z, x)
    r = np.hypot(y, hxz)
    el = np.arctan2(z, x)
    az = np.arctan2(y, hxz)
    return az, el, r

def rotation_matrix_from_vectors(v1, v2): #Rotation matrix from v1 to v2
    a, b = (v1 / np.linalg.norm(v1)).reshape(3), (v2 / np.linalg.norm(v2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def closest_point(point, points): #Return the index of the closest point
    dist = np.linalg.norm(point-points, axis=1)
    return np.argmin(dist), dist

def sort_points(points, cloud):
    sorted_points = np.zeros(np.shape(points))
    length = np.shape(points)[0]
    indexes = np.zeros(length)
    distances = np.zeros((length,length))
    for i, point in enumerate(points):
        ind, dist = closest_point(point, cloud)
        indexes[i] = ind
        distances[i] = dist
        sorted_points[ind,:] = point

    #duplicates = [int(f) for f in list((Counter(indexes) - Counter(set(indexes))).keys())]

    #for duplicate in duplicates:
    #    locations = np.where(indexes == duplicate)[0]
    #    i = np.argmax(distances[locations,duplicates])
    #    j = np.argmin(distances[locations,duplicates])
    #
    #    dist = np.linalg.norm(points[locations[i]]-cloud, axis=1)
    #    new_index = np.argsort(dist)[1]
    #    sorted_points[new_index,:] = points[locations[j]]

    return sorted_points

def kabsch_rotation(mapping_points, aligned_points):
    h = mapping_points.T @ aligned_points
    u, s, vt = np.linalg.svd(h)
    v = vt.T

    d = np.linalg.det(v @ u.T)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    return v @ e @ u.T
