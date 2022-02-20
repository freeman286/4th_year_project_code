import numpy as np

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
    return np.argmin(dist)

def kabsch_rotation(mapping_points, aligned_points):
    h = mapping_points.T @ aligned_points
    u, s, vt = np.linalg.svd(h)
    v = vt.T

    d = np.linalg.det(v @ u.T)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])

    return v @ e @ u.T
