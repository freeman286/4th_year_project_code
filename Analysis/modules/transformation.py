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
