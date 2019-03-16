import numpy as np


def rot_z(angle):
    sin = np.sin(angle)
    cos = np.cos(angle)

    return np.array([[cos, -sin], [sin, cos]])
