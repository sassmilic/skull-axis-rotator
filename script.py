import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spherical_to_cartesian(azimuth_deg, elevation_deg, r=1):
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.array([x, y, z])

def rotation_matrix_from_vectors(a, b):
    """Returns a rotation matrix that rotates vector a to vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.allclose(c, 1):
        return np.identity(3)
    if np.allclose(c, -1):
        perp = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
        v = np.cross(a, perp)
        v = v / np.linalg.norm(v)
        return -np.identity(3) + 2 * np.outer(v, v)
    s = np.linalg.norm(v)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    return np.identity(3) + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))

def plot_sphere_with_face_direction(azimuth_deg, elevation_deg, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='lightgrey', alpha=0.15, edgecolor='none')

    t = np.linspace(0, 2 * np.pi, 400)
    horizontal = np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)])
    vertical = np.vstack([np.cos(t), np.zeros_like(t), np.sin(t)])

    forward = np.array([1, 0, 0])
    target = spherical_to_cartesian(azimuth_deg, elevation_deg)
    R = rotation_matrix_from_vectors(forward, target)

    horiz_rot = R @ horizontal
    vert_rot = R @ vertical
    ax.plot(horiz_rot[0], horiz_rot[1], horiz_rot[2], color='red', linewidth=1.5)
    ax.plot(vert_rot[0], vert_rot[1], vert_rot[2], color='red', linewidth=1.5)

    ear_arc = np.vstack([
        np.zeros_like(t),
        np.cos(t),
        np.sin(t)
    ])
    ear_rot = R @ ear_arc
    ax.plot(ear_rot[0], ear_rot[1], ear_rot[2], color='black', alpha=0.3, linewidth=1.2, linestyle='--')

    ax.quiver(0, 0, 0, *target, color='darkred', linewidth=2)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.axis('off')

def plot_sphere_grid(poses):
    n_rows = len(poses)
    n_cols = max(len(row) for row in poses)
    fig = plt.figure(figsize=(3 * n_cols, 3 * n_rows))
    for i, row in enumerate(poses):
        for j, (az, alt) in enumerate(row):
            ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1, projection='3d')
            ax.set_box_aspect([1, 1, 1])
            plot_sphere_with_face_direction(az, alt, ax=ax)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

# Define an optimal minimal non-redundant table
poses = [
    [(0, 60), (45, 60), (90, 60), (135, 60), (180, 60)],
    [(0, 30), (45, 30), (90, 30), (135, 30), (180, 30)],
    [(0, 0), (45, 0), (90, 0), (135, 0), (180, 0)],
    [(0, -30), (45, -30), (90, -30), (135, -30), (180, -30)],
    [(0, -60), (45, -60), (90, -60), (135, -60), (180, -60)],
]

plot_sphere_grid(poses)
