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
        # 180 degree rotation around any perpendicular vector
        perp = np.array([1, 0, 0]) if not np.allclose(a, [1, 0, 0]) else np.array([0, 1, 0])
        v = np.cross(a, perp)
        v = v / np.linalg.norm(v)
        H = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        return -np.identity(3) + 2 * np.outer(v, v)
    s = np.linalg.norm(v)
    kmat = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    R = np.identity(3) + kmat + (kmat @ kmat) * ((1 - c) / (s ** 2))
    return R

def plot_sphere_with_face_direction(azimuth_deg, elevation_deg):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    # Sphere mesh
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='lightgrey', alpha=0.3, edgecolor='none')

    # Base longitude (vertical) and latitude (horizontal) lines
    t = np.linspace(0, 2 * np.pi, 400)
    # horizontal line (in x-y plane at z = 0)
    horizontal = np.vstack([np.cos(t), np.sin(t), np.zeros_like(t)])
    # vertical line (in x-z plane at y = 0)
    vertical = np.vstack([np.cos(t), np.zeros_like(t), np.sin(t)])

    # Rotate from default "forward" ([1, 0, 0]) to given direction
    forward = np.array([1, 0, 0])
    target = spherical_to_cartesian(azimuth_deg, elevation_deg)
    R = rotation_matrix_from_vectors(forward, target)

    horiz_rot = R @ horizontal
    vert_rot = R @ vertical

    ax.plot(horiz_rot[0], horiz_rot[1], horiz_rot[2], color='red', linewidth=1.5)
    ax.plot(vert_rot[0], vert_rot[1], vert_rot[2], color='red', linewidth=1.5)

    # Draw direction vector representing "forward-facing direction"
    ax.quiver(0, 0, 0, *target, color='darkred', linewidth=2)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.axis('off')
    plt.title(f"Face Direction: Azimuth = {azimuth_deg}°, Elevation = {elevation_deg}°")
    plt.show()

# Example
plot_sphere_with_face_direction(45, 30)
