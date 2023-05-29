import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def set_axes_equal(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def plot_circle_and_vector(origin, gaze_end, resolution, radius):
    # Define the gaze direction vector
    gaze_direction = gaze_end - origin

    # Normalize the gaze direction vector
    gaze_direction_norm = gaze_direction / np.linalg.norm(gaze_direction)

    # Generate points for the base circle centered at the origin
    theta = np.linspace(0, 2 * np.pi, resolution)
    circle_points = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])

    # Calculate the rotation axis and angle to align the circle with gaze_direction
    rotation_axis = np.cross([0, 0, 1], gaze_direction_norm)
    rotation_angle = np.arccos(np.dot([0, 0, 1], gaze_direction_norm))

    # Apply the rotation to the circle points
    rotated_circle_points = np.dot(circle_points, rotation_matrix(rotation_axis, rotation_angle))

    # Translate the circle points to align with gaze_end
    translated_circle_points = rotated_circle_points + gaze_end

    # Plot the base circle and the gaze direction vector
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(translated_circle_points[:, 0], translated_circle_points[:, 1], translated_circle_points[:, 2], 'b-')
    ax.quiver(origin[0], origin[1], origin[2], gaze_direction[0], gaze_direction[1], gaze_direction[2], color='r')

    # Set plot limits and labels
    max_range = np.array([translated_circle_points[:, 0].max() - translated_circle_points[:, 0].min(),
                          translated_circle_points[:, 1].max() - translated_circle_points[:, 1].min(),
                          translated_circle_points[:, 2].max() - translated_circle_points[:, 2].min()]).max() / 2.0
    mid_x = (translated_circle_points[:, 0].max() + translated_circle_points[:, 0].min()) * 0.5
    mid_y = (translated_circle_points[:, 1].max() + translated_circle_points[:, 1].min()) * 0.5
    mid_z = (translated_circle_points[:, 2].max() + translated_circle_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set plot aspect ratio and equal scale for all axes
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)
   
    # Show the plot
    plt.show()

def rotation_matrix(axis, angle):
    """
    Generate a rotation matrix for a rotation around the given axis by the specified angle.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    return np.array([[a*a + b*b - c*c - d*d, 2 * (b*c - a*d), 2 * (b*d + a*c)],
                     [2 * (b*c + a*d), a*a + c*c - b*b - d*d, 2 * (c*d - a*b)],
                     [2 * (b*d - a*c), 2 * (c*d + a*b), a*a + d*d - b*b - c*c]])

# Example usage
origin = np.array([1, 1, 1])
gaze_end = np.array([1, 0, 0])
resolution = 9
radius = 1

plot_circle_and_vector(origin, gaze_end, resolution, radius)