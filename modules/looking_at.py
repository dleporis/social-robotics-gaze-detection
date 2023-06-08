import numpy as np
import matplotlib.pyplot as plt

def point_in_cone(cone_vertex, cone_axis, cone_angle, point, gaze_scale=100):
    # Step 1: Define the unit vector along the cone axis
    cone_axis = cone_axis * gaze_scale
    axis_vector_magnitude = np.sqrt(cone_axis.dot(cone_axis))
    axis_unit_vector = cone_axis / axis_vector_magnitude

    # Step 2: Define vector point_vector and normalize it
    point_vector = point - cone_vertex
    point_vector_magnitude = np.sqrt(point_vector.dot(point_vector))
    if point_vector_magnitude == 0:
        return True # Point is inside the cone
    point_unit_vector = point_vector / point_vector_magnitude

    # Step 3: Calculate the dot product
    dot_product = np.dot(point_unit_vector, axis_unit_vector)

    # Step 4: Calculate the angle and compare with the cone angle
    angle = np.arccos(dot_product)

    if angle <= cone_angle and point_vector_magnitude <= axis_vector_magnitude:

        return True  # Point is inside the cone

    else:
        return False  # Point is outside the cone
"""
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


def plot_cone(cone_vertex, cone_axis, cone_angle, resolution=100):
    # Generate points on the cone surface
    theta = np.linspace(0, 2 * np.pi, resolution)
    x = cone_vertex[0] + np.cos(theta) * np.sin(cone_angle) * cone_axis[0]
    y = cone_vertex[1] + np.cos(theta) * np.sin(cone_angle) * cone_axis[1]
    z = cone_vertex[2] + np.cos(theta) * np.sin(cone_angle) * cone_axis[2]


def plot_cone(ax, cone_vertex, cone_axis, cone_angle, num_points=100):
    # Generate points on the cone surface
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = cone_vertex[0] + np.cos(theta) * np.sin(cone_angle) * cone_axis[0]
    y = cone_vertex[1] + np.cos(theta) * np.sin(cone_angle) * cone_axis[1]
    z = cone_vertex[2] + np.cos(theta) * np.sin(cone_angle) * cone_axis[2]

    # Plot the cone surface
    ax.plot3D(x, y, z, 'b-', alpha=0.3)

    # Plot the cone base (circle)
    ax.plot3D([cone_vertex[0]], [cone_vertex[1]], [cone_vertex[2]], 'bo', markersize=5, label='Cone Vertex')

    # Plot the lines connecting the vertex to the circle
    for i in range(len(x)):
        ax.plot3D([cone_vertex[0], x[i]], [cone_vertex[1], y[i]], [cone_vertex[2], z[i]], 'b--', alpha=0.3)

# Define the cone properties
cone_origin = np.array([0, 0, 0])
cone_axis = np.array([0, -10, 0])
fov_angle = 120
cone_angle = np.deg2rad(fov_angle / 2)

# Create a 3D plot
fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot the cone


# Generate points and check if they are inside the cone
for x in range(-16, 16, 2):
    for y in range(-16, 16, 2):
        for z in range(-16, 16, 2):
            pose_edge = np.array([[10, -2, 10],
                                  [x, y, z]])
            
            is_inside = point_in_cone(cone_origin, cone_axis, cone_angle, pose_edge[1])
            

            x_coord = pose_edge[1, 0]
            y_coord = pose_edge[1, 1]
            z_coord = pose_edge[1, 2]
            if is_inside:
                ax.scatter3D(x_coord, y_coord, z_coord, c='green', label='Pose Vertex')
                print(str(pose_edge[1]) + " is inside the fov cone: " + str(is_inside))
            else:
                pass
                #ax.scatter3D(x_coord, y_coord, z_coord, c='red', label='Pose Vertex')

plot_cone(ax, cone_origin, cone_axis, cone_angle)
# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('FOV Cone')

# Set plot aspect ratio and equal scale for all axes
ax.set_box_aspect([1, 1, 1])
set_axes_equal(ax)

# Display the plot
plt.show()
"""
