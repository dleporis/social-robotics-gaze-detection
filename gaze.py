import numpy as np
import matplotlib.pyplot as plt


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

def point_in_cone(cone_vertex, cone_axis, cone_angle, point):
    # Step 1: Define the unit vector along the cone axis
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

def plot_circle_and_vector(origin, gaze_end, resolution, radius, ax):
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
    
    ax.plot(translated_circle_points[:, 0], translated_circle_points[:, 1], translated_circle_points[:, 2], 'b-', label="cone base")
    ax.quiver(origin[0], origin[1], origin[2], gaze_direction[0], gaze_direction[1], gaze_direction[2], color='r',label="gaze_direction")
    """
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
    """
    

def apply_random_transformation(points):
    # Generate random rotation matrix
    rotation_matrix = np.random.rand(3, 3)
    rotation_matrix, _ = np.linalg.qr(rotation_matrix)  # Make it a valid rotation matrix
    
    # Generate random translation vector
    translation_vector = np.random.rand(3)
    
    # Apply transformation to points
    transformed_points = []
    for point in points:
        transformed_point = np.dot(rotation_matrix, point) + translation_vector
        transformed_points.append(transformed_point)
    
    return transformed_points


# Define the points
l_eye = np.array([3.5, 0, 0])
r_eye = np.array([-3.5, 0, 0])
nose = np.array([0, -1.5, -3])

l_ear = np.array([l_eye[0] + 4, 9, -2])
r_ear = np.array([r_eye[0] - 4, 9, -2])

# Define the field of view (FOV)
fov = np.deg2rad(120)  # Convert to radians

while True:
    all_points = [l_eye, r_eye, nose, l_ear, r_ear]
    l_eye, r_eye, nose, l_ear, r_ear = apply_random_transformation(all_points)
    

    """
    eye_midpoint = (l_eye + r_eye) / 2
    ear_midpoint = (l_ear + r_ear) / 2
    A = nose - eye_midpoint

    # calculate
    ear_nose = np.vstack((ear_midpoint, nose))
    eye_parallel_direction = ear_nose[1] - ear_nose[0]
    eye_parallel = np.vstack((eye_midpoint, eye_midpoint + eye_parallel_direction))
    ear_eye_direction = eye_midpoint - ear_midpoint
    ear_eye = np.vstack((ear_midpoint, ear_midpoint + ear_eye_direction))

    eye_parallel_direction_unit = eye_parallel_direction / np.linalg.norm(eye_parallel_direction)
    ear_eye_direction_unit = ear_eye_direction / np.linalg.norm(ear_eye_direction)
    gaze_direction_3d = (eye_parallel_direction_unit + ear_eye_direction_unit) / np.linalg.norm(eye_parallel_direction_unit + ear_eye_direction_unit)
    gaze_direction_3d_length = np.linalg.norm(gaze_direction_3d)
    print("gaze_direction_3d_length")
    print(gaze_direction_3d_length)

    """

    eye_midpoint = (l_eye + r_eye) / 2
    ear_midpoint = (l_ear + r_ear) / 2
    print("eye_midpoint")
    print(eye_midpoint)
    print("ear_midpoint")
    print(ear_midpoint)
    # calculate
    ear_nose = nose - ear_midpoint
    print("ear_nose")
    print(ear_nose)
    ear_eye = eye_midpoint - ear_midpoint
    print("ear_eye")
    print(ear_eye)

    ear_nose_unit = ear_nose / np.linalg.norm(ear_nose)
    print("ear_nose_unit")
    print(ear_nose_unit)
    ear_eye_unit = ear_eye / np.linalg.norm(ear_eye)
    print("ear_eye_unit")
    print(ear_eye_unit)
    gaze_direction_3d = (ear_nose_unit + ear_eye_unit) / np.linalg.norm(ear_nose_unit + ear_eye_unit)
    print("gaze_direction_3d")
    print(gaze_direction_3d)
    ###
    # Define the parameters
    alpha = fov / 2  # Angle in radians
    scale = 100
    gaze_direction_3d_scaled = gaze_direction_3d * scale
    length_centroid = np.linalg.norm(gaze_direction_3d_scaled)
    print("length_centroid")
    print(length_centroid)

    # Calculate the base center
    base_center = eye_midpoint + gaze_direction_3d_scaled
    print(base_center)
    base_points = np.zeros((4, 3))
    
    resolution = 9
    radius = np.tan(alpha) * length_centroid

    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    plot_circle_and_vector(eye_midpoint, base_center, resolution, radius, ax)

    # Generate points and check if they are inside the cone
    for x in range(-500, 500, 20):
        for y in range(-500, 500, 20):
            for z in range(-500, 500, 20):
                pose_edge = np.array([[10, -2, 10],
                                    [x, y, z]])
                is_inside =  point_in_cone(eye_midpoint, gaze_direction_3d_scaled, alpha, pose_edge[1])

                x_coord = pose_edge[1, 0]
                y_coord = pose_edge[1, 1]
                z_coord = pose_edge[1, 2]
                if is_inside:
                    ax.scatter3D(x_coord, y_coord, z_coord, c='green')
                    print(str(pose_edge[1]) + " is inside the fov cone: " + str(is_inside))
                else:
                    pass
                    #ax.scatter3D(x_coord, y_coord, z_coord, c='red', label='Pose Vertex')

    
    # Plot the points
    ax.scatter(l_eye[0], l_eye[1], l_eye[2], color='red', label='l_eye')
    ax.scatter(r_eye[0], r_eye[1], r_eye[2], color='green', label='r_eye')
    ax.scatter(nose[0], nose[1], nose[2], color='blue', label='nose')
    ax.scatter(l_ear[0], l_ear[1], l_ear[2], color='orange', label='l_ear')
    ax.scatter(r_ear[0], r_ear[1], r_ear[2], color='purple', label='r_ear')

    # Set plot limits and labels
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_zlim([-200, 200])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend
    ax.legend()
    # Set plot aspect ratio and equal scale for all axes
    ax.set_box_aspect([1, 1, 1])
    set_axes_equal(ax)

    # Show the plot
    plt.show()