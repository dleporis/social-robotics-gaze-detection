import numpy as np
import matplotlib.pyplot as plt

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
horizontal_fov = np.deg2rad(120)  # Convert to radians
vertical_fov = np.deg2rad(60)

while True:
    #all_points = [l_eye, r_eye, nose, l_ear, r_ear]
    #l_eye, r_eye, nose, l_ear, r_ear = apply_random_transformation(all_points)
    
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

    ###
    # Define the parameters
    alpha = horizontal_fov / 2  # Angle in radians
    beta = vertical_fov /2  # Angle in radians
    length_centroid = np.linalg.norm(gaze_direction_3d)
    print("length_centroid")
    print(length_centroid)
    base_side_a = np.tan(alpha) * length_centroid * 2
    base_side_b = np.tan(beta) * length_centroid * 2
    print("base_side_a")
    print(base_side_a)
    print("base_side_b")
    print(base_side_b)

    # Calculate the base center
    base_center = eye_midpoint + gaze_direction_3d
    print(base_center)
    base_points = np.zeros((4, 3))
    
    # 1 Calculate the direction of the line
    base_h_direction = l_eye - r_eye

    # Calculate the normalized direction of the line
    base_h_direction_unit = base_h_direction / np.linalg.norm(base_h_direction)

    # Calculate the line segment
    base_h = np.vstack((base_center, base_center + base_h_direction_unit))

    # 2 Calculate the normal vector of the plane
    base_normal = gaze_direction_3d

    # Define a point on the plane (base_h_point can be used)
    base_point = base_h[0]

    # Define the plane by its normal and a point on the plane
    base_plane = (base_normal, base_point)

    # 3
    # Calculate the cross product of base_h_direction_unit and base_normal to get base_v_direction
    base_v_direction = np.cross(base_h_direction_unit, base_normal)

    # Calculate the normalized direction of the line
    base_v_direction_unit = base_v_direction / np.linalg.norm(base_v_direction)

    # Calculate the line segment
    base_v = np.vstack((base_center, base_center + base_v_direction_unit))


    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 4
    # Plot the base plane
    base_origin = base_plane[1]
    base_normal_scaled = base_plane[0] * 10  # Scaling the normal vector for visualization
    base_points = np.array([base_origin - base_normal_scaled, base_origin + base_normal_scaled])
    ax.plot(*base_points.T, color='cyan', label='base')

    # Plot the base_h line segment
    ax.plot(*base_h.T, color='red', label='base_h')

    # Plot the base_v line segment
    ax.plot(*base_v.T, color='green', label='base_v')
    # 5 Calculate the corners of the base rectangle
    corner_1 = base_center + 0.5 * base_side_a * base_h_direction_unit + 0.5 * base_side_b * base_v_direction_unit
    corner_2 = base_center - 0.5 * base_side_a * base_h_direction_unit + 0.5 * base_side_b * base_v_direction_unit
    corner_3 = base_center - 0.5 * base_side_a * base_h_direction_unit - 0.5 * base_side_b * base_v_direction_unit
    corner_4 = base_center + 0.5 * base_side_a * base_h_direction_unit - 0.5 * base_side_b * base_v_direction_unit

    # Define the base rectangle by its corners
    base_rectangle = np.vstack((corner_1, corner_2, corner_3, corner_4, corner_1))

    # 6 # Plot the base_rectangle
    ax.plot(*base_rectangle.T, color='blue', label='base_rectangle')

    # Plot lines from eye_midpoint to base_rectangle vertices
    ax.plot([eye_midpoint[0], corner_1[0]], [eye_midpoint[1], corner_1[1]], [eye_midpoint[2], corner_1[2]], color='magenta')
    ax.plot([eye_midpoint[0], corner_2[0]], [eye_midpoint[1], corner_2[1]], [eye_midpoint[2], corner_2[2]], color='magenta')
    ax.plot([eye_midpoint[0], corner_3[0]], [eye_midpoint[1], corner_3[1]], [eye_midpoint[2], corner_3[2]], color='magenta')
    ax.plot([eye_midpoint[0], corner_4[0]], [eye_midpoint[1], corner_4[1]], [eye_midpoint[2], corner_4[2]], color='magenta')

    
    # Define the line parallel to l_eye - r_eye line and perpendicular to gaze_direction_3d
    #line_parallel = np.vstack((eye_midpoint - 0.5 * (l_eye - r_eye), eye_midpoint + 0.5 * (l_eye - r_eye)))
    #line_perpendicular = np.vstack((base_center - 0.5 * length_centroid * gaze_direction_3d, base_center + 0.5 * length_centroid * gaze_direction_3d))


    # Plot the points
    ax.scatter(l_eye[0], l_eye[1], l_eye[2], color='red', label='l_eye')
    ax.scatter(r_eye[0], r_eye[1], r_eye[2], color='green', label='r_eye')
    ax.scatter(nose[0], nose[1], nose[2], color='blue', label='nose')
    ax.scatter(l_ear[0], l_ear[1], l_ear[2], color='orange', label='l_ear')
    ax.scatter(r_ear[0], r_ear[1], r_ear[2], color='purple', label='r_ear')

    # Set plot limits and labels
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-20, 20])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add legend
    ax.legend()

    # Show the plot
    plt.show()