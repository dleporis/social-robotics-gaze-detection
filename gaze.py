import numpy as np
import matplotlib.pyplot as plt
import time
#from mpl_toolkits.mplot3d import Axes3D


import numpy as np

def apply_random_transformation(points):
    # Generate random rotation matrix
    rotation_matrix = np.random.rand(3, 3)
    rotation_matrix, _ = np.linalg.qr(rotation_matrix)  # Make it a valid rotation matrix
    print(rotation_matrix)
    # Generate random translation vector
    translation_vector = np.random.rand(3)
    
    # Apply transformation to points
    transformed_points = []
    for point in points:
        transformed_point = np.dot(rotation_matrix, point) + translation_vector
        transformed_points.append(transformed_point)
    
    return transformed_points


# Define the points
LeftEye = np.array([3.5, 1, 1])
RightEye = np.array([-3.5, 1, 1])
Nose = np.array([0, -1.5, -10])

LeftEar = np.array([LeftEye[0] + 4, 9, -2])
RightEar = np.array([RightEye[0] - 4, 9, -2])

# Define the field of view (FOV)
horizontal_fov = np.deg2rad(120)  # Convert to radians
vertical_fov = np.deg2rad(60)

while True:
    all_points = [LeftEye, RightEye, Nose, LeftEar, RightEar]

    LeftEye, RightEye, Nose, LeftEar, RightEar = apply_random_transformation(all_points)
    print(LeftEye, RightEye, Nose, LeftEar, RightEar)

    EyeMidpoint = (LeftEye + RightEye) / 2
    EarMidpoint = (LeftEar + RightEar) / 2
    A = Nose - EyeMidpoint

    # calculate
    ear_nose = np.vstack((EarMidpoint, Nose))
    eye_parallel_direction = ear_nose[1] - ear_nose[0]
    eye_parallel = np.vstack((EyeMidpoint, EyeMidpoint + eye_parallel_direction))
    ear_eye_direction = EyeMidpoint - EarMidpoint
    ear_eye = np.vstack((EarMidpoint, EarMidpoint + ear_eye_direction))

    eye_parallel_direction_unit = eye_parallel_direction / np.linalg.norm(eye_parallel_direction)
    ear_eye_direction_unit = ear_eye_direction / np.linalg.norm(ear_eye_direction)
    gaze_direction_3d = (eye_parallel_direction_unit + ear_eye_direction_unit) / np.linalg.norm(eye_parallel_direction_unit + ear_eye_direction_unit)


    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(*LeftEye, color='blue', label='LeftEye')
    ax.scatter(*RightEye, color='red', label='RightEye')
    ax.scatter(*Nose, color='green', label='NoseCenter')
    ax.scatter(*EyeMidpoint, color='orange', label='EyeMidpoint')
    #ax.scatter(*A, color='purple', label='A')
    ax.scatter(*LeftEar, color='cyan', label='LeftEar')
    ax.scatter(*RightEar, color='magenta', label='RightEar')
    ax.scatter(*EarMidpoint, color='yellow', label='EarMidpoint')

    # Plot segments
    ax.plot(*ear_nose.T, color='blue', label='ear_nose')
    ax.plot(*eye_parallel.T, color='red', label='eye_parallel')
    #ax.plot(*ear_eye.T, color='green', label='ear_eye')

    # Plot gaze_direction_3d
    origin = EyeMidpoint
    gaze_end = EyeMidpoint + gaze_direction_3d
    ax.plot([origin[0], gaze_end[0]], [origin[1], gaze_end[1]], [origin[2], gaze_end[2]], color='orange', label='gaze_direction_3d')

     # Plot gaze_direction_3d
    origin = EyeMidpoint
    gaze_end = origin + gaze_direction_3d
    ax.plot([origin[0], gaze_end[0]], [origin[1], gaze_end[1]], [origin[2], gaze_end[2]], color='orange', label='gaze_direction_3d')

    # Show the plot
    plt.show()

    # Set plot limits and labels
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

