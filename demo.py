from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d

def midpoint_3d(point1, point2):
    return (point1 + point2) / 2

def head_direction(nose, r_eye, l_eye, r_ear, l_ear):
    eye_midpoint = midpoint_3d(r_eye, l_eye)
    ear_midpoint = midpoint_3d(r_ear, l_ear)

    # Vector between ear midpoint and nose
    vector_ear_nose = nose - ear_midpoint
    # Vector between ear midpoint and eye midpoint
    vector_ear_eye = eye_midpoint - ear_midpoint

    # Calculate the rotation matrix
    desired_direction = np.array([0, 0, 1])  # Desired right direction
    rotation_matrix = np.eye(3)
    rotation_matrix[:, 2] = vector_ear_eye / np.linalg.norm(vector_ear_eye)
    rotation_matrix[:, 0] = np.cross(rotation_matrix[:, 2], desired_direction)
    rotation_matrix[:, 1] = np.cross(rotation_matrix[:, 2], rotation_matrix[:, 0])

    # Calculate the head direction vector
    head_direction = rotation_matrix @ np.array([1, 0, 0])
    magnitude = np.linalg.norm(head_direction)
    normal_head_direction = head_direction / magnitude

    return normal_head_direction, eye_midpoint
    """
    angle = np.arccos(np.dot(vector_ear_nose, vector_ear_eye) / (np.linalg.norm(vector_ear_nose) * np.linalg.norm(vector_ear_eye)))
    half_angle = angle / 2

    vector_eye_ear = ear_midpoint - eye_midpoint
    opposite_dir_vector_eye_ear = vector_eye_ear
    rotation_matrix_1 = np.array([[1, 0, 0],
                                [0, np.cos(-half_angle), -np.sin(-half_angle)],
                                [0, np.sin(-half_angle), np.cos(-half_angle)]])

    rotation_matrix_2 = np.array([[1, 0, 0],
                                [0, np.cos(half_angle), -np.sin(half_angle)],
                                [0, np.sin(half_angle), np.cos(half_angle)]])

    rotation_matrix_3 = np.array([[np.cos(half_angle), -np.sin(half_angle), 0],
                                [np.sin(half_angle), np.cos(half_angle), 0],
                                [0, 0, 1]])

    rotation_matrix_4 = np.array([[np.cos(-half_angle), -np.sin(-half_angle), 0],
                                [np.sin(-half_angle), np.cos(-half_angle), 0],
                                [0, 0, 1]])

    rotation_matrix_5 = np.array([[np.cos(half_angle),0, np.sin(half_angle)],
                                [0, 1, 0],
                                [-np.sin(half_angle), 0, np.cos(half_angle)]])

    rotation_matrix_6 = np.array([[np.cos(-half_angle),0, np.sin(-half_angle)],
                                [0, 1, 0],
                                [-np.sin(-half_angle), 0, np.cos(-half_angle)]])

    head_direction = opposite_dir_vector_eye_ear @ rotation_matrix_6
    magnitude = np.sqrt(np.dot(head_direction,head_direction))
    normal_head_direction = head_direction / magnitude
    return normal_head_direction, eye_midpoint
    """
"""
def gaze_direction(img, pose):
    size = img.shape
    nose, l_eye, r_eye, l_ear, r_ear = get_keypoint_values(pose)
    img_points = np.array([nose, l_eye, r_eye, l_ear, r_ear], dtype="double")
    model_points = np.array(
        [
            (0, 0, 0),
            (-225, 120, -135),
            (225, 120, -135),
            (-350, 85, -350),
            (350, 85, -350),
        ],
        dtype="double",
    )

    focal_length = size[1]
    center = size[1] / 2, size[0] / 2
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype="double",
    )

    dist_coeffs = np.zeros((4, 1))
    flags = [
        cv2.SOLVEPNP_ITERATIVE,
        cv2.SOLVEPNP_P3P,
        cv2.SOLVEPNP_EPNP,
        cv2.SOLVEPNP_AP3P,
        cv2.SOLVEPNP_IPPE,
        cv2.SOLVEPNP_IPPE_SQUARE,
        cv2.SOLVEPNP_SQPNP,
    ]

    with contextlib.suppress(Exception):
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, img_points, camera_matrix, dist_coeffs, flags=flags[6]
        )

    # print(img_points)
    nose_end_point2D, jacobian = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]),
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs,
    )

    # Responsible for the red dots
    #for p in img_points:
    #    cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)

    p1 = int(img_points[0][0]), int(img_points[0][1])
    p2 = int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])
    cv2.line(img, p1, p2, (255, 0, 0), 2)

"""
if __name__ == '__main__':
    parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                        'Press esc to exit, "p" to (un)pause video or process next image.')
    parser.add_argument('-m', '--model',
                        help='Required. Path to checkpoint with a trained model '
                             '(or an .xml file in case of OpenVINO inference).',
                        type=str, required=True)
    parser.add_argument('--video', help='Optional. Path to video file or camera id.', type=str, default='')
    parser.add_argument('-d', '--device',
                        help='Optional. Specify the target device to infer on: CPU or GPU. '
                             'The demo will look for a suitable plugin for device specified '
                             '(by default, it is GPU).',
                        type=str, default='GPU')
    parser.add_argument('--use-openvino',
                        help='Optional. Run network with OpenVINO as inference engine. '
                             'CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.',
                        action='store_true')
    parser.add_argument('--use-tensorrt', help='Optional. Run network with TensorRT as inference engine.',
                        action='store_true')
    parser.add_argument('--images', help='Optional. Path to input image(s).', nargs='+', default='')
    parser.add_argument('--height-size', help='Optional. Network input layer height size.', type=int, default=256)
    parser.add_argument('--extrinsics-path',
                        help='Optional. Path to file with camera extrinsics.',
                        type=str, default=None)
    parser.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')
    args = parser.parse_args()

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    stride = 8
    if args.use_openvino:
        from modules.inference_engine_openvino import InferenceEngineOpenVINO
        net = InferenceEngineOpenVINO(args.model, args.device)
    else:
        from modules.inference_engine_pytorch import InferenceEnginePyTorch
        net = InferenceEnginePyTorch(args.model, args.device, use_tensorrt=args.use_tensorrt)

    canvas_3d = np.zeros((720*2, 1280*2, 3), dtype=np.uint8)
    plotter = Plotter3d(canvas_3d.shape[:2])
    canvas_3d_window_name = 'Canvas 3D'
    cv2.namedWindow(canvas_3d_window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

    window_name_2d = 'ICV 3D Human Pose Estimation'
    cv2.namedWindow(window_name_2d, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name_2d, Plotter3d.mouse_callback)

    file_path = args.extrinsics_path
    if file_path is None:
        file_path = os.path.join('data', 'extrinsics.json')
    with open(file_path, 'r') as f:
        extrinsics = json.load(f)
    R = np.array(extrinsics['R'], dtype=np.float32)
    t = np.array(extrinsics['t'], dtype=np.float32)

    frame_provider = ImageReader(args.images)
    is_video = False
    if args.video != '':
        frame_provider = VideoReader(args.video)
        is_video = True
    base_height = args.height_size
    fx = args.fx

    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    for frame in frame_provider:
        current_time = cv2.getTickCount()
        if frame is None:
            break
        input_scale = base_height / frame.shape[0]
        scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
        if fx < 0:  # Focal length is unknown
            fx = np.float32(0.8 * frame.shape[1])

        inference_result = net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
        """
        print("############### New frame Poses 3D #####################")
        print(poses_3d)
        print("## Same frame Poses 2D ##")
        print(poses_2d)
        """
        edges = []
        if len(poses_3d):
            poses_3d = rotate_poses(poses_3d, R, t)
            poses_3d_copy = poses_3d.copy()
            x = poses_3d_copy[:, 0::4]
            y = poses_3d_copy[:, 1::4]
            z = poses_3d_copy[:, 2::4]
            """
            print("x")
            print(x)
            print("y")
            print(y)
            print("z")
            print(z)
            """
            poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
            print("\n#################\nposes_3d")
            print(poses_3d)
            head_dirs_3d = np.zeros((poses_3d.shape[0], 3))
            all_eye_midpoints_3d = np.zeros((poses_3d.shape[0], 3))
            print("poses_3d.shape, head_dirs_3d.shape, all_eye_midpoints_3d.shape")
            print(poses_3d.shape)
            print(head_dirs_3d.shape)
            print(all_eye_midpoints_3d.shape)
            print("head_dirs_3d")
            print(head_dirs_3d)
            
            print("all_eye_midpoints_3d")
            print(all_eye_midpoints_3d)
            for idx, pose_3d in enumerate(poses_3d):
                # face_names = ['nose' 1, 'r_eye' 15, 'l_eye' 16, 'r_ear' 17, 'l_ear' 18]
                print("\nPose idx")
                print(idx)
                nose = pose_3d[1]
                r_eye = pose_3d[15]
                l_eye = pose_3d[16]
                r_ear = pose_3d[17]
                l_ear = pose_3d[18]
                print("nose, r_eye, l_eye, r_ear, l_ear")
                print(nose)
                print(r_eye)
                print(l_eye)
                print(r_ear)
                print(l_ear)
                head_dir_3d, midpoint_eyes_3d = head_direction(nose, r_eye, l_eye, r_ear, l_ear)
                print("midpoint_eyes")
                print(midpoint_eyes_3d)
                print("head_dir_3d")
                print(head_dir_3d)
                
                head_dirs_3d[idx] = head_dir_3d
                all_eye_midpoints_3d[idx] = midpoint_eyes_3d
            
            edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))

            print("head_dirs_3d")
            print(head_dirs_3d)
            print("all_eye_midpoints_3d")
            print(all_eye_midpoints_3d)
            
        plotter.plot(canvas_3d, poses_3d, edges, head_dirs_3d, all_eye_midpoints_3d, gaze_scale=50)
        
        cv2.imshow(canvas_3d_window_name, canvas_3d)

        
        draw_poses(frame, poses_2d)
        current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
        if mean_time == 0:
            mean_time = current_time
        else:
            mean_time = mean_time * 0.95 + current_time * 0.05
        cv2.putText(frame, 'processing FPS: {}'.format(int(1 / mean_time * 10) / 10),
                    (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        cv2.imshow(window_name_2d, frame)

        key = cv2.waitKey(delay)
        if key == esc_code:
            break
        if key == p_code:
            if delay == 1:
                delay = 0
            else:
                delay = 1
        if delay == 0 or not is_video:  # allow to rotate 3D canvas while on pause
            key = 0
            while (key != p_code
                   and key != esc_code
                   and key != space_code):
                print("\n###################")
                plotter.plot(canvas_3d, poses_3d, edges, head_dirs_3d, all_eye_midpoints_3d, gaze_scale=50)
                cv2.imshow(canvas_3d_window_name, canvas_3d)
                key = cv2.waitKey(33)
            if key == esc_code:
                break
            else:
                delay = 1
