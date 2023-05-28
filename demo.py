from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np
import datetime

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses
from modules.looking_at import point_in_cone
import logging
from tabulate import tabulate
#import matplotlib.pyplot as plt

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d

def midpoint_3d(point1, point2):
    return (point1 + point2) / 2

def head_direction_old(nose, r_eye, l_eye, r_ear, l_ear):
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

def head_direction(nose, r_eye, l_eye, r_ear, l_ear):
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
    gaze_direction_3d = (eye_parallel_direction_unit + ear_eye_direction_unit) #/ np.linalg.norm(eye_parallel_direction_unit + ear_eye_direction_unit)

    """
    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(*l_eye, color='blue', label='LeftEye')
    ax.scatter(*r_eye, color='red', label='r_eye')
    ax.scatter(*nose, color='green', label='noseCenter')
    ax.scatter(* eye_midpoint, color='orange', label=' eye_midpoint')
    #ax.scatter(*A, color='purple', label='A')
    ax.scatter(*l_ear, color='cyan', label='l_ear')
    ax.scatter(*r_ear, color='magenta', label='r_ear')
    ax.scatter(*ear_midpoint, color='yellow', label='ear_midpoint')

    # Plot segments
    ax.plot(*ear_nose.T, color='blue', label='ear_nose')
    ax.plot(*eye_parallel.T, color='red', label='eye_parallel')
    #ax.plot(*ear_eye.T, color='green', label='ear_eye')

    # Plot gaze_direction_3d
    origin =  eye_midpoint
    gaze_end =  eye_midpoint + gaze_direction_3d
    ax.plot([origin[0], gaze_end[0]], [origin[1], gaze_end[1]], [origin[2], gaze_end[2]], color='orange', label='gaze_direction_3d')

    # Show the plot
    plt.show()
    """
    return gaze_direction_3d, eye_midpoint
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
def finish(input, output, o_file, log_file):
    input.release()
    output.release()
    cv2.destroyAllWindows()
    print("Output: {}\nLog: {}".format(o_file, log_file))


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

    canvas_3d = np.zeros(( 1080, 1080, 3), dtype=np.uint8)
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
        
        height_video_in = 1080
        
        timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
        if args.video == str(0) or args.video == str(1) or args.video == str(2) or args.video == (3):
            width_video_in = 1920
            timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
            output_file = "recorded_videos/"+timestamp+"video"+args.video+".mp4"
            logging_file = "recorded_videos/"+timestamp+"video"+args.video+".log"
        else:
            width_video_in = 2336
            output_file = args.video
            head, sep, tail = output_file.partition('.')
            output_file = head + "gaze_estimation" + timestamp + ".mp4"
            logging_file = head + "gaze_estimation" + timestamp + ".log"
       
        print("Reading: {}".format(args.video))
        print("Writing to: {}".format(output_file))
        frame_provider = VideoReader(args.video, width_video_in, height_video_in)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_out = cv2.VideoWriter(output_file, fourcc, 30.0, (width_video_in+canvas_3d.shape[1], height_video_in))
        # Set up logging
        logging.basicConfig(filename=logging_file, level=logging.INFO)
        headers = ['Frame', 'Person ID', 'midpoint_eyes_3d', "head_dir_3d", "sees_people"]
        is_video = True
    base_height = args.height_size
    fx = args.fx

    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    for frame_index, frame in enumerate(frame_provider):
        frame_data = []
        try:
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
            
            plotter.clear(canvas_3d)
            edges = []
            if len(poses_3d):
                print("############### Frame number: {} ###############".format(frame_index))
                poses_3d = rotate_poses(poses_3d, R, t)
                poses_3d_copy = poses_3d.copy()
                x = poses_3d_copy[:, 0::4]
                y = poses_3d_copy[:, 1::4]
                z = poses_3d_copy[:, 2::4]

                poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

                poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
             
                head_dirs_3d = np.zeros((poses_3d.shape[0], 3))
                all_eye_midpoints_3d = np.zeros((poses_3d.shape[0], 3))
                all_eyes_3d = np.zeros((poses_3d.shape[0],2, 3))

                gaze_scale = 100
                fov_angle = 120
                cone_angle = np.deg2rad(fov_angle / 2)
                seen_people_matrix = np.zeros((poses_3d.shape[0], poses_3d.shape[0]))
                
                for observer_idx, pose_3d in enumerate(poses_3d):
                    seen_people = []
                    # face_names = ['nose' 1, 'r_eye' 15, 'l_eye' 16, 'r_ear' 17, 'l_ear' 18]
                    
                    #print("\n#### Person number: {}".format(observer_idx))
                    
                    nose = pose_3d[1]
                    r_eye = pose_3d[15]
                    l_eye = pose_3d[16]
                    r_ear = pose_3d[17]
                    l_ear = pose_3d[18]
                    
                    head_dir_3d, midpoint_eyes_3d = head_direction(nose, r_eye, l_eye, r_ear, l_ear)
                    

                    head_dirs_3d[observer_idx] = head_dir_3d
                    all_eye_midpoints_3d[observer_idx] = midpoint_eyes_3d
                    all_eyes_3d[observer_idx] = [r_eye, l_eye]

                     
                    for observed_idx, other_pose in enumerate(poses_3d):
                        print("Observer: {} Observed person: {}".format(observer_idx, observed_idx))
                        if observed_idx == observer_idx:
                            print("SKIPPING: Observer {} cannot observe person {} cause it is him/her -self.".format(observer_idx, observed_idx))
                            continue
                        is_inside_fov = False
                        point = 0
                        while is_inside_fov == False and point < 19:
                            #print(other_pose[point])
                            is_inside_fov = point_in_cone(midpoint_eyes_3d, head_dir_3d, cone_angle, other_pose[point], gaze_scale=gaze_scale)
                            point = point + 1

                        if is_inside_fov == True:
                            print("Observer: {} sees observed person: {}\n".format(observer_idx, observed_idx))
                            seen_people = np.append(seen_people, observed_idx)
                        else:
                            print("Observer: {} cannot see person: {}\n".format(observer_idx, observed_idx))
                            
                    print("Observer: {} sees people: {}\n".format(observer_idx, seen_people))
                    frame_datapoint = [frame_index, observer_idx, midpoint_eyes_3d, head_dir_3d, seen_people]
                    frame_data.append(frame_datapoint)  # Add data to the frame

                table = tabulate(frame_data, headers, tablefmt='grid')
                print(table)
                # Log data
                logging.info('\n' + table)
                edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
                fov_pyramids = plotter.plot(canvas_3d, poses_3d, edges, head_dirs_3d, all_eye_midpoints_3d, all_eyes_3d, gaze_scale=gaze_scale)
            

            cv2.imshow(canvas_3d_window_name, canvas_3d)

            
            draw_poses(frame, poses_2d)
            # Print the table for the frame to the terminal
            
            current_time = (cv2.getTickCount() - current_time) / cv2.getTickFrequency()
            if mean_time == 0:
                mean_time = current_time
            else:
                mean_time = mean_time * 0.95 + current_time * 0.05
            cv2.putText(frame, 'processing FPS: {}'.format(int(1 / mean_time * 10) / 10),
                        (40, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            cv2.putText(frame, 'Frame number: {}'.format(frame_index),
                        (40, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255))
            cv2.imshow(window_name_2d, frame)

            concat = np.hstack((canvas_3d, frame))
            video_out.write(concat)
          
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
                    fov_pyramids = plotter.plot(canvas_3d, poses_3d, edges, head_dirs_3d, all_eye_midpoints_3d, all_eyes_3d, gaze_scale=100)
                    cv2.imshow(canvas_3d_window_name, canvas_3d)
                    key = cv2.waitKey(33)
                if key == esc_code:
                    break
                else:
                    delay = 1
        except KeyboardInterrupt:
            print("ctrl+c keyboard interupt detected")
            finish(frame_provider, video_out, output_file, logging_file)

finish(frame_provider, video_out, output_file, logging_file)        