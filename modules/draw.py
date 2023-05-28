import math

import cv2
import numpy as np
import contextlib

previous_position = []
theta, phi = 3.1415/4, -3.1415/6
should_rotate = False
scale_dx = 800
scale_dy = 800


class Plotter3d:
    SKELETON_EDGES = np.array([[11, 10], [10, 9], [9, 0], [0, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [0, 12],
                               [12, 13], [13, 14], [0, 1], [1, 15], [15, 16], [1, 17], [17, 18], [6, 12] ]) # 19 eye_midpoint - 20 gaze direction vector end
    def __init__(self, canvas_size, origin=(0.5, 0.5), scale=1):
        self.origin = np.array([origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)  # x, y
        self.scale = np.float32(scale)
        self.theta = 0
        self.phi = 0
        self.fov_angles = [120, 60]
        self.FOV_COLOR = (255,0,255)
        self.FOV_OPACITY = 50
        axis_length = 400
        axes = [
            np.array([[-axis_length/2, -axis_length/2, 0], [axis_length/2, -axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, axis_length/2, 0]], dtype=np.float32),
            np.array([[-axis_length/2, -axis_length/2, 0], [-axis_length/2, -axis_length/2, axis_length]], dtype=np.float32)]
        step = 20
        for step_id in range(axis_length // step + 1):  # add grid
            axes.append(np.array([[-axis_length / 2, -axis_length / 2 + step_id * step, 0],
                                  [axis_length / 2, -axis_length / 2 + step_id * step, 0]], dtype=np.float32))
            axes.append(np.array([[-axis_length / 2 + step_id * step, -axis_length / 2, 0],
                                  [-axis_length / 2 + step_id * step, axis_length / 2, 0]], dtype=np.float32))
        self.axes = np.array(axes)

    def plot(self, img, vertices, edges, head_dirs_3d, all_eye_midpoints_3d, all_eyes_3d, gaze_scale=50):
        global theta, phi
        img.fill(0)
        R = self._get_rotation(theta, phi)
        self._draw_axes(img, R)
        if len(edges) != 0:
            self._plot_edges(img, vertices, edges, R)
        if len(head_dirs_3d) != 0:
            self._plot_gaze(img, head_dirs_3d, all_eye_midpoints_3d, gaze_scale, R)
        fov_pyramids = self._plot_field_of_view(img, head_dirs_3d, all_eye_midpoints_3d, all_eyes_3d, gaze_scale, R)
        return fov_pyramids
    
    def clear(self, img):
        img.fill(0)
        R = self._get_rotation(theta, phi)
        self._draw_axes(img, R)
        
    def _plot_gaze(self, img, gaze_direction_vectors, gaze_origins, gaze_scale, R):
        for idx, gaze_origin in enumerate(gaze_origins):
            gaze_end = gaze_origin + gaze_direction_vectors[idx] * gaze_scale 
            gaze_origin_2d = np.dot(gaze_origin, R)
            gaze_end_2d = np.dot(gaze_end, R)
            gaze_origin_2d = gaze_origin_2d * self.scale + self.origin
            gaze_end_2d = gaze_end_2d * self.scale + self.origin
            # Convert the points to integers
            gaze_origin_2d_int = gaze_origin_2d.astype(int)
            
            gaze_end_2d_int = gaze_end_2d.astype(int)

            # Draw the line on the canvas
            cv2.arrowedLine(img, tuple(gaze_origin_2d_int), tuple(gaze_end_2d_int), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, str(idx), (gaze_origin_2d_int[0], gaze_origin_2d_int[1]-50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

    def _plot_field_of_view(self, img, gaze_direction_vectors, gaze_origins, all_eyes_3d, gaze_scale, R):
        h_fov = np.deg2rad(self.fov_angles[0])  # Horizontal field of view angle
        v_fov = np.deg2rad(self.fov_angles[1])  # Vertical field of view angle
        alpha = h_fov / 2  # Angle in radians
        beta = v_fov /2  # Angle in radians
        all_fov_pyramids = np.zeros((gaze_origins.shape[0],5, 3))
        for idx, gaze_origin in enumerate(gaze_origins):
            length_centroid = np.linalg.norm(gaze_direction_vectors[idx] * gaze_scale)
            base_side_a = np.tan(alpha) * length_centroid * 2
            base_side_b = np.tan(beta) * length_centroid * 2
             # Calculate the base center
            base_center = gaze_origin + gaze_direction_vectors[idx] * gaze_scale
            #base_points = np.zeros((4, 3))
            # 1 Calculate the direction of the line
            all_eyes_3d[idx]
            base_h_direction = all_eyes_3d[idx][1] - all_eyes_3d[idx][0]

            # Calculate the normalized direction of the line
            base_h_direction_unit = base_h_direction / np.linalg.norm(base_h_direction)

            # Calculate the line segment
            #base_h = np.vstack((base_center, base_center + base_h_direction_unit))

            # 2 Calculate the normal vector of the plane
            base_normal = gaze_direction_vectors[idx]

            # Define a point on the plane (base_h_point can be used)
            #base_point = base_h[0]

            # Define the plane by its normal and a point on the plane
            #base_plane = (base_normal, base_point)

            # 3
            # Calculate the cross product of base_h_direction_unit and base_normal to get base_v_direction
            base_v_direction = np.cross(base_h_direction_unit, base_normal)

            # Calculate the normalized direction of the line
            base_v_direction_unit = base_v_direction / np.linalg.norm(base_v_direction)

            # Calculate the line segment
            #base_v = np.vstack((base_center, base_center + base_v_direction_unit))
            # Plot the base plane
            #base_origin = base_plane[1]
            #base_normal_scaled = base_plane[0] * 10  # Scaling the normal vector for visualization
            #base_points = np.array([base_origin - base_normal_scaled, base_origin + base_normal_scaled])
            # 5 Calculate the corners of the base rectangle
            corner_1 = base_center + 0.5 * base_side_a * base_h_direction_unit + 0.5 * base_side_b * base_v_direction_unit
            corner_2 = base_center - 0.5 * base_side_a * base_h_direction_unit + 0.5 * base_side_b * base_v_direction_unit
            corner_3 = base_center - 0.5 * base_side_a * base_h_direction_unit - 0.5 * base_side_b * base_v_direction_unit
            corner_4 = base_center + 0.5 * base_side_a * base_h_direction_unit - 0.5 * base_side_b * base_v_direction_unit

            corner_1_2d = np.dot(corner_1, R) * self.scale + self.origin
            corner_2_2d = np.dot(corner_2, R) * self.scale + self.origin
            corner_3_2d = np.dot(corner_3, R) * self.scale + self.origin
            corner_4_2d = np.dot(corner_4, R) * self.scale + self.origin
            corner_1_2d_int = corner_1_2d.astype(int)
            corner_2_2d_int = corner_2_2d.astype(int)
            corner_3_2d_int = corner_3_2d.astype(int)
            corner_4_2d_int = corner_4_2d.astype(int)
            # Define the base rectangle by its corners
            base_rectangle_2d_int = np.vstack((corner_1_2d_int, corner_2_2d_int, corner_3_2d_int, corner_4_2d_int, corner_1_2d_int))
            # Draw the lines of the base rectangle
            gaze_origin_2d = np.dot(gaze_origin, R) * self.scale + self.origin
            # Convert the points to integers
            gaze_origin_2d_int = gaze_origin_2d.astype(int)
            for i in range(4):
                cv2.line(img, tuple(base_rectangle_2d_int[i]), tuple(base_rectangle_2d_int[i+1]), (255, 0, 255), 1, cv2.LINE_AA)
                cv2.line(img, tuple(base_rectangle_2d_int[i]), tuple(gaze_origin_2d_int), (255, 0, 255), 1, cv2.LINE_AA)
            # Draw the lines of the base
            #cv2.line(img, tuple(base_h[0].astype(int)), tuple(base_h[1].astype(int)), (0, 255, 255), 1, cv2.LINE_AA)
            #cv2.line(img, tuple(base_v[0].astype(int)), tuple(base_v[1].astype(int)), (0, 255, 255), 1, cv2.LINE_AA)
            all_fov_pyramids[idx] = [gaze_origin, corner_1, corner_2, corner_3, corner_4]
        
        return all_fov_pyramids
    
    def _draw_axes(self, img, R):
        axes_2d = np.dot(self.axes, R)
        axes_2d = axes_2d * self.scale + self.origin
        for axe in axes_2d:
            axe = axe.astype(int)
            cv2.line(img, tuple(axe[0]), tuple(axe[1]), (128, 128, 128), 1, cv2.LINE_AA)

    def _plot_edges(self, img, vertices, edges, R):
        vertices_2d = np.dot(vertices, R)
        edges_vertices = vertices_2d.reshape((-1, 2))[edges] * self.scale + self.origin
        for edge_vertices in edges_vertices:
            edge_vertices = edge_vertices.astype(int)
            cv2.line(img, tuple(edge_vertices[0]), tuple(edge_vertices[1]), (255, 255, 255), 1, cv2.LINE_AA)
    def _get_rotation(self, theta, phi):
        sin, cos = math.sin, math.cos
        return np.array([
            [ cos(theta),  sin(theta) * sin(phi)],
            [-sin(theta),  cos(theta) * sin(phi)],
            [ 0,                       -cos(phi)]
        ], dtype=np.float32)  # transposed

    @staticmethod
    def mouse_callback(event, x, y, flags, params):
        global previous_position, theta, phi, should_rotate, scale_dx, scale_dy
        if event == cv2.EVENT_LBUTTONDOWN:
            previous_position = [x, y]
            should_rotate = True
        if event == cv2.EVENT_MOUSEMOVE and should_rotate:
            theta += (x - previous_position[0]) / scale_dx * 6.2831  # 360 deg
            phi -= (y - previous_position[1]) / scale_dy * 6.2831 * 2  # 360 deg
            phi = max(min(3.1415 / 2, phi), -3.1415 / 2)
            previous_position = [x, y]
        if event == cv2.EVENT_LBUTTONUP:
            should_rotate = False
        if event == cv2.EVENT_RBUTTONUP:
            scale_dx += 10 * (x - previous_position[0])
            scale_dy += 10 * (y - previous_position[1])
            previous_position = [x, y]


body_edges = np.array(
    [[0, 1],  # neck - nose
     [1, 16], [16, 18],  # nose - l_eye - l_ear
     [1, 15], [15, 17],  # nose - r_eye - r_ear
     [0, 3], [3, 4], [4, 5],     # neck - l_shoulder - l_elbow - l_wrist
     [0, 9], [9, 10], [10, 11],  # neck - r_shoulder - r_elbow - r_wrist
     [6, 12],                     # l_hip - r_hip
     [0, 6], [6, 7], [7, 8],        # neck - l_hip - l_knee - l_ankle
     [0, 12], [12, 13], [13, 14]])    # neck - r_hip - r_knee - r_ankle
     
# eye_midpoint - gaze direction vector end 
"""
kpt_names = ['neck' 0, 'nose' 1, 'pelvis' 2,
                'l_sho' 3, 'l_elb' 4, 'l_wri' 5, 'l_hip' 6, 'l_knee' 7, 'l_ank' 8,
                'r_sho' 9, 'r_elb' 10, 'r_wri' 11, 'r_hip' 12, 'r_knee' 13, 'r_ank' 14,
                'r_eye' 15, 'l_eye' 16, 'r_ear' 17, 'l_ear' 18]
face_names = ['nose' 1, 'r_eye' 15, 'l_eye' 16, 'r_ear' 17, 'l_ear' 18]
"""
def draw_poses(img, poses_2d):
    for pose_id in range(len(poses_2d)):
        pose = np.array(poses_2d[pose_id][0:-1]).reshape((-1, 3)).transpose()
        was_found = pose[2, :] > 0
        for edge in body_edges:
            if was_found[edge[0]] and was_found[edge[1]]:
                cv2.line(img, tuple(pose[0:2, edge[0]].astype(int)), tuple(pose[0:2, edge[1]].astype(int)),
                         (255, 255, 0), 4, cv2.LINE_AA)
        for kpt_id in range(pose.shape[1]):
            if pose[2, kpt_id] != -1:
                cv2.circle(img, tuple(pose[0:2, kpt_id].astype(int)), 3, (0, 255, 255), -1, cv2.LINE_AA)
