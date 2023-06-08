import math

import cv2
import numpy as np
#from PIL import Image, ImageFont, ImageDraw
import contextlib
import networkx as nx

previous_position = []
theta, phi = 3.1415/4, -3.1415/6
should_rotate = False
scale_dx = 800
scale_dy = 800


class Plotter3d:
    SKELETON_EDGES = np.array([[11, 10], [10, 9], [9, 0], [0, 3], [3, 4], [4, 5], [0, 6], [6, 7], [7, 8], [0, 12],
                               [12, 13], [13, 14], [0, 1], [1, 15], [15, 16], [1, 17], [17, 18], [6, 12] ]) # 19 eye_midpoint - 20 gaze direction vector end
    def __init__(self, canvas_size, cone_angle, origin=(0.5, 0.5), scale=1):
        self.canvas_size = canvas_size
        self.origin = np.array([origin[1] * canvas_size[1], origin[0] * canvas_size[0]], dtype=np.float32)  # x, y
        self.scale = np.float32(scale)
        self.theta = 0
        self.phi = 0
        self.fov_angles = [120, 60]
        self.cone_angle = cone_angle
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
        self.person_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
                (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128)]

    def plot(self, img, vertices, edges, head_dirs_3d, all_eye_midpoints_3d, adjacency_table, adjacency_matrix, gaze_scale=50):
        global theta, phi
        """
        adjacency_matrix = np.array([[0, 1, 1, 0, 1],
                       [1, 0, 0, 0, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])
        """
        img.fill(0)
        R = self._get_rotation(theta, phi)
        self._draw_axes(img, R)
        if len(edges) != 0:
            self._plot_edges(img, vertices, edges, R)
        if len(head_dirs_3d) != 0:
            self._plot_gaze(img, head_dirs_3d, all_eye_midpoints_3d, gaze_scale, R)
            self._plot_adjecency_graph(img, adjacency_matrix)
            self._write_table(img, adjacency_table)
    

    def clear(self, img):
        img.fill(0)
        R = self._get_rotation(theta, phi)
        self._draw_axes(img, R)
        
    def _plot_gaze(self, img, gaze_direction_vectors, gaze_origins, gaze_scale, R):
        for idx, gaze_origin in enumerate(gaze_origins):
            gaze_dir_vectors_magnitude = np.linalg.norm(gaze_direction_vectors[idx])
            #print("gaze_dir_vectors_magnitude")
            #print(gaze_dir_vectors_magnitude)
            gaze_end = gaze_origin + gaze_direction_vectors[idx] * gaze_scale
            gaze_direction_vector_scaled = gaze_origin-gaze_end
            gaze_magnitude = np.linalg.norm(gaze_direction_vector_scaled)
            gaze_direction_norm = gaze_direction_vectors[idx] / gaze_magnitude
            #print("gaze_magnitude")
            #print(gaze_magnitude)
            #print("self.cone_angle")
            #print(self.cone_angle)
            """
            # convert to 2d
            gaze_origin_2d = np.dot(gaze_origin, R)
            gaze_end_2d = np.dot(gaze_end, R)
            gaze_origin_2d = gaze_origin_2d * self.scale + self.origin
            gaze_end_2d = gaze_end_2d * self.scale + self.origin
            
            # Convert the points to integers
            gaze_origin_2d_int = gaze_origin_2d.astype(int)
            gaze_origin_2d_int
            gaze_end_2d_int = gaze_end_2d.astype(int)
            """
            # all in one convert to 2d integers
            gaze_origin_2d_int = self._float3d_to_2d_int( gaze_origin, R)
            gaze_end_2d_int = self._float3d_to_2d_int( gaze_end, R)

            # Draw the gaze line on the canvas
            cv2.arrowedLine(img, tuple(gaze_origin_2d_int), tuple(gaze_end_2d_int), (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, str(idx), (gaze_origin_2d_int[0], gaze_origin_2d_int[1]-50), cv2.FONT_HERSHEY_COMPLEX, 1, self.person_colors[idx % len(self.person_colors)])
            
            # plot adjecency matrix
            #cv2.putText(img, adjacency_table, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0),  cv2.LINE_AA)
            # plot the cone
            cone_base_radius = gaze_magnitude * np.tan(self.cone_angle)
            cone_axis_normalized = gaze_direction_vector_scaled / gaze_magnitude
            #print("gaze_direction_norm")
            #print(gaze_direction_norm)
            #print("cone_axis_normalized")
            #print(cone_axis_normalized)
            #perpendicular_vector = np.array([-cone_axis_normalized[1], cone_axis_normalized[0], 0])
            #perpendicular_vector_normalized = perpendicular_vector / np.linalg.norm(perpendicular_vector)
            resolution = 12
            # Generate points for the base circle centered at the origin
            theta = np.linspace(0, 2 * np.pi, resolution)
            circle_points = np.column_stack([cone_base_radius * np.cos(theta), cone_base_radius * np.sin(theta), np.zeros_like(theta)])
            # Calculate the rotation axis and angle to align the circle with gaze_direction
            rotation_axis = np.cross([0, 0, 1], gaze_direction_norm)
            rotation_angle = np.arccos(np.dot([0, 0, 1], gaze_direction_norm))
            # Apply the rotation to the circle points
            rotated_circle_points = np.dot(circle_points, self._rotation_matrix(rotation_axis, rotation_angle))
            # Translate the circle points to align with gaze_end
            translated_circle_points = rotated_circle_points + gaze_end
            #print("translated_circle_points")
            #print(translated_circle_points)
            translated_circle_points_2d_int = []
            for circle_point in translated_circle_points:
                circle_point_2d_int = self._float3d_to_2d_int(circle_point, R)
                # plot lines from origing to the cone base circle
                
                cv2.line(img, tuple(gaze_origin_2d_int), circle_point_2d_int, self.person_colors[idx % len(self.person_colors)], 1, cv2.LINE_AA)
                translated_circle_points_2d_int.append(tuple(circle_point_2d_int))
            #print("translated_circle_points_2d_int")
            #print(translated_circle_points_2d_int)

            for index, point_2d in enumerate(translated_circle_points_2d_int):
                cv2.line(img, translated_circle_points_2d_int[index], translated_circle_points_2d_int[(index+1) % len(translated_circle_points_2d_int)], self.person_colors[idx % len(self.person_colors)], 1, cv2.LINE_AA)
                

    def _write_table(self, img, table_data):
       
        # Split the table data by newlines
        table_rows = table_data.strip().split('\n')

        # Define font settings
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        font_thickness = 2

        # Define starting position
        width = self.canvas_size[0]
        height = self.canvas_size[1]
        x = 0 
        y = int(height * 0.75) 
        line_height = 20

        # Tab settings
        tab_width = 120  # Adjust this value to control the tab spacing

        # Draw the table data
        for row in table_rows:
            # Split row by tabs
            if '+' in row:
                columns = row.split('+')

            else:
                columns = row.split('|')

            # Draw each column
            for col_idx, column in enumerate(columns):
                cv2.putText(img, column, (x + col_idx * tab_width, y), font_face, font_scale,
                            (0, 0, 255), thickness=font_thickness)

            y += line_height

        


    def _float3d_to_2d_int(self, input3d, R):
        output2d = np.dot(input3d, R) * self.scale + self.origin
        output2d_int = output2d.astype(int)
        return output2d_int
    
    def _rotation_matrix(self, axis, angle):
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
    
    def _plot_field_of_view_pyramids(self, img, gaze_direction_vectors, gaze_origins, all_eyes_3d, gaze_scale, R):
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
    
    def _plot_adjecency_graph(self, image, adj_matrix):
        
        """
        # Sample adjacency matrix
        adj_matrix = np.array([[0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


        adj_matrix = np.array([[0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        """

        """
        Plot the adjacency graph on the given image.

        Args:
        - image: The input image.
        - adj_matrix: The adjacency matrix representing the graph.
        """

        # Create a networkx graph from the adjacency matrix
        graph = nx.DiGraph(adj_matrix)

        # Perform layout for the graph
        layout = nx.kamada_kawai_layout(graph)

        # Set image size and create an empty image
        graph_image_size = (300, 300)
        graph_image = np.ones((graph_image_size[0], graph_image_size[1], 3), np.uint8) * 255

        # Calculate the scaling factor based on the graph size and image size
        graph_size = max(max(abs(pos[0]), abs(pos[1])) for pos in layout.values())
        prev_scale_factor = 120
        scale_factor = min(graph_image_size) / (2.5 * graph_size)
        if math.isinf(scale_factor):
            scale_factor = prev_scale_factor
        # Draw the graph on the image
        for edge in graph.edges:
            start_node = edge[0]
            end_node = edge[1]
            start_point = (int(layout[start_node][0] * scale_factor + graph_image_size[0] / 2),
                        int(layout[start_node][1] * scale_factor + graph_image_size[1] / 2))
            end_point = (int(layout[end_node][0] * scale_factor + graph_image_size[0] / 2),
                        int(layout[end_node][1] * scale_factor + graph_image_size[1] / 2))
            cv2.arrowedLine(graph_image, start_point, end_point, self.person_colors[start_node % len(self.person_colors)], 2, tipLength=0.25)

        for node, position in layout.items():
            center = (int(position[0] * scale_factor + graph_image_size[0] / 2),
                    int(position[1] * scale_factor + graph_image_size[1] / 2))
            cv2.circle(graph_image, center, 10, self.person_colors[node % len(self.person_colors)], -1)
            cv2.putText(graph_image, str(node), (center[0] + 5, center[1] + 5), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)

        # Calculate the offset values to align the images
        img_size = image.shape[:2]
        graph_size = graph_image.shape[:2]
        x_offset = img_size[1] - graph_size[1]
        y_offset = img_size[0] - graph_size[0]

        # Paste the graph image on top of the original image
        image[y_offset:y_offset + graph_size[0], x_offset:x_offset + graph_size[1]] = graph_image

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
