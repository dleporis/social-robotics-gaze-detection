import cv2
import numpy as np
import networkx as nx

def _plot_adjecency_graph(image, adj_matrix):

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
    scale_factor = min(graph_image_size) / (2.5 * graph_size)

    # Define an array of 10 different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 128)]

    # Draw the graph on the image
    for edge in graph.edges:
        start_node = edge[0]
        end_node = edge[1]
        start_point = (int(layout[start_node][0] * scale_factor + graph_image_size[0] / 2),
                       int(layout[start_node][1] * scale_factor + graph_image_size[1] / 2))
        end_point = (int(layout[end_node][0] * scale_factor + graph_image_size[0] / 2),
                     int(layout[end_node][1] * scale_factor + graph_image_size[1] / 2))
        cv2.arrowedLine(graph_image, start_point, end_point, colors[start_node % len(colors)], 2, tipLength=0.25)

    for node, position in layout.items():
        center = (int(position[0] * scale_factor + graph_image_size[0] / 2),
                  int(position[1] * scale_factor + graph_image_size[1] / 2))
        cv2.circle(graph_image, center, 10, colors[node % len(colors)], -1)
        cv2.putText(graph_image, str(node), (center[0] + 5, center[1] + 5), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0), 2)

    # Calculate the offset values to align the images
    img_size = image.shape[:2]
    graph_size = graph_image.shape[:2]
    x_offset = img_size[1] - graph_size[1]
    y_offset = img_size[0] - graph_size[0]

    # Paste the graph image on top of the original image
    image[y_offset:y_offset + graph_size[0], x_offset:x_offset + graph_size[1]] = graph_image


    # Display the resulting image
    cv2.imshow('Graph', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

adj_matrix = np.array([[0, 1, 1, 0, 1],
                       [1, 0, 0, 0, 1],
                       [0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]])

img_size = (600, 600)
img = np.zeros((img_size[0], img_size[1], 3), np.uint8) * 255
_plot_adjecency_graph(img, adj_matrix)
