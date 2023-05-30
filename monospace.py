import cv2

image = cv2.imread("people3.png")

table_data = '''
    +------------+----------+----------+----------+
    |   Observer |   Sees 0 |   Sees 1 |   Sees 2 |
    +============+==========+==========+==========+
    |          0 |        0 |        0 |        0 |
    +------------+----------+----------+----------+
    |          1 |        0 |        0 |        0 |
    +------------+----------+----------+----------+
    |          2 |        0 |        0 |        0 |
    +------------+----------+----------+----------+
'''

table_data = '''
+------------+----------+----------+----------+
|   Observer |   Sees 0 |   Sees 1 |   Sees 2 |
+============+==========+==========+==========+
|          0 |        0 |        0 |        0 |
+------------+----------+----------+----------+
|          1 |        0 |        0 |        0 |
+------------+----------+----------+----------+
|          2 |        0 |        0 |        0 |
+------------+----------+----------+----------+
'''

# Split the table data by newlines
table_rows = table_data.strip().split('\n')

# Define font settings
font_face = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
line_type = cv2.LINE_AA

# Define starting position
x = 10
y = 30
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
        cv2.putText(image, column, (x + col_idx * tab_width, y), font_face, font_scale,
                    (0, 0,255), thickness=font_thickness)

    y += line_height

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()