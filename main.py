import cv2
import os
import numpy as np
import copy
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image_dataset_from_directory
from tensorflow.image import convert_image_dtype
from tensorflow.data.experimental import AUTOTUNE
from keras.models import load_model

# fuctions
def click_event(event, x, y, flags, params): # in order to get the coordinates of the chessboard from the input camera image
    if event == cv2.EVENT_LBUTTONDOWN:       # LBUTTONDOWN - click the left mouse button
        persp_change_input_coords.append((x, y)) 

def convert_to_float(image, label):          # convert image from uint8 to float32
    image = convert_image_dtype(image, dtype=tf.float32)
    return image, label

if __name__ == "__main__":
    # constants
    chessboard_columns_names = 'abcdefgh'
    chessboard_rows_names = '12345678'
    straight_lines_intersection_pattern = np.zeros((3, 3, 3), dtype=np.uint8)
    straight_lines_intersection_pattern[:, :, 2] = np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]], dtype=np.uint8)
    intersection_region_side_len = 10
    chess_pieces_cnn_model = load_model('model_cnn.keras')
    chess_pieces_names = ["CG", "CK", "CS", "CP", "CH", "CW", "XX", "BG", "BK", "BS", "BP", "BH", "BW"]
    chess_pieces_scan_photos_path = './tiles_scan/photos'

    # variables
    persp_change_input_coords = []
    straight_lines_intersection_coords_matrix = np.zeros(shape=(9, 9), dtype=tuple)
    chessboard = [["XX" for _ in range(8)] for _ in range(8)]

    # camera options
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    # getting coords for the perspetive change - 2D homography
    cv2.namedWindow("Select chessboard")
    cv2.setMouseCallback("Select chessboard", click_event)
    while len(persp_change_input_coords) < 4:
        _, frame = cap.read()
        cv2.imshow("Select chessboard", frame) # the first coords - the top left corner, the second coords - the top right corner, the third coords - the right bottom corner, the fourth coords - left bottom corner
        cv2.waitKey(1)
    #cv2.imwrite("./photos/chessboard_view_from_camera.jpg", frame)
    cv2.destroyWindow("Select chessboard")

    # the perspective change matrix M
    top_view_first_width = int(np.sqrt(((persp_change_input_coords[1][0]-persp_change_input_coords[0][0])**2)+((persp_change_input_coords[1][1]-persp_change_input_coords[0][1])**2)))
    top_view_second_width = int(np.sqrt(((persp_change_input_coords[2][0]-persp_change_input_coords[3][0])**2)+((persp_change_input_coords[2][1]-persp_change_input_coords[3][1])**2)))
    top_view_width = max(top_view_first_width, top_view_second_width)

    top_view_first_height = int(np.sqrt(((persp_change_input_coords[0][0]-persp_change_input_coords[3][0])**2)+((persp_change_input_coords[0][1]-persp_change_input_coords[3][1])**2)))
    top_view_second_height = int(np.sqrt(((persp_change_input_coords[1][0]-persp_change_input_coords[2][0])**2)+((persp_change_input_coords[1][1]-persp_change_input_coords[2][1])**2)))
    top_view_height = max(top_view_first_height, top_view_second_height)
    
    persp_change_input_coords = np.float32(persp_change_input_coords)
    persp_change_top_view_coords = np.float32([[0, 0], [top_view_width-1, 0], [top_view_width-1, top_view_height-1], [0, top_view_height-1]])

    M = cv2.getPerspectiveTransform(persp_change_input_coords, persp_change_top_view_coords)

    # naming the chessboard tiles (for the tile name - view from camera dictionary)
    tiles_names = [f"{col}{row}" for row in chessboard_rows_names for col in chessboard_columns_names]

    # finding tiles
    _, frame = cap.read()

    # the perspective change - 2D homography
    top_view = cv2.warpPerspective(frame, M, (top_view_width, top_view_height), flags=cv2.INTER_LINEAR)

    # light equalization - CLAHE algorithm
    lab = cv2.cvtColor(top_view, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l)
    clahe_lab = cv2.merge((clahe_l, a, b))
    clahe_top_view = cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2BGR)
    #cv2.imwrite("./photos/CLAHE.jpg", clahe_top_view)
    
    # smoothing
    gaussian = cv2.GaussianBlur(clahe_top_view, ksize=(7, 7), sigmaX=3.0)
    #cv2.imwrite("./photos/smoothing.jpg", gaussian)

    # grayscale
    gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    edges = cv2.Canny(gray, threshold1=60.0, threshold2=150.0)
    #cv2.imwrite("./photos/canny.jpg", edges)

    # finding horizontal and vertical lines - Hough transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=50)
    top_view_with_hor_and_ver_lines = copy.deepcopy(top_view) # in order to draw lines on the top_view image and not change the top_view image itself
    lines_view = np.zeros(np.shape(top_view), dtype=np.uint8)
    if lines is not None:
        for r_and_theta in lines:
            r, theta = r_and_theta[0]
            if np.isclose(theta, 0, atol=0.01) or np.isclose(theta, np.pi, atol=0.01): # finding vertical lines
                x1, x2 = int(r), int(r)
                y1, y2 = 0, top_view_height-1
            elif np.isclose(theta, np.pi/2, atol=0.01): # finding horizontal lines
                x1, x2 = 0, top_view_width-1
                a = -np.cos(theta)/np.sin(theta)
                b = r/np.sin(theta)
                y1 = int(a*x1+b)
                y2 = int(a*x2+b)
            else:
                continue
            # limiting lines
            y1 = max(0, min(top_view_height-1, y1))
            y2 = max(0, min(top_view_height-1, y2))
            cv2.line(lines_view, (x1, y1), (x2, y2), (0, 0, 255), 1) # drawing red lines - thickness: 1
            cv2.line(top_view_with_hor_and_ver_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #cv2.imwrite("./photos/horizontal_and_vertical_lines_on_chessboard.jpg", top_view_with_hor_and_ver_lines)

    # finding intersections of the lines (finding tiles)
    intersection_view = np.zeros(np.shape(top_view), dtype=np.uint8)
    intersection_row = 0
    intersection_col = 0
    for i in range(1, top_view_height-1):
        for j in range(1, top_view_width-1):
            window = lines_view[i-1:i+2, j-1:j+2, :]
            if np.all(window == straight_lines_intersection_pattern): # finding intesection
                if np.any(intersection_view[max(i-intersection_region_side_len, 0):min(i+intersection_region_side_len+1, top_view_height), max(j-intersection_region_side_len, 0):min(j+intersection_region_side_len+1, top_view_width), 2] == 255): # there are no intersection in the neighbourhood of the potential new intersection 
                    continue
                else:
                    intersection_view[i, j, 2] = 255
                    straight_lines_intersection_coords_matrix[intersection_row, intersection_col] = (i, j)
                    intersection_col += 1
                    if intersection_col > 8:
                        intersection_row += 1
                        intersection_col = 0
                    if intersection_row > 8:
                        break
        if intersection_row > 8:
            break

    cv2.namedWindow("View")
    while True:
        _, frame = cap.read()

        # the perspective change - 2D homography
        top_view = cv2.warpPerspective(frame, M, (top_view_width, top_view_height), flags=cv2.INTER_LINEAR)

        # viewing
        cv2.imshow("View", top_view)

        # the dictionary of the tiles view
        tiles_view = []
        for i in range(8):
            for j in range(8):
                y_first, x_first = straight_lines_intersection_coords_matrix[8-i-1, j]
                _, x_last= straight_lines_intersection_coords_matrix[8-i-1, j+1]
                y_last, _ = straight_lines_intersection_coords_matrix[8-i, j]
                tile = np.array(top_view[y_first:y_last, x_first:x_last, :], dtype=np.uint8)
                tiles_view.append(tile)
        tiles = {names:views for (names, views) in zip(tiles_names, tiles_view)} 

        key = cv2.waitKey(1)
        if key == 97: # 'a'
            #cv2.imwrite("./photos/chessboard_pattern_.jpg", top_view)
            # chess game registration
            tile_order = iter(tiles)
            # scanning all the tiles to predict which chess piece is on it
            for i in range(8*8):
                # cleaning the folder where the tile scan will be stored
                for photo in os.listdir(chess_pieces_scan_photos_path):
                    tile_scan_path = os.path.join(chess_pieces_scan_photos_path, photo)
                    if os.path.isfile(tile_scan_path):
                        os.remove(tile_scan_path)
                tile_name = next(tile_order)
                # scanning
                cv2.imwrite(chess_pieces_scan_photos_path + "/scan.jpg", tiles[tile_name])
                # load scan
                scan = image_dataset_from_directory(
                    directory='./tiles_scan',
                    image_size=[224, 224], # resizing the images to the given shape
                    interpolation='nearest', # the interpolation type while resizing
                    batch_size=1, # the number of images loaded at the same time
                    shuffle=False # randomize
                )
                scan = (
                    scan
                    .map(convert_to_float) # mapping (convert from uint8 to float)
                    .cache() # use cache to speed up iterations
                    .prefetch(buffer_size=AUTOTUNE) # prefetch the new batch while the earlier one is still being processed
                )
                tile_predict = chess_pieces_cnn_model.predict(scan) # returns the propabilities that the shown piece belongs to given class
                # finding the class of the shown piece
                figure_idx = np.argmax(tile_predict, axis=1)
                chessboard[7-i//8][i%8] = chess_pieces_names[figure_idx[0]]
            # showing the register game
            for i in range(8):
                print(chessboard[i])
                print("\n")
        if key == 27: # 'Esc'
            break

    # cleaning
    cv2.destroyAllWindows()
    cap.release()
