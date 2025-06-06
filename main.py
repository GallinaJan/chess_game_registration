import cv2
import numpy as np
import matplotlib.pyplot as plt

# global variables
persp_change_coords = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        persp_change_coords.append((x, y)) 

if __name__ == "__main__":
    # constants
    tile_columns_names = 'abcdefgh'
    tile_rows_names = '12345678'
    lines_intersection_pattern = np.zeros((3, 3, 3), dtype=np.uint8)
    lines_intersection_pattern[:, :, 2] = np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]], dtype=np.uint8)
    intersection_dist = 10
    first_iter = True

    # variables
    tiles_found = False
    intersection_coords = np.zeros(shape=(9, 9), dtype=tuple)

    # camera options
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    # getting coords for the perspetive change
    cv2.namedWindow("Select chessboard")
    cv2.setMouseCallback("Select chessboard", click_event)

    while len(persp_change_coords) < 4:
        # frame
        _, frame = cap.read()
        cv2.imshow("Select chessboard", frame) # the first coords - the top left corner, the second coords - the top right corner, the third coords - the right bottom corner, the fourth coords - left bottom corner
        cv2.waitKey(1)

    cv2.destroyWindow("Select chessboard")

    # naming the chessboard tiles
    tiles_names = [f"{col}{row}" for row in tile_rows_names for col in tile_columns_names]

    # the perspective change matrix M
    width_top_view_first_side = int(np.sqrt(((persp_change_coords[1][0]-persp_change_coords[0][0])**2)+((persp_change_coords[1][1]-persp_change_coords[0][1])**2)))
    width_top_view_second_side = int(np.sqrt(((persp_change_coords[2][0]-persp_change_coords[3][0])**2)+((persp_change_coords[2][1]-persp_change_coords[3][1])**2)))
    width_top_view_max = max(width_top_view_first_side, width_top_view_second_side)
    
    height_top_view_first_side = int(np.sqrt(((persp_change_coords[0][0]-persp_change_coords[3][0])**2)+((persp_change_coords[0][1]-persp_change_coords[3][1])**2)))
    height_top_view_second_side = int(np.sqrt(((persp_change_coords[1][0]-persp_change_coords[2][0])**2)+((persp_change_coords[1][1]-persp_change_coords[2][1])**2)))
    height_top_view_max = max(height_top_view_first_side, height_top_view_second_side)
    
    persp_change_taken_coords = np.float32(persp_change_coords)
    persp_change_coords_top_view = np.float32([[0, 0], [width_top_view_max-1, 0], [width_top_view_max-1, height_top_view_max-1], [0, height_top_view_max-1]])

    M = cv2.getPerspectiveTransform(persp_change_taken_coords, persp_change_coords_top_view)

    while True:
        # frame
        _, frame = cap.read()

        # perspective change
        top_view = cv2.warpPerspective(frame, M, (width_top_view_max, height_top_view_max), flags=cv2.INTER_LINEAR)

        # light equalization - CLAHE algorithm
        lab = cv2.cvtColor(top_view, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        clahe_l = clahe.apply(l)
        clahe_lab = cv2.merge((clahe_l, a, b))
        clahe_top_view = cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2BGR)
        
        # smoothing
        gaussian = cv2.GaussianBlur(clahe_top_view, (7, 7), 3)

        # finding tiles
        if not tiles_found:
            gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 60, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
            lines_view = np.zeros(np.shape(top_view), dtype=np.uint8)
            if lines is not None:
                for r_and_theta in lines:
                    r, theta = r_and_theta[0]
                    if np.isclose(theta, 0, atol=0.01) or np.isclose(theta, np.pi, atol=0.01):
                        x1, x2 = int(r), int(r)
                        y1, y2 = 0, height_top_view_max-1
                    elif np.isclose(theta, np.pi/2, atol=0.01):
                        x1, x2 = 0, width_top_view_max-1
                        a = -np.cos(theta)/np.sin(theta)
                        b = r/np.sin(theta)
                        y1 = int(a*x1+b)
                        y2 = int(a*x2+b)
                    else:
                        continue
                    # limiting lines
                    y1 = max(0, min(height_top_view_max-1, y1))
                    y2 = max(0, min(height_top_view_max-1, y2))
                    cv2.line(lines_view, (x1, y1), (x2, y2), (0, 0, 255), 1)
                intersection_view = np.zeros(np.shape(top_view), dtype=np.uint8)
                intersection_row = 0
                intersection_col = 0
                for i in range(1, height_top_view_max-1):
                    for j in range(1, width_top_view_max-1):
                        window = lines_view[i-1:i+2, j-1:j+2, :]
                        if np.all(window == lines_intersection_pattern):
                            if np.any(intersection_view[max(i-intersection_dist, 0):min(i+intersection_dist+1, height_top_view_max), max(j-intersection_dist, 0):min(j+intersection_dist+1, width_top_view_max), 2] == 255):
                                continue
                            else:
                                intersection_view[i, j, 2] = 255
                                intersection_coords[intersection_row, intersection_col] = (i, j)
                                intersection_col += 1
                                if intersection_col > 8:
                                    intersection_row += 1
                                    intersection_col = 0
                                if intersection_row > 8:
                                    break
                    if intersection_row > 8:
                        break
                if np.sum(intersection_view[:, :, 2]) != 9*9*255:
                    continue
                tiles_found = True

        # the dictionary of the tiles view
        tiles_view = []

        for i in range(8):
            for j in range(8):
                y_first, x_first = intersection_coords[8-i-1, j]
                _, x_last= intersection_coords[8-i-1, j+1]
                y_last, _ = intersection_coords[8-i, j]
                tile = np.array(top_view[y_first:y_last+1, x_first:x_last, :], dtype=np.uint8)
                tiles_view.append(tile)

        tiles = {names:views for (names, views) in zip(tiles_names, tiles_view)} 

        # viewing
        if first_iter:
            keys = iter(tiles)
            tile_name = next(keys)
            first_iter = False
        cv2.imshow("View", tiles[tile_name])
        key = cv2.waitKey(1)  # wait for the key ('Esc')
        if key == 27:
            break
        elif key == 97: # 'a'
            cv2.imwrite("./dataset/test/black_knight/black_knight_"+tile_name+".jpg", tiles[tile_name])
            if tile_name == "h8":
                break
            else:
                tile_name = next(keys)

    # cleaning
    cap.release()
    cv2.destroyAllWindows()
